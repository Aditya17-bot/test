import json
import os
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st
from fpdf import FPDF
import py3Dmol

from src.logic_engine import (
    load_rare_disease_catalog,
    resolve_uniprot_id,
    get_af_structure_url,
    get_protein_anomalies_cached,
    detect_structural_hotspots,
    download_pdb_locally,
    extract_plddt_from_pdb,
)

st.set_page_config(page_title="Structural Anomaly Detector", layout="wide")

st.title("Rare-Disease Structural Anomaly Detector")
st.markdown("---")

st.sidebar.header("Search Gateway")
demo_mode = st.sidebar.checkbox("Demo mode (instant)", value=False)
use_cache = st.sidebar.checkbox("Use cached anomalies", value=True)
simulate_hotspot = st.sidebar.checkbox("Simulate hotspot (preview)", value=False)

try:
    disease_map = load_rare_disease_catalog("data/references/en_product6.xml")
    selected_disease = st.sidebar.selectbox(
        "Select Rare Disease", options=list(disease_map.keys())
    )
    default_gene = disease_map[selected_disease]
except Exception:
    st.sidebar.warning("Orphanet Catalog not found.")
    default_gene = "SOD1"

gene_symbol = st.sidebar.text_input("Gene Symbol / UniProt ID", value=default_gene)
compare_genes = st.sidebar.multiselect(
    "Compare genes (optional)",
    options=list(disease_map.values()) if "disease_map" in locals() else [],
    default=[],
)
download_btn = st.sidebar.button("Download PDB")
run_btn = st.sidebar.button("Run Anomaly Detection", type="primary")
compare_btn = st.sidebar.button("Run Comparison")

high_risk_threshold = 0.6
dbscan_eps = 10.0
dbscan_min_samples = 3


def get_cluster_palette() -> List[str]:
    return [
        "#FF3B30",
        "#FF9500",
        "#FFCC00",
        "#34C759",
        "#32ADE6",
        "#5856D6",
        "#AF52DE",
    ]


def build_cluster_legend(hotspots: pd.DataFrame) -> pd.DataFrame:
    if hotspots.empty:
        return pd.DataFrame(columns=["cluster_id", "color", "residues"])

    palette = get_cluster_palette()
    clusters = (
        hotspots[hotspots["cluster_id"] >= 0]
        .groupby("cluster_id")["residue_num"]
        .apply(list)
        .reset_index()
        .sort_values("cluster_id")
    )
    clusters["color"] = [
        palette[i % len(palette)] for i in range(len(clusters))
    ]
    clusters["residues"] = clusters["residue_num"].apply(
        lambda vals: ", ".join(map(str, vals))
    )
    return clusters[["cluster_id", "color", "residues"]]


def resolve_local_pdb(uid: str) -> Optional[str]:
    candidates = []
    if os.path.isdir("data/alphafold"):
        for name in os.listdir("data/alphafold"):
            if name.lower().endswith(".pdb") and uid.lower() in name.lower():
                candidates.append(os.path.join("data", "alphafold", name))
    return candidates[0] if candidates else None


def render_plddt_py3dmol(
    pdb_path: str,
    plddt_df: pd.DataFrame,
    hotspots: pd.DataFrame,
    anomaly_emphasis: bool = False,
    pdb_text_override: Optional[str] = None,
) -> str:
    if pdb_text_override is not None:
        pdb_text = pdb_text_override
    else:
        with open(pdb_path, "r", encoding="utf-8", errors="replace") as handle:
            pdb_text = handle.read()

    view = py3Dmol.view(width=760, height=680)
    view.addModel(pdb_text, "pdb")
    view.setBackgroundColor("white")

    if anomaly_emphasis:
        view.setStyle({}, {"cartoon": {"color": "#CCCCCC", "opacity": 0.7}})
    elif plddt_df is None or plddt_df.empty:
        view.setStyle({}, {"cartoon": {"color": "#B0B0B0"}})
    else:
        bands = {
            "very_high": {"color": "#1f77b4", "min": 90, "max": 101},
            "high": {"color": "#7fc7ff", "min": 70, "max": 90},
            "low": {"color": "#ffbf00", "min": 50, "max": 70},
            "very_low": {"color": "#d62728", "min": 0, "max": 50},
        }
        for band in bands.values():
            residues = plddt_df[
                (plddt_df["plddt"] >= band["min"]) & (plddt_df["plddt"] < band["max"])
            ]["residue_num"].tolist()
            if residues:
                view.setStyle(
                    {"resi": residues},
                    {"cartoon": {"color": band["color"]}},
                )

    if hotspots is not None and not hotspots.empty:
        anomaly_residues = (
            hotspots[hotspots["cluster_id"] >= 0]["residue_num"].unique().tolist()
        )
        if anomaly_residues:
            view.setStyle(
                {"resi": anomaly_residues},
                {"cartoon": {"color": "#ff2d55", "opacity": 1.0}},
            )
            view.setStyle(
                {"resi": anomaly_residues},
                {
                    "stick": {"color": "#ff2d55", "radius": 0.6 if not anomaly_emphasis else 1.1}
                },
            )
            if anomaly_emphasis:
                view.addSurface(
                    py3Dmol.VDW,
                    {"opacity": 0.8, "color": "#ff2d55"},
                    {"resi": anomaly_residues},
                )
                view.setStyle(
                    {"resi": anomaly_residues},
                    {"sphere": {"color": "#ff2d55", "radius": 1.4}},
                )
        elif anomaly_emphasis:
            # Keep the base visible if there are no anomalies to highlight
            view.setStyle({}, {"cartoon": {"color": "#B0B0B0", "opacity": 0.8}})

    view.zoomTo()
    return view._make_html()


def build_deformed_pdb_text(pdb_path: str, anomaly_residues: List[int], magnitude: float = 3.0) -> str:
    if not pdb_path or not os.path.exists(pdb_path) or not anomaly_residues:
        return ""

    try:
        with open(pdb_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()

        coords = []
        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except Exception:
                    continue

        if not coords:
            return "".join(lines)

        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)

        residue_set = set(anomaly_residues)
        new_lines = []
        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    res_num = int(line[22:26].strip())
                except Exception:
                    res_num = None

                if res_num in residue_set:
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        dx = x - cx
                        dy = y - cy
                        dz = z - cz
                        norm = (dx * dx + dy * dy + dz * dz) ** 0.5 or 1.0
                        x2 = x + magnitude * dx / norm
                        y2 = y + magnitude * dy / norm
                        z2 = z + magnitude * dz / norm
                        line = f"{line[:30]}{x2:8.3f}{y2:8.3f}{z2:8.3f}{line[54:]}"
                    except Exception:
                        pass

            new_lines.append(line)

        return "".join(new_lines)
    except Exception:
        return ""


def assign_plddt_band(plddt: float) -> str:
    if plddt >= 90:
        return "Very high (pLDDT >= 90)"
    if plddt >= 70:
        return "High (70 <= pLDDT < 90)"
    if plddt >= 50:
        return "Low (50 <= pLDDT < 70)"
    return "Very low (pLDDT < 50)"


def build_report(uid: str, protein_df: pd.DataFrame, hotspots: pd.DataFrame) -> Dict[str, object]:
    gii_score = float(protein_df["am_pathogenicity"].mean()) if not protein_df.empty else 0.0
    num_clusters = (
        int(hotspots[hotspots["cluster_id"] >= 0]["cluster_id"].nunique())
        if not hotspots.empty
        else 0
    )
    status = "STABLE"
    if gii_score > 0.6 or num_clusters > 0:
        status = "CRITICAL"
    elif gii_score > 0.4:
        status = "WARNING"

    return {
        "uniprot_id": uid,
        "instability_index": gii_score,
        "anomaly_clusters": num_clusters,
        "status": status,
    }


def report_to_pdf(report: Dict[str, object]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Rare-Disease Structural Anomaly Report", ln=True)
    pdf.ln(4)
    for key, value in report.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=True)
    pdf_bytes = pdf.output(dest="S")
    return bytes(pdf_bytes)

if download_btn and not demo_mode:
    uid = resolve_uniprot_id(gene_symbol)
    if not uid:
        st.sidebar.error("Could not resolve UniProt ID.")
    else:
        pdb_url = get_af_structure_url(uid)
        if not pdb_url:
            st.sidebar.error("Could not find AlphaFold PDB URL.")
        else:
            os.makedirs(os.path.join("data", "alphafold"), exist_ok=True)
            local_pdb = download_pdb_locally(pdb_url, uid)
            if local_pdb:
                st.sidebar.success(f"Downloaded to {local_pdb}")
            else:
                st.sidebar.error("Download failed. Check connection.")

if compare_btn:
    if demo_mode:
        st.sidebar.info("Comparison is for real data only. Disable Demo mode.")
    else:
        rows = []
        for gene in compare_genes:
            uid = resolve_uniprot_id(gene)
            if not uid:
                continue
            am_file = "data/raw/AlphaMissense_hg38.tsv.gz"
            df = get_protein_anomalies_cached(uid, am_file) if use_cache else get_protein_anomalies_cached(uid, am_file)
            if df.empty:
                continue
            report = build_report(uid, df, pd.DataFrame())
            rows.append(report)
        if rows:
            st.subheader("Gene Comparison")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No comparison results (ensure cache exists or genes are valid).")

if run_btn:
    with st.spinner(f"Analyzing {gene_symbol}..."):
        if demo_mode:
            uid = "P13569"
        else:
            uid = resolve_uniprot_id(gene_symbol)

        if not uid:
            st.error(f"Could not find UniProt ID for {gene_symbol}")
        else:
            if demo_mode:
                am_file = "data/demo/demo_anomalies.csv"
                protein_df = pd.read_csv(am_file)
            else:
                am_file = "data/raw/AlphaMissense_hg38.tsv.gz"
                if use_cache:
                    protein_df = get_protein_anomalies_cached(uid, am_file)
                else:
                    protein_df = get_protein_anomalies_cached(uid, am_file)

            if protein_df.empty:
                st.warning(f"No anomaly data found for {uid}")
            else:
                if demo_mode:
                    local_pdb = "data/demo/sample.pdb"
                else:
                    local_candidate = resolve_local_pdb(uid)
                    if local_candidate and os.path.exists(local_candidate):
                        local_pdb = local_candidate
                    else:
                        pdb_url = get_af_structure_url(uid)
                        local_pdb = download_pdb_locally(pdb_url, uid)

                plddt_df = (
                    extract_plddt_from_pdb(local_pdb)
                    if local_pdb and os.path.exists(local_pdb)
                    else pd.DataFrame()
                )

                high_risk_df = protein_df[
                    protein_df["am_pathogenicity"] > high_risk_threshold
                ]
                hotspots = detect_structural_hotspots(
                    local_pdb,
                    high_risk_df,
                    eps=float(dbscan_eps),
                    min_samples=int(dbscan_min_samples),
                )
                simulated_hotspots = False
                if simulate_hotspot and (hotspots is None or hotspots.empty):
                    residue_pool = []
                    if not plddt_df.empty:
                        residue_pool = plddt_df["residue_num"].tolist()
                    if protein_df is not None and "residue_num" in protein_df.columns:
                        residue_pool = sorted(
                            set(residue_pool).intersection(set(protein_df["residue_num"].tolist()))
                        ) or residue_pool
                    if len(residue_pool) >= 3:
                        mid = len(residue_pool) // 2
                        window = residue_pool[max(0, mid - 2) : min(len(residue_pool), mid + 4)]
                        if len(window) >= 3:
                            hotspots = pd.DataFrame(
                                {"residue_num": window, "cluster_id": [0] * len(window)}
                            )
                            simulated_hotspots = True

                st.subheader("3D Anomaly Map (Side-by-Side)")
                if not local_pdb:
                    st.error(
                        "3D Structure could not be downloaded. Check internet or SSL settings."
                    )
                else:
                    deformed_pdb_text = ""
                    if not hotspots.empty:
                        anomaly_residues = (
                            hotspots[hotspots["cluster_id"] >= 0]["residue_num"]
                            .unique()
                            .tolist()
                        )
                        deformed_pdb_text = build_deformed_pdb_text(
                            local_pdb, anomaly_residues, magnitude=3.5
                        )
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.caption("Base (pLDDT only)")
                        st.components.v1.html(
                            render_plddt_py3dmol(local_pdb, plddt_df, pd.DataFrame()),
                            height=650,
                        )
                    with v_col2:
                        st.caption("Anomaly Overlay (visual scenario)")
                        st.components.v1.html(
                            render_plddt_py3dmol(
                                local_pdb,
                                plddt_df,
                                hotspots,
                                anomaly_emphasis=True,
                                pdb_text_override=deformed_pdb_text or None,
                            ),
                            height=650,
                        )
                    if hotspots.empty:
                        st.info("No hotspots detected at the current threshold.")
                    else:
                        cluster_count = hotspots[hotspots["cluster_id"] >= 0][
                            "residue_num"
                        ].nunique()
                        st.write(
                            f"AI Insight: {cluster_count} residues in structural hotspots."
                        )
                        legend_df = build_cluster_legend(hotspots)
                        if not legend_df.empty:
                            st.subheader("Cluster Legend")
                            st.dataframe(
                                legend_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                st.markdown("---")
                st.subheader("Instability Plot (2D)")
                chart_data = (
                    protein_df.groupby("residue_num")["am_pathogenicity"]
                    .mean()
                    .reset_index()
                )

                line_chart = (
                    alt.Chart(chart_data)
                    .mark_area(
                        line={"color": "darkred"},
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color="white", offset=0),
                                alt.GradientStop(color="red", offset=1),
                            ],
                            x1=1,
                            x2=1,
                            y1=1,
                            y2=0,
                        ),
                    )
                    .encode(
                        x=alt.X("residue_num:Q", title="Residue Position"),
                        y=alt.Y("am_pathogenicity:Q", title="Anomaly Score"),
                        tooltip=["residue_num", "am_pathogenicity"],
                    )
                    .interactive()
                )

                st.altair_chart(line_chart, use_container_width=True)

                st.subheader("Model Confidence (pLDDT)")
                if local_pdb and os.path.exists(local_pdb):
                    if plddt_df.empty:
                        st.warning("No pLDDT values found in the PDB.")
                    else:
                        plddt_plot_df = plddt_df.copy()
                        plddt_plot_df["band"] = plddt_plot_df["plddt"].apply(assign_plddt_band)

                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Mean pLDDT", f"{plddt_plot_df['plddt'].mean():.1f}")
                        m_col2.metric("Min pLDDT", f"{plddt_plot_df['plddt'].min():.1f}")
                        m_col3.metric("Max pLDDT", f"{plddt_plot_df['plddt'].max():.1f}")

                        conf_chart = (
                            alt.Chart(plddt_plot_df)
                            .mark_line()
                            .encode(
                                x=alt.X("residue_num:Q", title="Residue Position"),
                                y=alt.Y(
                                    "plddt:Q",
                                    title="pLDDT (0-100)",
                                    scale=alt.Scale(domain=[0, 100]),
                                ),
                                color=alt.Color(
                                    "band:N",
                                    scale=alt.Scale(
                                        domain=[
                                            "Very high (pLDDT >= 90)",
                                            "High (70 <= pLDDT < 90)",
                                            "Low (50 <= pLDDT < 70)",
                                            "Very low (pLDDT < 50)",
                                        ],
                                        range=["#1f77b4", "#2ca02c", "#ffbf00", "#d62728"],
                                    ),
                                    legend=alt.Legend(title="Confidence Band"),
                                ),
                                tooltip=["residue_num", "plddt", "band"],
                            )
                            .interactive()
                        )
                        st.altair_chart(conf_chart, use_container_width=True)
                else:
                    st.warning("Model confidence requires a local PDB file.")

                st.markdown("---")
                st.subheader("The 'Safety Check' Report")

                report = build_report(uid, protein_df, hotspots)
                status = report["status"]

                r_col1, r_col2, r_col3 = st.columns(3)
                r_col1.metric("Status", status)
                r_col2.metric("Instability Index", f"{report['instability_index']:.3f}")
                r_col3.metric("Anomaly Clusters", report["anomaly_clusters"])

                st.subheader("Export Report")
                st.download_button(
                    "Download JSON",
                    data=json.dumps(report, indent=2),
                    file_name=f"{uid}_anomaly_report.json",
                    mime="application/json",
                )
                st.download_button(
                    "Download PDF",
                    data=report_to_pdf(report),
                    file_name=f"{uid}_anomaly_report.pdf",
                    mime="application/pdf",
                )
else:
    st.info("Select a disease or enter a gene symbol and click 'Run Anomaly Detection'.")
