import streamlit as st
import altair as alt
import pandas as pd
import os
from streamlit_molstar import st_molstar
from src.logic_engine import (
    load_rare_disease_catalog, 
    get_uniprot_id, 
    get_af_structure_url, 
    get_protein_anomalies_smart, 
    detect_structural_hotspots,
    download_pdb_locally
)

st.set_page_config(page_title="Structural Anomaly Detector", layout="wide")

st.title("🧬 Rare-Disease Structural Anomaly Detector")
st.markdown("---")

st.sidebar.header("Search Gateway")
try:
    disease_map = load_rare_disease_catalog("data/references/en_product6.xml")
    selected_disease = st.sidebar.selectbox("Select Rare Disease", options=list(disease_map.keys()))
    default_gene = disease_map[selected_disease]
except Exception:
    st.sidebar.warning("Orphanet Catalog not found.")
    default_gene = "SOD1"

gene_symbol = st.sidebar.text_input("Gene Symbol", value=default_gene)
run_btn = st.sidebar.button("Run Anomaly Detection", type="primary")

if run_btn:
    with st.spinner(f"Analyzing {gene_symbol}..."):
        uid = get_uniprot_id(gene_symbol)
        
        if not uid:
            st.error(f"Could not find UniProt ID for {gene_symbol}")
        else:
            am_file = "data/raw/AlphaMissense_hg38.tsv.gz"
            protein_df = get_protein_anomalies_smart(uid, am_file)
            
            if protein_df.empty:
                st.warning(f"No anomaly data found for {uid}")
            else:
                pdb_url = get_af_structure_url(uid)
                local_pdb = download_pdb_locally(pdb_url, uid)
                
                high_risk_df = protein_df[protein_df['am_pathogenicity'] > 0.9]
                hotspots = detect_structural_hotspots(local_pdb, high_risk_df)
                
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("3D Anomaly Map")
                    if not local_pdb:
                        st.error("3D Structure could not be downloaded. Check internet or SSL settings.")
                    else:
                        st_molstar(local_pdb, key=f"mol_{uid}", height=500)
                        if not hotspots.empty:
                            cluster_count = hotspots[hotspots['cluster_id'] >= 0]['residue_num'].nunique()
                            st.write(f"📍 **AI Insight:** {cluster_count} residues in structural hotspots.")

                with col2:
                    st.subheader("Instability Plot (2D)")
                    chart_data = protein_df.groupby('residue_num')['am_pathogenicity'].mean().reset_index()
                    
                    line_chart = alt.Chart(chart_data).mark_area(
                        line={'color':'darkred'},
                        color=alt.Gradient(
                            gradient='linear',
                            stops=[alt.GradientStop(color='white', offset=0),
                                   alt.GradientStop(color='red', offset=1)],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    ).encode(
                        x=alt.X('residue_num:Q', title='Residue Position'),
                        y=alt.Y('am_pathogenicity:Q', title='Anomaly Score'),
                        tooltip=['residue_num', 'am_pathogenicity']
                    ).interactive()
                    
                    st.altair_chart(line_chart, width="stretch")

                st.markdown("---")
                st.subheader("The 'Safety Check' Report")
                
                gii_score = protein_df['am_pathogenicity'].mean()
                num_clusters = hotspots[hotspots['cluster_id'] >= 0]['cluster_id'].nunique() if not hotspots.empty else 0
                
                status = "🟢 STABLE"
                if gii_score > 0.6 or num_clusters > 0:
                    status = "🔴 CRITICAL"
                elif gii_score > 0.4:
                    status = "🟡 WARNING"

                r_col1, r_col2, r_col3 = st.columns(3)
                r_col1.metric("Status", status)
                r_col2.metric("Instability Index", f"{gii_score:.3f}")
                r_col3.metric("Anomaly Clusters", num_clusters)
else:
    st.info("Select a disease or enter a gene symbol and click 'Run Anomaly Detection'.")