import pandas as pd
import numpy as np
import requests
import io
import streamlit as st
import xml.etree.ElementTree as ET
from Bio.PDB import PDBParser
from sklearn.cluster import DBSCAN
import os

# FUNCTION: load_rare_disease_catalog
# Purpose: Parses the Orphanet XML file to create a searchable map of rare diseases and their primary associated genes.
# Input: xml_path (String) - Path to the Orphanet 'en_product6.xml' file.
# Output: disease_map (Dictionary) - A dictionary where keys are disease names and values are gene symbols.
@st.cache_data
def load_rare_disease_catalog(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    disease_map = {}

    for disorder in root.findall(".//Disorder"):
        name_element = disorder.find("Name")
        gene_elements = disorder.findall(".//Gene")
        
        if name_element is not None and gene_elements:
            disease_name = name_element.text
            primary_gene = gene_elements[0].find("Symbol").text
            disease_map[disease_name] = primary_gene
            
    return disease_map

# FUNCTION: get_uniprot_id
# Purpose: Connects to the UniProt REST API to translate a human gene symbol into a unique Protein Accession ID.
# Input: gene_symbol (String) - The symbol of the gene (e.g., 'ALS2').
# Output: uniprot_id (String/None) - The primary UniProt Accession ID or None if not found.
@st.cache_data
def get_uniprot_id(gene_symbol):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_symbol} AND organism_id:9606&format=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data.get('results'):
            return data['results'][0]['primaryAccession']
        return None
    except Exception:
        return None

# FUNCTION: get_af_structure_url
# Purpose: Uses the AlphaFold API to find the REAL current download link for a protein.
# Input: uniprot_id (String)
# Output: pdb_url (String/None)
def get_af_structure_url(uniprot_id):
    if not uniprot_id:
        return None
    
    # We use the official API to find the correct file version
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id.strip()}"
    
    try:
        response = requests.get(api_url, verify=False, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                # The API returns a list of files; we take the pdbUrl from the first one
                return data[0].get('pdbUrl')
    except Exception as e:
        print(f"AlphaFold API Lookup Error: {e}")
    
    # Fallback to the most likely URL if the API is down
    return f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id.strip()}-F1-model_v4.pdb"

# FUNCTION: get_protein_anomalies_smart
# Purpose: Scans the AlphaMissense dataset using chunked reading, ensuring column names are strictly mapped to the first 4 columns.
# Input: uniprot_id (String), file_path (String) - Path to the AlphaMissense TSV file.
# Output: protein_df (DataFrame) - A filtered table containing variants and scores.
@st.cache_data
def get_protein_anomalies_smart(uniprot_id, file_path):
    # These are the specific columns for the hg38 AlphaMissense file
    # 5: uniprot_id, 7: protein_variant, 8: pathogenicity, 9: class
    col_map = {5: 'uniprot_id', 7: 'variant', 8: 'am_pathogenicity', 9: 'am_class'}
    
    try:
        chunks = pd.read_csv(
            file_path, 
            sep='\t', 
            compression='gzip', # Explicitly handle the .gz
            comment='#', 
            usecols=list(col_map.keys()),
            names=[col_map[i] for i in sorted(col_map.keys())], # Placeholder names
            header=None,
            chunksize=200000, # Larger chunks for faster GZ processing
            engine='c'
        )
        
        relevant_chunks = []
        for chunk in chunks:
            # Re-assign correct names to ensure no mapping errors
            chunk.columns = ['uniprot_id', 'variant', 'am_pathogenicity', 'am_class']
            
            # Clean the ID column
            chunk['uniprot_id'] = chunk['uniprot_id'].astype(str).str.strip()
            
            filtered = chunk[chunk['uniprot_id'] == uniprot_id].copy()
            
            if not filtered.empty:
                # Extract residue number (e.g., 'P123A' -> 123)
                filtered['residue_num'] = filtered['variant'].str.extract('(\d+)').fillna(0).astype(int)
                relevant_chunks.append(filtered)
                
        if not relevant_chunks:
            return pd.DataFrame(columns=['uniprot_id', 'variant', 'am_pathogenicity', 'am_class', 'residue_num'])
            
        return pd.concat(relevant_chunks)
        
    except Exception as e:
        st.error(f"Error reading compressed file: {e}")
        return pd.DataFrame()

# UPDATED FUNCTION: detect_structural_hotspots
# Change the input from 'pdb_url' to 'local_pdb_path' for better performance
# Purpose: Extracts 3D coordinates from a PDB file and applies DBSCAN clustering to identify physical "hotspots".
# Input: pdb_url (String), high_risk_df (DataFrame), eps (Float), min_samples (Int).
# Output: cluster_results (DataFrame) - Table mapping residue numbers to Cluster IDs.
def detect_structural_hotspots(local_pdb_path, high_risk_df, eps=10.0, min_samples=3):
    if high_risk_df.empty or not local_pdb_path or not os.path.exists(local_pdb_path):
        return pd.DataFrame()

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", local_pdb_path)
        model = structure[0]
        
        coords = []
        residue_ids = []
        target_residues = high_risk_df['residue_num'].unique()
        
        for res_num in target_residues:
            try:
                residue = model['A'][res_num]
                atom = residue['CA']
                coords.append(atom.get_coord())
                residue_ids.append(res_num)
            except KeyError:
                continue
                
        if len(coords) < min_samples:
            return pd.DataFrame()

        x_data = np.array(coords)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(x_data)
        
        return pd.DataFrame({
            'residue_num': residue_ids,
            'cluster_id': dbscan.labels_
        })
    except Exception:
        return pd.DataFrame()
    
# FUNCTION: download_pdb_locally
# Purpose: Downloads the PDB file. Added SSL bypass and error logging.
def download_pdb_locally(url, uniprot_id):
    if not url:
        return None
        
    try:
        os.makedirs("temp_pdb", exist_ok=True)
        local_path = os.path.join("temp_pdb", f"{uniprot_id.strip()}.pdb")
        
        # verify=False bypasses the SSL issues on your machine
        response = requests.get(url, verify=False, timeout=15) 
        
        if response.status_code == 200:
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Successfully downloaded structure to {local_path}")
            return local_path
        else:
            print(f"Download failed for {uniprot_id}. URL: {url} | Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"CRITICAL ERROR during download: {e}")
        return None
    