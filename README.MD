# Rare-Disease Protein Impact Explorer 🧬

An intelligent AI-driven platform that translates rare-disease genetic variants into 3D structural insights. By integrating AlphaFold's protein modeling with AlphaMissense pathogenicity predictions, this tool helps researchers visualize how specific mutations physically disrupt protein function. [web:63][web:136]

## The Innovation
While standard tools provide 1D pathogenicity scores, this platform introduces **Spatial Proximity Analysis**. It calculates 3D distances between multiple mutations to identify "Synergistic Hotspots"—clusters of variants that sit near each other in 3D space even if they are far apart in the linear protein sequence. [web:109][web:151]

## Team Members
- **[Moinuddin Hasan Raichur Mohammed]** - [Role/Contribution]
- **[Aditya The Sridhar]** - [Role/Contribution]
- **[Aaditya Tripathi]** - [Role/Contribution]
- **[Goureesankar s nair]** - [Role/Contribution]

## Core Workflow
1. **Input**: User provides a Gene Symbol (e.g., *ALS2*) or a list of missense variants.
2. **Translation**: The app queries the **UniProt REST API** to map gene names to Protein Accession IDs. [web:128]
3. **Modeling**: The system fetches 3D structures from the **AlphaFold DB** (CC-BY-4.0). [web:64]
4. **Analysis**: **AlphaMissense** scores are mapped onto the 3D structure, highlighting "Likely Pathogenic" regions in red. [web:90][web:136]
5. **Visualization**: Interactive 3D rendering is provided via the **Mol* Viewer**. [web:109]

## Installation & Setup
1. Clone the repository: `git clone https://github.com/your-repo/rare-disease-explorer.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Technologies Used
- **Frontend**: Streamlit (Apache-2.0) [web:103]
- **Backend**: Python, FastAPI (MIT) [web:108]
- **Bioinformatics**: Biopython, UniProt API [web:118]
- **3D Rendering**: Mol* (MIT) [web:109]
- **AI Models**: AlphaFold2, AlphaMissense [web:76][web:84]

## Disclaimer
This tool is for research and educational purposes only. It uses AI predictions from the AlphaFold Database and AlphaMissense, which have varying levels of confidence and are not approved for clinical diagnostic use. [web:63][page:5]
