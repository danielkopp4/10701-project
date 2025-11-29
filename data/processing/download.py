"""
NHANES Data Download and Processing Script

Downloads and merges NHANES data files for developing improved BMI metrics.
Includes demographics, body measurements, lab results, and mortality linkage.
"""

import pandas as pd
import requests
import os
from pathlib import Path

# Create data directory
data_dir = Path("../nhanes")
data_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("NHANES Data Download Script")
print("="*80)

# NHANES cycles to download (2017-2018 to match your mortality data)
cycles = [
    "2017",
]

# Define file codes for each data type
# Format: (filename_prefix, description)
file_types = {
    'demographics': ('DEMO', 'Demographics'),
    'body_measures': ('BMX', 'Body Measures'),
    'blood_pressure': ('BPX', 'Blood Pressure'),
    'cholesterol': ('TCHOL', 'Total Cholesterol'),
    'glucose': ('GLU', 'Plasma Glucose'),
    'glycohemoglobin': ('GHB', 'Glycohemoglobin (HbA1c)'),
    'insulin': ('INS', 'Insulin'),
    'triglycerides': ('TRIGLY', 'Triglycerides'),
    'hdl': ('HDL', 'HDL Cholesterol'),
    'biochem': ('BIOPRO', 'Biochemistry Profile'),
    'complete_blood': ('CBC', 'Complete Blood Count'),
    'creatinine': ('ALB_CR', 'Albumin & Creatinine'),
    'dxa': ('DXX', 'Dual Energy X-ray Absorptiometry'),
    'physical_activity': ('PAQ', 'Physical Activity'),
    'smoking': ('SMQ', 'Smoking'),
    'alcohol': ('ALQ', 'Alcohol Use'),
    'diet': ('DBQ', 'Diet Behavior'),
    'medical_conditions': ('MCQ', 'Medical Conditions'),
    'diabetes': ('DIQ', 'Diabetes'),
    'cardiovascular': ('CDQ', 'Cardiovascular Health'),
    'weight_history': ('WHQ', 'Weight History'),
}

def download_xpt_file(cycle, file_code, description):
    """Download a single XPT file from NHANES"""
    
    # Convert cycle format (e.g., "2017-2018" -> "I")
    cycle_map = {
        "1999-2000": "",
        "2001-2002": "B",
        "2003-2004": "C",
        "2005-2006": "D",
        "2007-2008": "E",
        "2009-2010": "F",
        "2011-2012": "G",
        "2013-2014": "H",
        "2015-2016": "I",
        "2017-2018": "J",
    }
    
    cycle_letter = cycle_map.get(cycle, "J")
    filename = f"{file_code}_{cycle_letter}.XPT"
    
    # Construct URL - use CDC FTP server
    base_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
    cycle_path = cycle.replace("-", "-")  # Keep format as is
    url = f"{base_url}/{cycle_path}/DataFiles/{filename}"
    
    # Download path
    output_path = data_dir / filename
    
    # Skip if already downloaded
    if output_path.exists():
        print(f"✓ Already downloaded: {filename}")
        return output_path
    
    try:
        print(f"Downloading {description} ({filename})...", end=" ")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print("\n✓ Done")
        return output_path
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed: {e}")
        return None

def load_xpt_file(filepath):
    """Load an XPT file into a pandas DataFrame"""
    if filepath and filepath.exists():
        try:
            df = pd.read_sas(filepath, format='xport', encoding='utf-8')
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    return None

# Download all files
print("\n" + "="*80)
print("DOWNLOADING NHANES FILES")
print("="*80 + "\n")

downloaded_files = {}

for cycle in cycles:
    print(f"\nCycle: {cycle}")
    print("-" * 40)
    
    for key, (code, desc) in file_types.items():
        filepath = download_xpt_file(cycle, code, desc)
        if filepath:
            downloaded_files[key] = filepath

print("\n" + "="*80)
print("LOADING AND MERGING DATA")
print("="*80 + "\n")

# Load all downloaded files
dataframes = {}
for key, filepath in downloaded_files.items():
    print(f"Loading {key}...", end=" ")
    df = load_xpt_file(filepath)
    if df is not None:
        dataframes[key] = df
        print(f"✓ ({df.shape[0]} rows, {df.shape[1]} columns)")
    else:
        print("✗ Failed")

# Merge all dataframes on SEQN (unique ID in NHANES)
if dataframes:
    print("\n" + "-"*80)
    print("Merging datasets on SEQN (Subject ID)...")
    
    # Start with demographics as base
    merged_df = dataframes.get('demographics')
    
    if merged_df is not None:
        # Verify demographics has unique SEQNs
        if merged_df['SEQN'].duplicated().any():
            print("WARNING: Demographics file has duplicate SEQNs!")
        
        base_rows = len(merged_df)
        
        for key, df in dataframes.items():
            if key != 'demographics' and df is not None:
                # Check if SEQN exists in dataframe
                if 'SEQN' not in df.columns:
                    print(f"  ✗ Skipping {key} - no SEQN column")
                    continue
                
                # Check for duplicates
                if df['SEQN'].duplicated().any():
                    print(f"WARNING: {key} has duplicate SEQNs, keeping first occurrence")
                    df = df.drop_duplicates(subset=['SEQN'], keep='first')
                
                print(f"  Merging {key}...", end=" ")
                before_cols = len(merged_df.columns)
                before_rows = len(merged_df)
                
                # Use left join to preserve all demographic records
                merged_df = merged_df.merge(df, on='SEQN', how='left', suffixes=('', f'_{key}'))
                
                after_cols = len(merged_df.columns)
                after_rows = len(merged_df)
                
                # Verify row count didn't change (should stay same with left join)
                if after_rows != before_rows:
                    print(f"Row count changed: {before_rows} → {after_rows}!")
                else:
                    print(f"✓ (+{after_cols - before_cols} columns)")
        
        # Verify final row count matches demographics
        if len(merged_df) != base_rows:
            print(f"\nWARNING: Final row count ({len(merged_df)}) doesn't match demographics ({base_rows})!")
        
        # Check for columns with suffixes (indicates name collisions)
        suffix_cols = [col for col in merged_df.columns if any(f'_{key}' in col for key in dataframes.keys())]
        if suffix_cols:
            print(f"\nFound {len(suffix_cols)} columns with suffixes (name collisions):")
            for col in suffix_cols[:5]:  # Show first 5
                print(f"    {col}")
        
        # Save merged dataset
        output_csv = "../nhanes_2017_2018_merged.csv"
        merged_df.to_csv(output_csv, index=False)
        
        print("\n" + "="*80)
        print("MERGED DATASET SUMMARY")
        print("="*80)
        print(f"Total participants: {len(merged_df)}")
        print(f"Total variables: {len(merged_df.columns)}")
        print(f"Duplicate SEQNs in final: {merged_df['SEQN'].duplicated().sum()}")
        print(f"\nSaved to: {output_csv}")
        
        # Display key variables
        print("\n" + "="*80)
        print("KEY VARIABLES AVAILABLE:")
        print("="*80)
        
        key_vars = {
            'SEQN': 'Respondent ID',
            'RIAGENDR': 'Gender',
            'RIDAGEYR': 'Age in years',
            'RIDRETH3': 'Race/Ethnicity',
            'BMXBMI': 'Body Mass Index (kg/m²)',
            'BMXWT': 'Weight (kg)',
            'BMXHT': 'Standing Height (cm)',
            'BMXWAIST': 'Waist Circumference (cm)',
            'BPXSY1': 'Systolic Blood Pressure (mmHg)',
            'BPXDI1': 'Diastolic Blood Pressure (mmHg)',
            'LBXTC': 'Total Cholesterol (mg/dL)',
            'LBXGLU': 'Fasting Glucose (mg/dL)',
            'LBXGH': 'Glycohemoglobin (%)',
            'LBXIN': 'Insulin (µU/mL)',
            'LBXTR': 'Triglycerides (mg/dL)',
        }
        
        available = []
        for var, desc in key_vars.items():
            if var in merged_df.columns:
                non_null = merged_df[var].notna().sum()
                available.append((var, desc, non_null))
        
        for var, desc, count in available:
            print(f"  {var:15s} - {desc:40s} ({count:,} non-null)")
        
        # Basic statistics
        print("\n" + "="*80)
        print("BASIC STATISTICS")
        print("="*80)
        
        stats_vars = ['RIDAGEYR', 'BMXBMI', 'BMXWT', 'BMXHT', 'BMXWAIST']
        available_stats = [v for v in stats_vars if v in merged_df.columns]
        
        if available_stats:
            print(merged_df[available_stats].describe())
        
    else:
        print("\n✗ Failed to load demographics file (required base)")
else:
    print("\n✗ No data files were successfully loaded")
