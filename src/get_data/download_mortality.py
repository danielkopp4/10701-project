"""
Merge NHANES data with mortality follow-up data

This script links NHANES health measurements with mortality outcomes
"""

import pandas as pd
import requests
from pathlib import Path
import os

def download_mortality():
    def download_mortality_file(cycle, data_dir):
        """Download NHANES mortality linkage file for a specific cycle"""
        
        # Mortality file naming pattern
        # Example: NHANES_2017_2018_MORT_2019_PUBLIC.dat
        cycle_underscore = cycle.replace("-", "_")
        filename = f"NHANES_{cycle_underscore}_MORT_2019_PUBLIC.dat"
        
        # CDC FTP URL for mortality linkage
        base_url = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"
        url = f"{base_url}/{filename}"
        
        output_path = data_dir / filename
        
        # Skip if already downloaded
        if output_path.exists():
            print(f"✓ Already downloaded: {filename}")
            return output_path
        
        try:
            print(f"Downloading mortality linkage file: {filename}...", end=" ")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print("✓ Done")
            return output_path
        
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed: {e}")
            print(f"\nPlease manually download from:")
            print(f"  {url}")
            print(f"And place in: {data_dir}")
            return None

    print("="*80)
    print("MERGING NHANES DATA WITH MORTALITY OUTCOMES")
    print("="*80)

    # Load NHANES data
    nhanes_file = Path("../nhanes_2017_2018_merged.csv")

    if not nhanes_file.exists():
        print("\n✗ NHANES data not found. Please run download_nhanes.py first.")
        exit(1)

    print("\nLoading NHANES data...")
    nhanes = pd.read_csv(nhanes_file)
    print(f"✓ Loaded {len(nhanes):,} NHANES participants")

    # Download mortality data for NHANES
    print("\n" + "="*80)
    print("DOWNLOADING MORTALITY LINKAGE FILES")
    print("="*80 + "\n")

    # NHANES cycles to download mortality data for
    cycles_to_download = ["2017-2018"]

    data_dir = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
    for cycle in cycles_to_download:
        download_mortality_file(cycle, data_dir)

    # Load mortality data for NHANES
    mort_files = list(data_dir.glob("NHANES_*_MORT_*.dat"))

    if not mort_files:
        print("\n⚠ No NHANES mortality files found")
        print("\nPlease ensure mortality files are downloaded.")
        exit(1)
        
    else:
        print(f"\nFound {len(mort_files)} mortality file(s)")
        
        # Read mortality data (using same format as NHIS)
        for mort_file in mort_files:
            print(f"\nLoading {mort_file.name}...")
            
            colspecs = [
                (0, 6),    # seqn
                (14, 15),  # eligstat
                (15, 16),  # mortstat
                (16, 19),  # ucod_leading
                (19, 20),  # diabetes
                (20, 21),  # hyperten
                (21, 22),  # dodqtr
                (22, 26),  # dodyear
                (42, 45),  # permth_int
                (45, 48),  # permth_exm
            ]
            
            column_names = [
                'SEQN', 'eligstat', 'mortstat', 'ucod_leading',
                'diabetes', 'hyperten', 'dodqtr', 'dodyear',
                'permth_int', 'permth_exm'
            ]
            
            dtypes = {
                'SEQN': int,
                'eligstat': str,
                'mortstat': 'Int64',
                'ucod_leading': 'Int64',
                'diabetes': 'Int64',
                'hyperten': 'Int64',
                'dodqtr': 'Int64',
                'dodyear': 'Int64',
                'permth_int': 'Int64',
                'permth_exm': 'Int64',
            }
            
            mortality = pd.read_fwf(
                mort_file,
                colspecs=colspecs,
                names=column_names,
                dtype=dtypes,
                na_values=['', '.']
            )
            
            print(f"✓ Loaded {len(mortality):,} mortality records")
            
            # Merge with NHANES data
            print("\nMerging datasets on SEQN...")
            merged = nhanes.merge(mortality, on='SEQN', how='left')
            
            print(f"✓ Merged dataset: {len(merged):,} rows, {len(merged.columns)} columns")
            
            # Calculate follow-up time and create outcome variables
            print("\nCreating outcome variables...")
            
            # Binary mortality outcome (fillna for cases where mortstat is NA)
            merged['died'] = (merged['mortstat'] == 1).fillna(False).astype(int)
            
            # Cause-specific mortality
            merged['died_cvd'] = ((merged['mortstat'] == 1) & 
                                (merged['ucod_leading'].isin([1, 5]))).fillna(False).astype(int)
            merged['died_cancer'] = ((merged['mortstat'] == 1) & 
                                    (merged['ucod_leading'] == 2)).fillna(False).astype(int)
            
            # Summary statistics
            print("\n" + "="*80)
            print("DATASET SUMMARY")
            print("="*80)
            print(f"Total participants: {len(merged):,}")
            print(f"Deaths: {merged['died'].sum():,} ({100*merged['died'].mean():.2f}%)")
            print(f"CVD deaths: {merged['died_cvd'].sum():,}")
            print(f"Cancer deaths: {merged['died_cancer'].sum():,}")
            
            # Check key variables availability
            print("\n" + "-"*80)
            print("KEY VARIABLES COMPLETENESS:")
            print("-"*80)
            
            key_vars = ['RIDAGEYR', 'RIAGENDR', 'BMXBMI', 'BMXWT', 'BMXHT', 
                    'BMXWAIST', 'BPXSY1', 'BPXDI1', 'LBXTC', 'LBXGLU']
            
            for var in key_vars:
                if var in merged.columns:
                    pct = 100 * merged[var].notna().mean()
                    print(f"  {var:15s}: {pct:5.1f}% complete")
            
            # Save merged dataset
            # output_file = Path("../nhanes_with_mortality.csv")
            output_file = data_dir / "nhanes_with_mortality.csv"
            merged.to_csv(output_file, index=False)
            print(f"\n✓ Saved merged dataset to: {output_file}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("config.env")
    
    download_mortality()