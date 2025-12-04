import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


def download_nhanes():
    data_dir = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NHANES Data Download Script")
    print("=" * 80)

    # All public-use NHANES cycles covered by the 2019 mortality release
    cycles: List[str] = [
        "1999-2000",
        "2001-2002",
        "2003-2004",
        "2005-2006",
        "2007-2008",
        "2009-2010",
        "2011-2012",
        "2013-2014",
        "2015-2016",
        "2017-2018",
    ]

    # Format: (filename_prefixes, description)
    # Each entry can have multiple filename options to try (primary first, then fallbacks)
    # Naming patterns: 2003+ uses standard codes, 2001-2002 uses L##_B format, 1999-2000 uses LAB## format
    file_types: Dict[str, tuple[List[str], str]] = {
        "demographics": (["DEMO"], "Demographics"),
        "body_measures": (["BMX"], "Body Measures"),
        "blood_pressure": (["BPX"], "Blood Pressure"),
        "cholesterol_total_hdl": (["TCHOL", "L13", "LAB13"], "Total Cholesterol & HDL"),
        "cholesterol_ldl_trig": (["L13AM", "LAB13AM", "TRIGLY"], "LDL & Triglycerides"),
        "glucose": (["GLU", "L10AM", "LAB10AM"], "Plasma Glucose & Insulin"),
        "glycohemoglobin": (["GHB", "L10", "LAB10"], "Glycohemoglobin (HbA1c)"),
        "insulin": (["INS", "LAB10AM", "L10_2", "L10AM", "GLU"], "Insulin"),
        "triglycerides": (["TRIGLY", "LAB13AM", "L13AM"], "Triglycerides"),
        "hdl": (["HDL", "LAB13", "L13"], "HDL Cholesterol"),
        "biochem": (["BIOPRO", "L40", "LAB18"], "Biochemistry Profile"),
        "complete_blood": (["CBC", "L25", "LAB25"], "Complete Blood Count"),
        "creatinine": (["ALB_CR", "L16", "LAB16"], "Albumin & Creatinine"),
        "hepatitis_a": (["HEPA", "L02HPA", "L02HPA_A"], "Hepatitis A Antibody"),
        "hepatitis_b_surface": (["L02HBS", "LAB02", "HEPB_S"], "Hepatitis B Surface Antibody"),
        "hepatitis_core": (["L02", "LAB02", "HEPBD"], "Hepatitis B & C Core"),
        "c_reactive_protein": (["CRP", "L11", "LAB11", "HSCRP"], "C-Reactive Protein"),
        "herpes": (["L09", "LAB09", "HSV"], "Herpes Simplex Virus Type-1 & Type-2"),
        "hiv": (["L03", "LAB03", "HIV"], "HIV Antibody Test"),
        "measles_rubella_varicella": (["L19", "LAB19"], "Measles, Rubella, & Varicella"),
        "iron": (["L40FE", "FETIB"], "Iron, TIBC, & Transferrin Saturation"),
        "thyroid": (["L40T4", "LAB18T4", "L11", "PTH", "THYROID", "THYROD"], "Thyroid - TSH & T4"),
        "vitamin_d": (["VID"], "Vitamin D"),
        "dxa": (["DXX"], "Dual Energy X-ray Absorptiometry"),
        "physical_activity": (["PAQ"], "Physical Activity"),
        "smoking": (["SMQ"], "Smoking"),
        "alcohol": (["ALQ"], "Alcohol Use"),
        "diet": (["DBQ"], "Diet Behavior"),
        "medical_conditions": (["MCQ"], "Medical Conditions"),
        "diabetes": (["DIQ"], "Diabetes"),
        "cardiovascular": (["CDQ"], "Cardiovascular Health"),
        "weight_history": (["WHQ"], "Weight History"),
        "fasting": (["PH", "FASTQX"], "Fasting Questionnaire"),
        "pregnancy": (["UC", "UCPREG"], "Pregnancy Test - Urine"),
    }

    # Cycle -> suffix letter used in filenames
    cycle_map: Dict[str, str] = {
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

    base_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

    def is_valid_xpt(path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            with open(path, "rb") as f:
                header = f.read(40)
            return header.startswith(b"HEADER RECORD*******LIBRARY HEADER")
        except OSError:
            return False

    def download_xpt_file(cycle: str, file_codes: List[str], description: str) -> Path | None:
        """
        Try to download a file with multiple possible filename patterns.
        Tries each file_code in order until one succeeds.
        Downloads with the actual filename but renames to the primary (first) filename for consistency.
        """
        if cycle not in cycle_map:
            raise ValueError(f"Missing file suffix mapping for cycle: {cycle}")

        cycle_letter = cycle_map[cycle]
        cycle_prefix = cycle[:cycle.find("-")]
        
        # Primary filename (what we want to save as)
        primary_code = file_codes[0]
        primary_filename = f"{primary_code}_{cycle_letter}.XPT" if cycle_letter else f"{primary_code}.XPT"
        primary_path = data_dir / primary_filename

        # Check if primary file already exists
        if primary_path.exists():
            if is_valid_xpt(primary_path):
                print(f"Already downloaded: {primary_filename}")
                return primary_path
            else:
                print(f"Existing file invalid, re-downloading: {primary_filename}")

        # Try each file code variant
        for idx, file_code in enumerate(file_codes):
            filename = f"{file_code}_{cycle_letter}.XPT" if cycle_letter else f"{file_code}.XPT"
            url = f"{base_url}/{cycle_prefix}/DataFiles/{filename}"
            temp_path = data_dir / filename

            try:
                # Only print for the first attempt
                if idx == 0:
                    print(f"Downloading {description} ({primary_filename})...", end=" ")
                    
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                with open(temp_path, "wb") as f:
                    f.write(response.content)

                if not is_valid_xpt(temp_path):
                    if temp_path.exists():
                        temp_path.unlink()
                    continue

                # Rename to primary filename if downloaded with a fallback name
                if temp_path != primary_path:
                    temp_path.rename(primary_path)

                print("Done")
                return primary_path
            except requests.exceptions.RequestException:
                # Silently try next fallback
                if temp_path.exists():
                    temp_path.unlink()
                continue

        # All attempts failed
        print(f"\nNo file for {description} in cycle {cycle}. Skipping...")
        return None

    def load_xpt_file(filepath: Path | None) -> pd.DataFrame | None:
        if filepath and filepath.exists():
            try:
                return pd.read_sas(filepath, format="xport", encoding="utf-8")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        return None

    print("\n" + "=" * 80)
    print("DOWNLOADING NHANES FILES")
    print("=" * 80 + "\n")

    downloaded_files: Dict[str, Dict[str, Path]] = {}

    for cycle in cycles:
        print(f"\nCycle: {cycle}")
        print("-" * 40)

        downloaded_files[cycle] = {}

        for key, (codes, desc) in file_types.items():
            filepath = download_xpt_file(cycle, codes, desc)
            if filepath:
                downloaded_files[cycle][key] = filepath

    print("\n" + "=" * 80)
    print("LOADING AND MERGING DATA")
    print("=" * 80 + "\n")

    merged_cycles: List[pd.DataFrame] = []

    for cycle, file_map in downloaded_files.items():
        dataframes: Dict[str, pd.DataFrame] = {}

        for key, filepath in file_map.items():
            print(f"[{cycle}] Loading {key}...", end=" ")
            df = load_xpt_file(filepath)
            if df is not None:
                dataframes[key] = df
                print(f"({df.shape[0]} rows, {df.shape[1]} columns)")
            else:
                print("Failed")

        if not dataframes:
            print(f"[{cycle}] No data files were successfully loaded")
            continue

        merged_df = dataframes.get("demographics")

        if merged_df is None:
            print(f"[{cycle}] Demographics file missing; skipping cycle")
            continue

        if merged_df["SEQN"].duplicated().any():
            print(f"[{cycle}] WARNING: Demographics file has duplicate SEQNs!")

        base_rows = len(merged_df)

        for key, df in dataframes.items():
            if key == "demographics" or df is None:
                continue

            if "SEQN" not in df.columns:
                print(f"[{cycle}] Skipping {key} - no SEQN column")
                continue

            if df["SEQN"].duplicated().any():
                print(f"[{cycle}] WARNING: {key} has duplicate SEQNs, keeping first occurrence")
                df = df.drop_duplicates(subset=["SEQN"], keep="first")

            print(f"[{cycle}] Merging {key}...", end=" ")
            before_cols = len(merged_df.columns)
            before_rows = len(merged_df)

            merged_df = merged_df.merge(
                df,
                on="SEQN",
                how="left",
                suffixes=("", f"_{key}"),
            )

            after_cols = len(merged_df.columns)
            after_rows = len(merged_df)

            if after_rows != before_rows:
                print(f"Row count changed: {before_rows} → {after_rows}!")
            else:
                print(f"(+{after_cols - before_cols} columns)")

        if len(merged_df) != base_rows:
            print(f"[{cycle}] WARNING: Final row count ({len(merged_df)}) doesn't match demographics ({base_rows})!")

        merged_df["survey_cycle"] = cycle
        merged_cycles.append(merged_df)

    if not merged_cycles:
        print("\nNo data files were successfully loaded across cycles")
        return

    combined_df = pd.concat(merged_cycles, ignore_index=True, sort=False)

    suffix_cols = [col for col in combined_df.columns if any(f"_{key}" in col for key in file_types.keys())]
    if suffix_cols:
        print(f"\nFound {len(suffix_cols)} columns with suffixes (name collisions). First few:")
        for col in suffix_cols[:5]:
            print(f"    {col}")

    output_csv = data_dir / "nhanes_all_cycles_merged.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print("MERGED DATASET SUMMARY")
    print("=" * 80)
    print(f"Total participants (all cycles): {len(combined_df)}")
    print(f"Total variables: {len(combined_df.columns)}")
    print(f"Duplicate SEQNs in final: {combined_df['SEQN'].duplicated().sum()}")
    print(f"Saved to: {output_csv}")

    key_vars = {
        "SEQN": "Respondent ID",
        "RIAGENDR": "Gender",
        "RIDAGEYR": "Age in years",
        "RIDRETH3": "Race/Ethnicity",
        "BMXBMI": "Body Mass Index (kg/m²)",
        "BMXWT": "Weight (kg)",
        "BMXHT": "Standing Height (cm)",
        "BMXWAIST": "Waist Circumference (cm)",
        "BPXSY1": "Systolic Blood Pressure (mmHg)",
        "BPXDI1": "Diastolic Blood Pressure (mmHg)",
        "LBXTC": "Total Cholesterol (mg/dL)",
        "LBXGLU": "Fasting Glucose (mg/dL)",
        "LBXGH": "Glycohemoglobin (%)",
        "LBXIN": "Insulin (µU/mL)",
        "LBXTR": "Triglycerides (mg/dL)",
    }

    available = []
    for var, desc in key_vars.items():
        if var in combined_df.columns:
            non_null = combined_df[var].notna().sum()
            available.append((var, desc, non_null))

    if available:
        print("\nKey variables availability (non-null counts across all cycles):")
        for var, desc, count in available:
            print(f"  {var:15s} - {desc:40s} ({count:,} non-null)")

    stats_vars = ["RIDAGEYR", "BMXBMI", "BMXWT", "BMXHT", "BMXWAIST"]
    available_stats = [v for v in stats_vars if v in combined_df.columns]

    if available_stats:
        print("\n" + "=" * 80)
        print("BASIC STATISTICS")
        print("=" * 80)
        print(combined_df[available_stats].describe())


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        def load_dotenv(*args, **kwargs):
            print("python-dotenv not installed; proceeding without loading config.env")

    load_dotenv("config.env")
    download_nhanes()
