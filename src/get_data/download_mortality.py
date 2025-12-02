import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

MORTALITY_BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"
NHANES_CYCLES = [
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
NHIS_YEARS = list(range(1986, 2019))  # public-use mortality linkage years

# Shared fixed-width layout for mortality PUFs
MORTALITY_COLSPECS = [
    (0, 6),  # identifier
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


def download_file(filename: str, data_dir: Path) -> Optional[Path]:
    """Download a mortality file if not present locally."""
    output_path = data_dir / filename
    if output_path.exists():
        print(f"✅ Already downloaded: {filename}")
        return output_path

    url = f"{MORTALITY_BASE_URL}/{filename}"

    try:
        print(f"Downloading {filename}...", end=" ")
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        print("✅ Done")
        return output_path
    except requests.RequestException as e:
        print(f"❌ Failed: {e}")
        print(f"   Manual download URL: {url}")
        return None


def load_mortality_file(
    filepath: Path,
    column_names: List[str],
    dtypes: Dict[str, object],
) -> pd.DataFrame:
    return pd.read_fwf(
        filepath,
        colspecs=MORTALITY_COLSPECS,
        names=column_names,
        dtype=dtypes,
        na_values=["", ".", " "],
    )


def locate_nhanes_base(data_dir: Path) -> Optional[Path]:
    candidates = [
        data_dir / "nhanes_all_cycles_merged.csv",
        data_dir / "nhanes_2017_2018_merged.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def download_nhanes_mortality(data_dir: Path) -> None:
    nhanes_file = locate_nhanes_base(data_dir)
    if nhanes_file is None:
        print("\n❌ NHANES merged data not found. Please run download_nhanes.py first.")
        return

    print("\nLoading NHANES merged data...")
    nhanes = pd.read_csv(nhanes_file, low_memory=False)
    print(f"✅ Loaded {len(nhanes):,} NHANES participants")

    print("\n" + "=" * 80)
    print("DOWNLOADING NHANES MORTALITY LINKAGE FILES")
    print("=" * 80 + "\n")

    for cycle in NHANES_CYCLES:
        cycle_underscore = cycle.replace("-", "_")
        filename = f"NHANES_{cycle_underscore}_MORT_2019_PUBLIC.dat"
        download_file(filename, data_dir)

    nhanes_mort_files = sorted(data_dir.glob("NHANES_*_MORT_*PUBLIC.dat"))
    if not nhanes_mort_files:
        print("\n⚠️️️️ No NHANES mortality files found")
        return

    column_names = [
        "SEQN",
        "eligstat",
        "mortstat",
        "ucod_leading",
        "diabetes",
        "hyperten",
        "dodqtr",
        "dodyear",
        "permth_int",
        "permth_exm",
    ]
    dtypes = {
        "SEQN": int,
        "eligstat": str,
        "mortstat": "Int64",
        "ucod_leading": "Int64",
        "diabetes": "Int64",
        "hyperten": "Int64",
        "dodqtr": "Int64",
        "dodyear": "Int64",
        "permth_int": "Int64",
        "permth_exm": "Int64",
    }

    mortality_dfs: List[pd.DataFrame] = []

    for mort_file in nhanes_mort_files:
        cycle_label = mort_file.name.split("_MORT")[0].replace("NHANES_", "").replace("_", "-")
        print(f"Loading {mort_file.name} (cycle {cycle_label})...")
        mortality = load_mortality_file(mort_file, column_names, dtypes)
        mortality["mortality_cycle"] = cycle_label
        mortality_dfs.append(mortality)
        print(f"✅ Loaded {len(mortality):,} mortality records")

    mortality_all = pd.concat(mortality_dfs, ignore_index=True)

    print("\nMerging NHANES data with mortality records on SEQN...")
    merged = nhanes.merge(mortality_all, on="SEQN", how="left")

    # Binary mortality outcome (eligibility-aware)
    is_eligible = merged["eligstat"].astype(str) == "1"
    merged["died"] = pd.NA
    merged.loc[is_eligible, "died"] = (merged.loc[is_eligible, "mortstat"] == 1).astype(int)
    merged["died"] = merged["died"].astype("Int64")

    merged["died_cvd"] = (
        (merged["mortstat"] == 1) & (merged["ucod_leading"].isin([1, 5]))
    ).fillna(False).astype(int)
    merged["died_cancer"] = (
        (merged["mortstat"] == 1) & (merged["ucod_leading"] == 2)
    ).fillna(False).astype(int)

    print("\n" + "=" * 80)
    print("NHANES DATASET SUMMARY")
    print("=" * 80)
    print(f"Total participants: {len(merged):,}")
    print(f"Deaths (eligible only): {merged['died'].sum(skipna=True):,}")
    print(f"CVD deaths: {merged['died_cvd'].sum():,}")
    print(f"Cancer deaths: {merged['died_cancer'].sum():,}")

    key_vars = [
        "RIDAGEYR",
        "RIAGENDR",
        "BMXBMI",
        "BMXWT",
        "BMXHT",
        "BMXWAIST",
        "BPXSY1",
        "BPXDI1",
        "LBXTC",
        "LBXGLU",
    ]
    print("\nKey variables completeness (non-null %):")
    for var in key_vars:
        if var in merged.columns:
            pct = 100 * merged[var].notna().mean()
            print(f"  {var:12s}: {pct:5.1f}%")

    output_file = data_dir / "nhanes_with_mortality.csv"
    merged.to_csv(output_file, index=False)
    print(f"\n✅ Saved NHANES + mortality dataset to: {output_file}")


def download_nhis_mortality(data_dir: Path) -> None:
    print("\n" + "=" * 80)
    print("DOWNLOADING NHIS MORTALITY LINKAGE FILES")
    print("=" * 80 + "\n")

    for year in NHIS_YEARS:
        filename = f"NHIS_{year}_MORT_2019_PUBLIC.dat"
        download_file(filename, data_dir)

    nhis_mort_files = sorted(data_dir.glob("NHIS_*_MORT_*PUBLIC.dat"))
    if not nhis_mort_files:
        print("\n⚠️️️️ No NHIS mortality files found")
        return

    column_names = [
        "publicid",
        "eligstat",
        "mortstat",
        "ucod_leading",
        "diabetes",
        "hyperten",
        "dodqtr",
        "dodyear",
        "permth_int",
        "permth_exm",
    ]
    dtypes = {
        "publicid": int,
        "eligstat": str,
        "mortstat": "Int64",
        "ucod_leading": "Int64",
        "diabetes": "Int64",
        "hyperten": "Int64",
        "dodqtr": "Int64",
        "dodyear": "Int64",
        "permth_int": "Int64",
        "permth_exm": "Int64",
    }

    mortality_dfs: List[pd.DataFrame] = []

    for mort_file in nhis_mort_files:
        year_label = mort_file.name.split("_MORT")[0].replace("NHIS_", "")
        print(f"Loading {mort_file.name} (year {year_label})...")
        mortality = load_mortality_file(mort_file, column_names, dtypes)
        mortality["survey_year"] = year_label
        mortality["source"] = "NHIS"

        mortality["died"] = (mortality["mortstat"] == 1).astype("Int64")
        mortality["event"] = mortality["died"]

        # NHIS has interview follow-up only; use permth_int for survival time
        mortality["time"] = pd.to_numeric(mortality["permth_int"], errors="coerce")

        mortality_dfs.append(mortality)
        print(f"✅ Loaded {len(mortality):,} mortality records")

    mortality_all = pd.concat(mortality_dfs, ignore_index=True)

    output_file = data_dir / "nhis_mortality.csv"
    mortality_all.to_csv(output_file, index=False)

    print("\n" + "=" * 80)
    print("NHIS MORTALITY SUMMARY")
    print("=" * 80)
    print(f"Total NHIS mortality records: {len(mortality_all):,}")
    print(f"Deaths: {mortality_all['died'].sum(skipna=True):,}")
    print(f"Saved mortality-only dataset to: {output_file}")


def download_mortality():
    data_dir = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MERGING NHANES DATA WITH MORTALITY OUTCOMES")
    print("=" * 80)

    download_nhanes_mortality(data_dir)
    download_nhis_mortality(data_dir)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        def load_dotenv(*args, **kwargs):
            print("⚠️️️️ python-dotenv not installed; proceeding without loading config.env")

    load_dotenv("config.env")
    download_mortality()
