import pandas as pd
from pathlib import Path
import os

# Define the essential variables to keep
ESSENTIAL_VARIABLES = [
    # Identifier
    'SEQN',        # Respondent sequence number (unique ID)

    # Demographics
    'RIAGENDR',    # Gender (1=Male, 2=Female)
    'RIDAGEYR',    # Age in years at screening
    'RIDRETH3',    # Race/Hispanic origin with NH Asian
    'DMDEDUC2',    # Education level - Adults 20+
    'INDFMPIR',    # Ratio of family income to poverty
    
    # Anthropometric Measurements
    'BMXWT',       # Weight (kg)
    'BMXHT',       # Standing height (cm)
    'BMXHEAD',     # Head circumference (cm)
    'BMXBMI',      # Body Mass Index (kg/m²)
    'BMXLEG',      # Upper leg length (cm)
    'BMXARML',     # Upper arm length (cm)
    'BMXARMC',     # Arm circumference (cm)
    'BMXWAIST',    # Waist circumference (cm)
    'BMXHIP',      # Hip circumference (cm)
    'BMXARMC',     # Arm circumference (cm) - duplicate [umm why?]
    
    # Blood Pressure (individual readings - will be averaged)
    'BPXSY1',      # Systolic BP - 1st reading (mm Hg)
    'BPXSY2',      # Systolic BP - 2nd reading (mm Hg)
    'BPXSY3',      # Systolic BP - 3rd reading (mm Hg)
    'BPXSY4',      # Systolic BP - 4th reading (mm Hg)
    'BPXDI1',      # Diastolic BP - 1st reading (mm Hg)
    'BPXDI2',      # Diastolic BP - 2nd reading (mm Hg)
    'BPXDI3',      # Diastolic BP - 3rd reading (mm Hg)
    'BPXDI4',      # Diastolic BP - 4th reading (mm Hg)
    'BPXPULS',     # Pulse - 60 sec (per min)
    
    # Lipid Panel
    'LBXTC',       # Total cholesterol (mg/dL)
    'LBDHDD',      # HDL cholesterol (mg/dL)
    'LBDLDL',      # LDL cholesterol (mg/dL)
    'LBXTR',       # Triglycerides (mg/dL)
    
    # Glucose Metabolism
    'LBXGLU',      # Fasting glucose (mg/dL)
    'LBXGH',       # Glycohemoglobin (HbA1c) (%)
    'LBXIN',       # Insulin (μU/mL)
    
    # Complete Blood Count
    'LBXWBCSI',    # White blood cell count (1000 cells/μL)
    'LBXLYPCT',    # Lymphocyte percent (%)
    'LBXMOPCT',    # Monocyte percent (%)
    'LBXNEPCT',    # Neutrophil percent (%)
    'LBXEOPCT',    # Eosinophil percent (%)
    'LBXBAPCT',    # Basophil percent (%)
    'LBXRBCSI',    # Red blood cell count (million cells/μL)
    'LBXHGB',      # Hemoglobin (g/dL)
    'LBXHCT',      # Hematocrit (%)
    'LBXMCVSI',    # Mean cell volume (fL)
    'LBXMCHSI',    # Mean cell hemoglobin (pg)
    'LBXRDW',      # Red cell distribution width (%)
    'LBXPLTSI',    # Platelet count (1000 cells/μL)
    'LBXNRBC',     # Nucleated red blood cells (per 100 WBC)
    
    # Comprehensive Metabolic Panel
    'LBXSATSI',    # Alanine aminotransferase ALT (U/L)
    'LBXSASSI',    # Aspartate aminotransferase AST (U/L)
    'LBXSAPSI',    # Alkaline phosphatase (U/L)
    'LBXSGTSI',    # Gamma glutamyl transferase GGT (U/L)
    'LBXSAL',      # Albumin (g/dL)
    'LBXSTP',      # Total protein (g/dL)
    'LBXSTB',      # Total bilirubin (mg/dL)
    'LBXSBU',      # Blood urea nitrogen (mg/dL)
    'LBXSCR',      # Creatinine (mg/dL)
    'LBXSUA',      # Uric acid (mg/dL)
    'LBXSGL',      # Glucose, serum (mg/dL)
    'LBXSCA',      # Total calcium (mg/dL)
    'LBXSPH',      # Phosphorus (mg/dL)
    'LBXSKSI',     # Potassium (mmol/L)
    'LBXSNASI',    # Sodium (mmol/L)
    'LBXSCLSI',    # Chloride (mmol/L)
    'LBXSC3SI',    # Bicarbonate (mmol/L)
    'LBXSIR',      # Iron, refrigerated serum (μg/dL)
    
    # Urinary Markers
    'URXUCR',      # Urine creatinine (mg/dL)
    'URXUMA',      # Urine albumin (μg/mL)
    
    # Diabetes Questionnaire (DIQ)
    'DIQ010',      # Doctor told you have diabetes
    'DIQ160',      # Ever told you have prediabetes
    'DIQ170',      # Ever told have health risk for diabetes
    'DIQ172',      # Feel could be at risk for diabetes
    'DIQ050',      # Taking insulin now
    'DIQ070',      # Take diabetic pills to lower blood sugar
    'DIQ230',      # How long taking insulin
    
    # Medical Conditions Questionnaire (MCQ)
    'MCQ080',      # Doctor ever said you were overweight
    'MCQ160B',     # Ever told had congestive heart failure
    'MCQ160C',     # Ever told you had coronary heart disease
    'MCQ160D',     # Ever told you had angina/angina pectoris
    'MCQ160E',     # Ever told you had heart attack
    'MCQ160F',     # Ever told you had a stroke
    'MCQ160L',     # Ever told you had any liver condition (NAFLD related to diabetes)
    'MCQ160M',     # Ever told you had thyroid problem (affects metabolism)
    'MCQ160N',     # Ever told you had gout (related to metabolic syndrome)
    'MCQ300A',     # Close relative had heart attack
    'MCQ300C',     # Close relative had diabetes
    
    # Cardiovascular Health Questionnaire (CDQ)
    'CDQ001',      # Ever had pain or discomfort in chest
    'CDQ002',      # Chest pain when walking uphill
    'CDQ003',      # Chest pain when walking normal pace
    'CDQ004',      # Chest pain when standing still
    'CDQ005',      # Chest discomfort in arms/jaw/neck/back
    'CDQ006',      # Severity of chest pain
    'CDQ008',      # Shortness of breath on stairs/inclines
    'CDQ010',      # Shortness of breath during other activities
    
    # Smoking Questionnaire (SMQ)
    'SMQ020',      # Smoked at least 100 cigarettes in life
    'SMQ040',      # Do you now smoke cigarettes
    
    # Alcohol Use Questionnaire (ALQ)
    'ALQ121',      # Past 12 mo how often drink any alcohol
    'ALQ130',      # Avg # alcoholic drinks/day - past 12 mos
    
    # Diet Behavior Questionnaire (DBQ)
    'DBD895',      # # of meals not home prepared
    'DBD900',      # # of meals from fast food or pizza place
    
    # Physical Activity Questionnaire (PAQ)
    'PAQ605',      # Vigorous work activity
    'PAQ620',      # Moderate work activity
    'PAQ635',      # Walk or bicycle
    'PAQ650',      # Vigorous recreational activities
    'PAQ665',      # Moderate recreational activities
    
    # Weight History Questionnaire (WHQ)
    'WHD010',      # Current self-reported height (inches)
    'WHD020',      # Current self-reported weight (pounds)
    'WHD050',      # Self-reported weight - 1 yr ago (pounds)
    'WHD140',      # Self-reported greatest weight (pounds)
    
    # Outcome Variables
    'mortstat',
    'diabetes',
    'hyperten', 
    
    # keep follow-up time and derived death flag
    'permth_exm',   # person-months from exam to death/censoring
    'permth_int',   # person-months from interview to death/censoring
    'died',         # binary death indicator created in download_mortality.py
    
    # survey design
    'SDMVSTRA',    # masked variance pseudo-stratum
    'SDMVPSU',     # masked variance pseudo-PSU
    'WTMEC2YR',    # MEC exam weight
    'WTINT2YR',    # interview weight
]

def cleanup_nhanes_data():
    # Define file paths
    raw_data_dir = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
    input_file = raw_data_dir / "nhanes_with_mortality.csv"
    processed_data_dir = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_file = processed_data_dir / os.getenv("DATASET_NAME", "nhanes.csv")
    
    print("Loading NHANES data with mortality...")
    df = pd.read_csv(input_file, low_memory=False)
    
    print(f"Original dataset shape: {df.shape}")
    
    if 'eligstat' in df.columns:
        before = len(df)
        eligible_mask = df['eligstat'].astype(str) == '1'
        df = df[eligible_mask].copy()
        print(f"Filtered to linkage-eligible participants: {before} -> {len(df)}")
    
    if 'RIDAGEYR' in df.columns:
        before = len(df)
        df = df[df['RIDAGEYR'] >= 18].copy()
        print(f"Filtered to adults: {before} -> {len(df)}")
    
    print(f"Original columns: {df.shape[1]}")
    
    # Find which variables exist in the dataset
    existing_vars = []
    for var in ESSENTIAL_VARIABLES:
        if var in df.columns and var not in existing_vars:
            existing_vars.append(var)
    missing_vars = [var for var in ESSENTIAL_VARIABLES if var not in df.columns]
    
    print(f"\nVariables found: {len(existing_vars)}/{len(ESSENTIAL_VARIABLES)}")
    
    if missing_vars:
        print(f"\n⚠️ Warning: {len(missing_vars)} variables not found in dataset:")
        for var in missing_vars:
            print(f"  - {var}")
    
    # Keep only existing essential variables
    df_cleaned = df[existing_vars].copy()

    for cycle_col in ('survey_cycle', 'mortality_cycle'):
        if cycle_col in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=cycle_col)
    
    print(f"\nCleaned dataset shape: {df_cleaned.shape}")
    print(f"Columns retained: {df_cleaned.shape[1]}")
    
    # Drop columns with more than 50% missing data
    print("\nDropping columns with >50% missing data...")
    missing_pct = (df_cleaned.isnull().sum() / len(df_cleaned)) * 100
    
    protected_cols = {
        'mortstat', 
        'diabetes', 
        'hyperten', 
        'permth_exm', 
        'permth_int', 
        'died', 
        'SDMVSTRA', 
        'SDMVPSU', 
        'WTMEC2YR',
        'WTINT2YR'
    }

    cols_to_drop = [
        col for col in missing_pct[missing_pct > 50].index
        if col not in protected_cols
    ]
    
    if cols_to_drop:
        print(f"  Dropping {len(cols_to_drop)} columns:")
        for col in cols_to_drop:
            pct = missing_pct[col]
            print(f"    - {col}: {pct:.1f}% missing")
        df_cleaned = df_cleaned.drop(columns=cols_to_drop)
        print(f"  ✅ Columns after dropping: {df_cleaned.shape[1]}")
    else:
        print("  ✅ No columns with >50% missing data")
    
    # Drop rows with more than 60% missing data
    print("\nDropping rows with >60% missing data...")
    initial_rows = len(df_cleaned)
    
    print("\nDropping rows with >60% missing feature data...")

    core_cols = [
        'SEQN', 'time', 'event', 'mortstat', 'permth_exm', 'permth_int', 'eligstat'
    ]
    core_cols = [c for c in core_cols if c in df_cleaned.columns]

    feature_cols = [c for c in df_cleaned.columns if c not in core_cols]

    if feature_cols:
        initial_rows = len(df_cleaned)
        # Count non-missing feature values per row
        nonmissing_features = df_cleaned[feature_cols].notna().sum(axis=1)
        feature_threshold = int(len(feature_cols) * 0.4)  # require at least 40% of features

        df_cleaned = df_cleaned[nonmissing_features >= feature_threshold].copy()

        rows_dropped = initial_rows - len(df_cleaned)
        print(f"  Dropped {rows_dropped:,} rows ({rows_dropped/initial_rows*100:.1f}%) based on features")
        print(f"  Remaining rows: {len(df_cleaned):,}")
    else:
        print("  No feature columns; skipping row-wise missingness filtering")
    
    # Create averaged blood pressure variables
    print("\nCreating averaged blood pressure variables...")
    
    # Average systolic blood pressure (BPXSY1-4)
    systolic_cols = [col for col in ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4'] if col in df_cleaned.columns]
    if systolic_cols:
        df_cleaned['BPXSY'] = df_cleaned[systolic_cols].mean(axis=1, skipna=True)
        print(f"  ✅ Created BPXSY (average of {len(systolic_cols)} readings)")
        # Drop individual readings
        df_cleaned = df_cleaned.drop(columns=systolic_cols)
    
    # Average diastolic blood pressure (BPXDI1-4)
    diastolic_cols = [col for col in ['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4'] if col in df_cleaned.columns]
    if diastolic_cols:
        df_cleaned['BPXDI'] = df_cleaned[diastolic_cols].mean(axis=1, skipna=True)
        print(f"  ✅ Created BPXDI (average of {len(diastolic_cols)} readings)")
        # Drop individual readings
        df_cleaned = df_cleaned.drop(columns=diastolic_cols)
    
    
    print("\nCreating survival analysis variables (time, event)...")

    # Time in months: prefer exam-based follow-up if available
    if 'permth_exm' in df_cleaned.columns:
        df_cleaned['time'] = df_cleaned['permth_exm'].astype('float')
    elif 'permth_int' in df_cleaned.columns:
        df_cleaned['time'] = df_cleaned['permth_int'].astype('float')
    else:
        print("  ⚠️ WARNING: permth_exm/permth_int not found; 'time' will be missing.")
        df_cleaned['time'] = pd.NA

    # Event indicator: 1 = died, 0 = censored
    if 'mortstat' in df_cleaned.columns:
        # Public-use LMF: 0=assumed alive, 1=assumed deceased, missing=not linkage-eligible
        df_cleaned['event'] = (df_cleaned['mortstat'] == 1).astype('int')
    elif 'died' in df_cleaned.columns:
        df_cleaned['event'] = df_cleaned['died'].astype('int')
    else:
        print(" ⚠️ WARNING: mortstat/died not found; 'event' will be missing.")
        df_cleaned['event'] = pd.NA
        
    if {'time', 'event'}.issubset(df_cleaned.columns):
        before = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned['time'].notna() & df_cleaned['event'].notna()].copy()
        df_cleaned = df_cleaned[df_cleaned['time'] > 0].copy()
        print(f"Filtered to rows with valid time & event: {before} -> {len(df_cleaned)}")
        
    print(f"\nFinal dataset shape after averaging: {df_cleaned.shape}")
    
    # Save cleaned dataset
    print(f"\nSaving cleaned data to: {output_file}")
    df_cleaned.to_csv(output_file, index=False)
    
    print("✅ Data cleanup complete!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Rows: {df_cleaned.shape[0]:,}")
    print(f"Columns: {df_cleaned.shape[1]}")
    print(f"Memory usage: {df_cleaned.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    missing_counts = df_cleaned.isnull().sum()
    vars_with_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    if len(vars_with_missing) > 0:
        print(f"\nVariables with missing values: {len(vars_with_missing)}")
        print("\nTop 10 variables with most missing values:")
        for var, count in vars_with_missing.head(10).items():
            pct = (count / len(df_cleaned)) * 100
            print(f"  {var}: {count:,} ({pct:.1f}%)")
    else:
        print("\n✅ No missing values found")
        
    assert df_cleaned['time'].min() > 0, "Found non-positive survival times"
    assert set(df_cleaned['event'].unique()) <= {0, 1}, "Event must be 0/1 only"
    assert set(df_cleaned['mortstat'].dropna().unique()) <= {0, 1}, "mortstat must be 0/1 in final dataset"


if __name__ == "__main__":
    cleanup_nhanes_data()
