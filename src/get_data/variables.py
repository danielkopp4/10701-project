"""
Show all variables available in the merged NHANES dataset
"""

import pandas as pd

# Load just the column names (don't load full data)
df = pd.read_csv('../nhanes/nhanes_2017_2018_merged.csv')

print('='*80)
print(f'TOTAL VARIABLES PER PATIENT: {len(df.columns)}')
print(f'TOTAL PATIENTS: {len(df):,}')
print('='*80)

# Categorize variables
categories = {}

for col in df.columns:
    # Determine category based on prefix
    if col == 'SEQN':
        cat = 'ID'
    elif col.startswith('RIA') or col.startswith('RID') or col.startswith('SDM') or col.startswith('DMD'):
        cat = 'Demographics'
    elif col.startswith('BMX'):
        cat = 'Body Measurements'
    elif col.startswith('BPX'):
        cat = 'Blood Pressure'
    elif col.startswith('LBX') or col.startswith('LBD'):
        cat = 'Laboratory'
    elif col.startswith('PAQ'):
        cat = 'Physical Activity'
    elif col.startswith('SMQ'):
        cat = 'Smoking'
    elif col.startswith('ALQ'):
        cat = 'Alcohol'
    elif col.startswith('DBQ'):
        cat = 'Diet'
    elif col.startswith('MCQ'):
        cat = 'Medical Conditions'
    elif col.startswith('DIQ'):
        cat = 'Diabetes'
    elif col.startswith('CDQ'):
        cat = 'Cardiovascular'
    elif col.startswith('WHQ'):
        cat = 'Weight History'
    else:
        cat = 'Other'
    
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(col)

# Print by category
for cat in sorted(categories.keys()):
    cols = categories[cat]
    print(f'\n{cat.upper()} ({len(cols)} variables)')
    print('-' * 80)
    
    for col in sorted(cols)[:50]:  # Show first 50 per category
        non_null = df[col].notna().sum()
        pct = 100 * non_null / len(df)
        print(f'  {col:30s} {non_null:6,} non-null ({pct:5.1f}%)')
    
    if len(cols) > 50:
        print(f'  ... and {len(cols) - 50} more variables')

# Key variables for BMI metric
print('\n' + '='*80)
print('KEY VARIABLES FOR DEVELOPING BMI ALTERNATIVE')
print('='*80)

key_vars = [
    ('SEQN', 'Patient ID'),
    ('RIDAGEYR', 'Age (years)'),
    ('RIAGENDR', 'Gender'),
    ('BMXBMI', 'BMI (kg/m²)'),
    ('BMXWT', 'Weight (kg)'),
    ('BMXHT', 'Height (cm)'),
    ('BMXWAIST', 'Waist Circumference (cm)'),
    ('BMXHIP', 'Hip Circumference (cm)'),
    ('BPXSY1', 'Systolic BP'),
    ('BPXDI1', 'Diastolic BP'),
    ('LBXTC', 'Total Cholesterol'),
    ('LBXGLU', 'Fasting Glucose'),
    ('LBXGH', 'Hemoglobin A1c'),
]

print('\nVariable                       Description                    Availability')
print('-' * 80)
for var, desc in key_vars:
    if var in df.columns:
        non_null = df[var].notna().sum()
        pct = 100 * non_null / len(df)
        print(f'{var:30s} {desc:30s} {non_null:6,} ({pct:5.1f}%)')
    else:
        print(f'{var:30s} {desc:30s} NOT FOUND')

print('\n' + '='*80)
print(f'Complete list saved. Total unique variables: {len(df.columns)}')
print('='*80)
