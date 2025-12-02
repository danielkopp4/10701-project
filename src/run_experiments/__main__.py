from .experiments import run_experiments

if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    import os
    
    
    load_dotenv("config.env")
    
    csv_path = (Path(os.getenv('PROCESSED_DATA_PATH', 'data/processed')) 
        / os.getenv('PROCESSED_DATA_FILE', 'nhanes.csv')
    )
    
    run_experiments(
        csv_path,
        synthetic_n=2000,
        T_eval_years=5
    )