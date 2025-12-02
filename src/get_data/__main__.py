from .download_nhanes import download_nhanes
from .download_mortality import download_mortality
from .cleanup import cleanup_nhanes_data

def main():
    from dotenv import load_dotenv
    load_dotenv("config.env")
    
    print("Starting data download and processing...\n")
    download_nhanes()
    
    print("\nDownloading mortality data and merging with NHANES...\n")
    download_mortality()
    
    print("\nCleaning up NHANES data...\n")
    cleanup_nhanes_data()
    
    print("\nData download and processing complete.")
    
if __name__ == "__main__":
    main()
