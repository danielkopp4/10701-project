from .download_nhanes import download_nhanes
from .download_mortality import download_mortality
from .cleanup import cleanup_nhanes_data

def main():
    from dotenv import load_dotenv
    load_dotenv("config.env")
    
    download_nhanes()
    download_mortality()
    cleanup_nhanes_data()
