import pandas as pd
from pathlib import Path

def convert_excel_to_csv():
    """
    Reads the raw Excel datasets from TÜİK, skips the necessary title rows,
    and saves them as clean CSV files in the processed directory.
    """
    # Define paths
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    
    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary of files and how many header rows to skip
    files_mapping = {
        "iller_arasi_goc.xlsx": 1,
        "illerin_goc_ozeti.xls": 2,
        "yas_cinsiyet_goc.xls": 2,
        "yas_cinsiyet_neden.xls": 2,
        "egitim_goc_nedeni.xls": 3, # Checking egitim_goc_nedeni.xls: row 0 is english, row 1 is [6+ yaş-age], row 2 is probably the actual columns? Wait, let's just use 2 for now, or check first.
    }
    
    for excel_file, skiprows in files_mapping.items():
        raw_path = raw_dir / excel_file
        if not raw_path.exists():
            print(f"File not found: {raw_path}")
            continue
            
        csv_filename = excel_file.split(".")[0] + ".csv"
        processed_path = processed_dir / csv_filename
        
        print(f"Processing {excel_file}...")
        
        # Read with correct skiprows
        df = pd.read_excel(raw_path, skiprows=skiprows)
        
        # Clean column names (remove newlines and excess spaces, e.g., 'İl\nProvince' -> 'İl Province')
        df.columns = [str(col).replace('\n', ' ').strip() for col in df.columns]
        
        # Save to CSV
        df.to_csv(processed_path, index=False)
        print(f" -> Saved to {csv_filename} with shape {df.shape}")

if __name__ == "__main__":
    convert_excel_to_csv()
