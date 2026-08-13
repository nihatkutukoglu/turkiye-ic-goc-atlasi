import pandas as pd
from pathlib import Path

def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).resolve().parent.parent

def load_migration_network() -> pd.DataFrame:
    """Loads the main migration network dataset."""
    root = get_project_root()
    path = root / "data" / "processed" / "iller_arasi_goc.csv"
    return pd.read_csv(path)

def load_migration_summary() -> pd.DataFrame:
    """Loads the migration summary by city dataset."""
    root = get_project_root()
    path = root / "data" / "processed" / "illerin_goc_ozeti.csv"
    return pd.read_csv(path)

def load_age_gender() -> pd.DataFrame:
    """Loads the migration by age and gender dataset."""
    root = get_project_root()
    path = root / "data" / "processed" / "yas_cinsiyet_goc.csv"
    return pd.read_csv(path)

def load_age_gender_reason() -> pd.DataFrame:
    """Loads the migration by age, gender, and reason dataset."""
    root = get_project_root()
    path = root / "data" / "processed" / "yas_cinsiyet_neden.csv"
    return pd.read_csv(path)

def load_education_reason() -> pd.DataFrame:
    """Loads the migration by education and reason dataset."""
    root = get_project_root()
    path = root / "data" / "processed" / "egitim_goc_nedeni.csv"
    return pd.read_csv(path)
