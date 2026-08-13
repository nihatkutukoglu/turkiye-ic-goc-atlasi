"""Temizlenmiş proje tabloları için merkezi veri yükleyiciler."""

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR


def _load_csv(filename: str, data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bulunamadı. Önce `python -m src.data_cleaning` çalıştırın."
        )
    return pd.read_csv(path)


def load_migration_flows(data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    return _load_csv("iller_arasi_goc.csv", data_dir)


def load_city_summary(data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    return _load_csv("illerin_goc_ozeti.csv", data_dir)


def load_city_year_metrics(data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    return _load_csv("city_year_metrics.csv", data_dir)


def load_age_gender(data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    return _load_csv("yas_cinsiyet_goc.csv", data_dir)


def load_age_gender_reason(data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    return _load_csv("yas_cinsiyet_neden.csv", data_dir)


def load_education_reason(data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    return _load_csv("egitim_goc_nedeni.csv", data_dir)


def load_city_profiles(data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    return _load_csv("city_profiles.csv", data_dir)
