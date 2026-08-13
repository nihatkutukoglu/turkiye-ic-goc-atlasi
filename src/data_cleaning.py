"""Ham TÜİK tablolarını analize hazır, doğrulanmış CSV'lere dönüştürür."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    END_YEAR,
    EXPECTED_YEARS,
    PROCESSED_DATA_DIR,
    PROVINCES,
    RAW_DATA_DIR,
    ensure_output_directories,
)
from src.feature_engineering import build_city_year_metrics


FLOW_COLUMNS = [
    "year",
    "destination_province",
    "origin_province",
    "destination_population",
    "origin_population",
    "migration_flow",
    "reverse_migration_flow",
    "bilateral_net_migration",
]

AGE_GROUP_PATTERN = re.compile(
    r"^(0-4|5-9|10-14|15-19|20-24|25-29|30-34|35-39|40-44|"
    r"45-49|50-54|55-59|60-64|65\+)$"
)

MIGRATION_REASONS = [
    "Tayin / iş değişikliği",
    "İşe başlamak / iş bulmak",
    "Eğitim",
    "Medeni durum değişikliği / ailevi nedenler",
    "Daha iyi konut ve yaşam koşulları",
    "Hane / aile fertlerinden birine bağımlı göç",
    "Aile yanına / memlekete geri dönme",
    "Sağlık / bakım",
    "Ev alınması",
    "Emeklilik",
    "Diğer",
    "Bilinmeyen",
]

EDUCATION_LEVELS = [
    "Okuma yazma bilmeyen",
    "Okuma yazma bilen fakat bir okul bitirmeyen",
    "İlkokul",
    "İlköğretim, ortaokul veya dengi okul",
    "Lise veya dengi okul",
    "Yükseköğretim",
    "Bilinmeyen",
]


def _turkish_upper(value: object) -> str:
    text = str(value).strip().replace("i", "İ").replace("ı", "I")
    return text.upper()


def _numeric(series: pd.Series, *, integer: bool = True) -> pd.Series:
    """Kaynak tablodaki bilgi-yok işaretlerini eksik değer olarak korur."""

    missing_tokens = {"-": pd.NA, "–": pd.NA, "—": pd.NA, "": pd.NA}
    result = pd.to_numeric(series.replace(missing_tokens), errors="coerce")
    return result.astype("Int64") if integer else result.astype(float)


def _assert_no_metadata(frame: pd.DataFrame) -> None:
    object_values = frame.select_dtypes(include="object").astype(str)
    metadata_pattern = r"TÜİK|TurkStat|Internal Migration Statistics|İç Göç İstatistikleri"
    assert not object_values.apply(
        lambda column: column.str.contains(metadata_pattern, case=False, regex=True).any()
    ).any(), "Processed tabloda kaynak veya dipnot satırı bulundu."


def clean_migration_flows(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, header=1)
    assert frame.shape[1] == len(FLOW_COLUMNS)
    frame.columns = FLOW_COLUMNS
    frame["destination_province"] = frame["destination_province"].map(
        _turkish_upper
    )
    frame["origin_province"] = frame["origin_province"].map(_turkish_upper)
    numeric_columns = [column for column in FLOW_COLUMNS if "province" not in column]
    for column in numeric_columns:
        frame[column] = _numeric(frame[column])
    frame = frame.sort_values(
        ["year", "destination_province", "origin_province"], ignore_index=True
    )
    validate_migration_flows(frame)
    return frame


def validate_migration_flows(frame: pd.DataFrame) -> None:
    assert list(frame.columns) == FLOW_COLUMNS
    assert not frame.isna().any().any()
    assert tuple(sorted(frame["year"].unique())) == EXPECTED_YEARS
    assert set(frame["destination_province"]) == set(PROVINCES)
    assert set(frame["origin_province"]) == set(PROVINCES)
    assert not frame.duplicated(
        ["year", "destination_province", "origin_province"]
    ).any()
    assert not frame["destination_province"].eq(frame["origin_province"]).any()
    expected_per_year = len(PROVINCES) * (len(PROVINCES) - 1)
    assert frame.groupby("year").size().eq(expected_per_year).all()

    nonnegative = [
        "destination_population",
        "origin_population",
        "migration_flow",
        "reverse_migration_flow",
    ]
    assert (frame[nonnegative] >= 0).all().all()
    assert frame["bilateral_net_migration"].eq(
        frame["migration_flow"] - frame["reverse_migration_flow"]
    ).all()

    mirror = frame[
        ["year", "destination_province", "origin_province", "migration_flow"]
    ].copy()
    mirror.columns = [
        "year",
        "origin_province",
        "destination_province",
        "mirror_flow",
    ]
    paired = frame.merge(
        mirror,
        on=["year", "destination_province", "origin_province"],
        validate="one_to_one",
    )
    assert paired["reverse_migration_flow"].eq(paired["mirror_flow"]).all()
    _assert_no_metadata(frame)


def clean_city_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, header=2)
    frame = frame.iloc[:, :6].copy()
    frame.columns = [
        "province",
        "population",
        "in_migration",
        "out_migration",
        "net_migration",
        "net_migration_rate",
    ]
    frame["province"] = frame["province"].map(_turkish_upper)
    frame = frame.loc[frame["province"].isin(PROVINCES)].copy()
    for column in frame.columns[1:-1]:
        frame[column] = _numeric(frame[column])
    frame["net_migration_rate"] = _numeric(
        frame["net_migration_rate"], integer=False
    )
    frame.insert(0, "year", END_YEAR)
    frame = frame.sort_values("province", ignore_index=True)

    assert len(frame) == len(PROVINCES)
    assert set(frame["province"]) == set(PROVINCES)
    assert not frame.isna().any().any()
    assert frame["net_migration"].eq(
        frame["in_migration"] - frame["out_migration"]
    ).all()
    mid_period_population = frame["population"] - frame["net_migration"] / 2
    assert np.allclose(
        frame["net_migration_rate"],
        1_000 * frame["net_migration"] / mid_period_population,
        atol=1e-6,
    )
    _assert_no_metadata(frame)
    return frame


def clean_age_gender(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, header=2).iloc[:, :4].copy()
    frame.columns = ["age_group", "total", "male", "female"]
    frame["age_group"] = frame["age_group"].astype(str).str.strip()
    frame = frame.loc[frame["age_group"].str.match(AGE_GROUP_PATTERN)].copy()
    for column in ("total", "male", "female"):
        frame[column] = _numeric(frame[column])
    frame.insert(0, "year", END_YEAR)
    frame = frame.reset_index(drop=True)

    assert len(frame) == 14
    assert not frame.isna().any().any()
    assert frame["total"].eq(frame["male"] + frame["female"]).all()
    assert (frame[["total", "male", "female"]] >= 0).all().all()
    _assert_no_metadata(frame)
    return frame


def clean_age_gender_reason(path: Path) -> pd.DataFrame:
    source = pd.read_excel(path, header=2).iloc[:, :15].copy()
    source.columns = ["age_group", "sex", "total", *MIGRATION_REASONS]
    source["age_group"] = source["age_group"].ffill().astype(str).str.strip()
    source = source.loc[source["age_group"].str.match(AGE_GROUP_PATTERN)].copy()
    source["sex"] = source["sex"].astype(str).str.strip()
    source = source.loc[source["sex"].str.contains("Erkek|Kadın", regex=True)].copy()
    source["sex"] = np.where(source["sex"].str.contains("Erkek"), "Erkek", "Kadın")
    source["total"] = _numeric(source["total"])
    for reason in MIGRATION_REASONS:
        source[reason] = _numeric(source[reason])

    reason_sum = source[MIGRATION_REASONS].sum(axis=1, min_count=1)
    assert reason_sum.eq(source["total"]).all()

    tidy = source.melt(
        id_vars=["age_group", "sex"],
        value_vars=MIGRATION_REASONS,
        var_name="reason",
        value_name="count",
    )
    tidy.insert(0, "year", END_YEAR)
    tidy = tidy.sort_values(["age_group", "sex", "reason"], ignore_index=True)
    assert tidy["count"].dropna().ge(0).all()
    assert not tidy[["year", "age_group", "sex", "reason"]].isna().any().any()
    assert not tidy["sex"].str.contains("Toplam", case=False).any()
    _assert_no_metadata(tidy)
    return tidy


def clean_education_reason(path: Path) -> pd.DataFrame:
    source = pd.read_excel(path, header=3).iloc[:, :9].copy()
    source.columns = ["reason", "total", *EDUCATION_LEVELS]
    source["reason"] = source["reason"].astype(str).str.strip()
    source = source.loc[~source["reason"].str.contains("Toplam", case=False)].copy()
    source = source.iloc[: len(MIGRATION_REASONS)].copy()
    source["reason"] = MIGRATION_REASONS
    source["total"] = _numeric(source["total"])
    for level in EDUCATION_LEVELS:
        source[level] = _numeric(source[level])

    education_sum = source[EDUCATION_LEVELS].sum(axis=1, min_count=1)
    assert education_sum.eq(source["total"]).all()

    tidy = source.melt(
        id_vars=["reason"],
        value_vars=EDUCATION_LEVELS,
        var_name="education_level",
        value_name="count",
    )
    tidy.insert(0, "year", END_YEAR)
    tidy = tidy.sort_values(["reason", "education_level"], ignore_index=True)
    assert tidy["count"].dropna().ge(0).all()
    assert not tidy[["year", "reason", "education_level"]].isna().any().any()
    _assert_no_metadata(tidy)
    return tidy


def _validate_summary_against_flows(
    summary: pd.DataFrame, city_year: pd.DataFrame
) -> None:
    latest = city_year.loc[city_year["year"].eq(END_YEAR)].copy()
    columns = [
        "population",
        "in_migration",
        "out_migration",
        "net_migration",
        "net_migration_rate",
    ]
    comparison = latest.merge(
        summary, on=["year", "province"], suffixes=("_derived", "_official")
    )
    assert len(comparison) == len(PROVINCES)
    for column in columns:
        assert np.allclose(
            comparison[f"{column}_derived"],
            comparison[f"{column}_official"],
            atol=1e-6,
        ), f"2025 resmi özet ile {column} uyuşmuyor."


def build_processed_datasets(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Beş ham kaynağı temizler, doğrular ve processed katmana yazar."""

    ensure_output_directories()
    datasets = {
        "iller_arasi_goc.csv": clean_migration_flows(
            raw_dir / "iller_arasi_goc.xlsx"
        ),
        "illerin_goc_ozeti.csv": clean_city_summary(
            raw_dir / "illerin_goc_ozeti.xls"
        ),
        "yas_cinsiyet_goc.csv": clean_age_gender(
            raw_dir / "yas_cinsiyet_goc.xls"
        ),
        "yas_cinsiyet_neden.csv": clean_age_gender_reason(
            raw_dir / "yas_cinsiyet_neden.xls"
        ),
        "egitim_goc_nedeni.csv": clean_education_reason(
            raw_dir / "egitim_goc_nedeni.xls"
        ),
    }
    datasets["city_year_metrics.csv"] = build_city_year_metrics(
        datasets["iller_arasi_goc.csv"]
    )
    _validate_summary_against_flows(
        datasets["illerin_goc_ozeti.csv"], datasets["city_year_metrics.csv"]
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in datasets.items():
        frame.to_csv(processed_dir / filename, index=False)
    return datasets


def main() -> None:
    datasets = build_processed_datasets()
    details = ", ".join(
        f"{filename}: {len(frame):,} satır" for filename, frame in datasets.items()
    )
    print(f"Processed veri setleri doğrulandı ve yazıldı. {details}")


if __name__ == "__main__":
    main()
