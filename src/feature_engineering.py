"""İl-yıl metrikleri ve uzun dönemli model özellikleri."""

import numpy as np
import pandas as pd

from src.config import END_YEAR, EXPECTED_YEARS, PROVINCES


CITY_YEAR_COLUMNS = [
    "year",
    "province",
    "population",
    "in_migration",
    "out_migration",
    "net_migration",
    "migration_turnover",
    "in_migration_rate",
    "out_migration_rate",
    "net_migration_rate",
    "turnover_rate",
    "in_out_ratio",
]

MODEL_FEATURES = [
    "mean_turnover_rate",
    "mean_net_migration_rate",
    "net_migration_rate_volatility",
    "net_migration_rate_trend",
    "recent_net_migration_rate",
    "latest_population_log",
]


def build_city_year_metrics(flows: pd.DataFrame) -> pd.DataFrame:
    """Tekil yönlü akışlardan 81 il × 18 yıl paneli üretir."""

    metrics = (
        flows.groupby(["year", "destination_province"], as_index=False)
        .agg(
            population=("destination_population", "first"),
            in_migration=("migration_flow", "sum"),
            out_migration=("reverse_migration_flow", "sum"),
            net_migration=("bilateral_net_migration", "sum"),
        )
        .rename(columns={"destination_province": "province"})
    )
    metrics["migration_turnover"] = (
        metrics["in_migration"] + metrics["out_migration"]
    )
    metrics["in_migration_rate"] = (
        1_000 * metrics["in_migration"] / metrics["population"]
    )
    metrics["out_migration_rate"] = (
        1_000 * metrics["out_migration"] / metrics["population"]
    )
    # TÜİK'in dönem ortası nüfus yaklaşımı: P - net_göç / 2.
    mid_period_population = metrics["population"] - metrics["net_migration"] / 2
    metrics["net_migration_rate"] = (
        1_000 * metrics["net_migration"] / mid_period_population
    )
    metrics["turnover_rate"] = (
        1_000 * metrics["migration_turnover"] / metrics["population"]
    )
    metrics["in_out_ratio"] = metrics["in_migration"] / metrics["out_migration"]
    metrics = metrics[CITY_YEAR_COLUMNS].sort_values(
        ["year", "province"], ignore_index=True
    )
    validate_city_year_metrics(metrics)
    return metrics


def validate_city_year_metrics(metrics: pd.DataFrame) -> None:
    expected_rows = len(EXPECTED_YEARS) * len(PROVINCES)
    assert len(metrics) == expected_rows, f"Beklenen {expected_rows} il-yıl satırı yok."
    assert tuple(sorted(metrics["year"].unique())) == EXPECTED_YEARS
    assert set(metrics["province"]) == set(PROVINCES)
    assert not metrics.duplicated(["year", "province"]).any()
    assert not metrics.isna().any().any()
    nonnegative = [
        "population",
        "in_migration",
        "out_migration",
        "migration_turnover",
        "in_migration_rate",
        "out_migration_rate",
        "turnover_rate",
        "in_out_ratio",
    ]
    assert (metrics[nonnegative] >= 0).all().all()
    assert np.allclose(
        metrics["net_migration"],
        metrics["in_migration"] - metrics["out_migration"],
    )
    assert metrics.groupby("year").size().eq(len(PROVINCES)).all()


def build_long_term_features(metrics: pd.DataFrame) -> pd.DataFrame:
    """Her ilin 2008–2025 davranışını tek bir profil satırında özetler."""

    validate_city_year_metrics(metrics)
    profiles = []
    for province, history in metrics.groupby("province", sort=True):
        history = history.sort_values("year")
        trend = np.polyfit(
            history["year"].to_numpy(),
            history["net_migration_rate"].to_numpy(),
            deg=1,
        )[0]
        latest = history.loc[history["year"].eq(END_YEAR)].iloc[0]
        profiles.append(
            {
                "province": province,
                "mean_turnover_rate": history["turnover_rate"].mean(),
                "mean_net_migration_rate": history["net_migration_rate"].mean(),
                "net_migration_rate_volatility": history[
                    "net_migration_rate"
                ].std(ddof=0),
                "net_migration_rate_trend": trend,
                "recent_net_migration_rate": history.tail(3)[
                    "net_migration_rate"
                ].mean(),
                "latest_population_log": np.log1p(latest["population"]),
                "latest_population": int(latest["population"]),
                "latest_in_migration": int(latest["in_migration"]),
                "latest_out_migration": int(latest["out_migration"]),
                "latest_net_migration": int(latest["net_migration"]),
                "latest_net_migration_rate": latest["net_migration_rate"],
            }
        )

    features = pd.DataFrame(profiles).sort_values("province", ignore_index=True)
    assert len(features) == len(PROVINCES)
    assert set(features["province"]) == set(PROVINCES)
    assert not features[MODEL_FEATURES].isna().any().any()
    return features
