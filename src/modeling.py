"""Uzun dönemli il profilleri için clustering eğitim ve persistence hattı."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import END_YEAR, MODELS_DIR, PROCESSED_DATA_DIR, START_YEAR
from src.data_loader import load_city_year_metrics
from src.feature_engineering import MODEL_FEATURES, build_long_term_features


MODEL_VERSION = "1.0.0"
K_RANGE = range(2, 9)
RANDOM_STATE = 42

FEATURE_DESCRIPTIONS = {
    "mean_turnover_rate": "2008–2025 ortalama toplam hareketlilik hızı (‰)",
    "mean_net_migration_rate": "2008–2025 ortalama net göç hızı (‰)",
    "net_migration_rate_volatility": "Yıllık net göç hızı standart sapması",
    "net_migration_rate_trend": "Net göç hızının yıllık doğrusal eğilimi",
    "recent_net_migration_rate": "2023–2025 ortalama net göç hızı (‰)",
    "latest_population_log": "2025 nüfusunun log1p dönüşümü",
}


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("scale", StandardScaler(), MODEL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _stability_for_k(values: pd.DataFrame, k: int) -> float:
    scaled = StandardScaler().fit_transform(values[MODEL_FEATURES])
    label_sets = [
        KMeans(n_clusters=k, random_state=seed, n_init=30).fit_predict(scaled)
        for seed in range(10)
    ]
    comparisons = [
        adjusted_rand_score(label_sets[first], label_sets[second])
        for first in range(len(label_sets))
        for second in range(first + 1, len(label_sets))
    ]
    return float(np.mean(comparisons))


def evaluate_kmeans_candidates(features: pd.DataFrame) -> pd.DataFrame:
    """k=2–8 için içsel clustering metriklerini ve seed kararlılığını hesaplar."""

    scaled = StandardScaler().fit_transform(features[MODEL_FEATURES])
    rows = []
    for k in K_RANGE:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=50,
        ).fit(scaled)
        labels = model.labels_
        sizes = pd.Series(labels).value_counts()
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled, labels),
                "davies_bouldin": davies_bouldin_score(scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(scaled, labels),
                "stability_ari": _stability_for_k(features, k),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_share": float(sizes.max() / len(labels)),
            }
        )
    return pd.DataFrame(rows)


def select_cluster_count(evaluation: pd.DataFrame) -> int:
    """Metrik dengesi ve kullanışlı profil dağılımına göre k seçer."""

    candidates = evaluation.loc[
        evaluation["min_cluster_size"].ge(3)
        & evaluation["max_cluster_share"].le(0.75)
    ].copy()
    if candidates.empty:
        candidates = evaluation.copy()
    candidates["selection_rank"] = (
        candidates["silhouette"].rank(ascending=False)
        + candidates["davies_bouldin"].rank(ascending=True)
        + candidates["calinski_harabasz"].rank(ascending=False)
        + candidates["stability_ari"].rank(ascending=False)
    )
    selected = candidates.sort_values(
        ["selection_rank", "max_cluster_share", "k"]
    ).iloc[0]
    return int(selected["k"])


def _name_clusters(cluster_summary: pd.DataFrame) -> dict[int, str]:
    """Profil adlarını centroid sıralamalarından, sabit eşik kullanmadan üretir."""

    remaining = set(cluster_summary.index.astype(int))
    names: dict[int, str] = {}

    large = int(cluster_summary["latest_population"].idxmax())
    large_net = cluster_summary.loc[large, "mean_net_migration_rate"]
    names[large] = (
        "Büyük Ölçekli Çekim Merkezleri"
        if large_net > 0
        else "Yüksek Hacimli Metropoller"
    )
    remaining.discard(large)

    if remaining:
        loss = int(
            cluster_summary.loc[list(remaining), "recent_net_migration_rate"].idxmin()
        )
        names[loss] = "Süregelen Net Göç Veren İller"
        remaining.discard(loss)

    if remaining:
        dynamic = int(
            cluster_summary.loc[list(remaining), "mean_turnover_rate"].idxmax()
        )
        names[dynamic] = "Küçük Ölçekli Yüksek Hareketlilik"
        remaining.discard(dynamic)

    fallback = [
        "Dengeli Orta Hareketlilik",
        "Düşük Hareketlilik Profili",
        "Gelişen Çekim Profili",
        "Dalgalı Göç Profili",
    ]
    for cluster_id, name in zip(
        sorted(
            remaining,
            key=lambda item: cluster_summary.loc[item, "mean_turnover_rate"],
            reverse=True,
        ),
        fallback,
    ):
        names[int(cluster_id)] = name
    return names


def _evaluate_hdbscan(scaled: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    model = HDBSCAN(min_cluster_size=4, min_samples=2)
    labels = model.fit_predict(scaled)
    retained = labels.ne(-1) if isinstance(labels, pd.Series) else labels != -1
    cluster_count = len(set(labels[retained]))
    metrics: dict[str, float | int] = {
        "cluster_count": cluster_count,
        "noise_count": int((labels == -1).sum()),
        "min_cluster_size": 4,
        "min_samples": 2,
    }
    if cluster_count > 1:
        metrics["silhouette_non_noise"] = float(
            silhouette_score(scaled[retained], labels[retained])
        )
    return labels, metrics


def train_and_save_models(
    city_year: pd.DataFrame | None = None,
    models_dir: Path = MODELS_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, object]:
    """K-Means ve PCA pipeline'larını eğitir, kaydeder ve reload testi yapar."""

    city_year = city_year if city_year is not None else load_city_year_metrics()
    features = build_long_term_features(city_year)
    evaluation = evaluate_kmeans_candidates(features)
    selected_k = select_cluster_count(evaluation)

    cluster_pipeline = Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "cluster",
                KMeans(
                    n_clusters=selected_k,
                    random_state=RANDOM_STATE,
                    n_init=50,
                ),
            ),
        ]
    )
    labels = cluster_pipeline.fit_predict(features[MODEL_FEATURES])
    features["cluster_id"] = labels

    pca_pipeline = Pipeline(
        [("preprocess", _preprocessor()), ("pca", PCA(n_components=2))]
    )
    coordinates = pca_pipeline.fit_transform(features[MODEL_FEATURES])
    features["pc1"] = coordinates[:, 0]
    features["pc2"] = coordinates[:, 1]

    scaled = cluster_pipeline.named_steps["preprocess"].transform(
        features[MODEL_FEATURES]
    )
    _, hdbscan_metrics = _evaluate_hdbscan(scaled)
    selected_metrics = evaluation.loc[evaluation["k"].eq(selected_k)].iloc[0]

    summary_columns = [
        *MODEL_FEATURES,
        "latest_population",
        "latest_in_migration",
        "latest_out_migration",
        "latest_net_migration",
        "latest_net_migration_rate",
    ]
    cluster_summary = features.groupby("cluster_id")[summary_columns].mean()
    cluster_summary.insert(0, "province_count", features.groupby("cluster_id").size())
    cluster_names = _name_clusters(cluster_summary)
    features["cluster_name"] = features["cluster_id"].map(cluster_names)

    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = models_dir / "migration_cluster_pipeline.joblib"
    pca_path = models_dir / "pca_model.joblib"
    metadata_path = models_dir / "cluster_metadata.json"
    profiles_path = processed_dir / "city_profiles.csv"

    joblib.dump(cluster_pipeline, pipeline_path)
    joblib.dump(pca_pipeline, pca_path)
    features.to_csv(profiles_path, index=False)

    pca_model = pca_pipeline.named_steps["pca"]
    metadata = {
        "model_version": MODEL_VERSION,
        "training_period": {"start": START_YEAR, "end": END_YEAR},
        "training_observations": len(features),
        "selected_k": selected_k,
        "selection_rule": (
            "k=2–8 arasında; en az 3 il/küme, en büyük küme en fazla %75; "
            "Silhouette, Davies–Bouldin, Calinski–Harabasz ve seed ARI sıra toplamı"
        ),
        "features": MODEL_FEATURES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "evaluation_metrics": {
            "inertia": float(selected_metrics["inertia"]),
            "silhouette": float(selected_metrics["silhouette"]),
            "davies_bouldin": float(selected_metrics["davies_bouldin"]),
            "calinski_harabasz": float(selected_metrics["calinski_harabasz"]),
            "stability_ari": float(selected_metrics["stability_ari"]),
        },
        "k_comparison": evaluation.round(6).to_dict(orient="records"),
        "cluster_names": {str(key): value for key, value in cluster_names.items()},
        "cluster_profiles": {
            str(cluster_id): {
                "name": cluster_names[int(cluster_id)],
                **{
                    key: float(value)
                    for key, value in row.items()
                    if key != "province_count"
                },
                "province_count": int(row["province_count"]),
                "provinces": sorted(
                    features.loc[
                        features["cluster_id"].eq(cluster_id), "province"
                    ].tolist()
                ),
            }
            for cluster_id, row in cluster_summary.iterrows()
        },
        "pca_explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
        "hdbscan_comparison": hdbscan_metrics,
        "software": {"scikit_learn": sklearn.__version__},
        "interpretation_note": (
            "Model gelecek tahmini yapmaz; illeri 2008–2025 göç davranışlarına "
            "göre benzer profillere ayırır."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reloaded = joblib.load(pipeline_path)
    reloaded_labels = reloaded.predict(features[MODEL_FEATURES])
    if not np.array_equal(labels, reloaded_labels):
        raise AssertionError("Kaydedilen modelin reload tahminleri eğitim sonucu ile uyuşmuyor.")

    return {
        "features": features,
        "evaluation": evaluation,
        "metadata": metadata,
        "pipeline_path": pipeline_path,
        "pca_path": pca_path,
        "metadata_path": metadata_path,
        "reload_consistent": True,
    }


def main() -> None:
    result = train_and_save_models()
    metadata = result["metadata"]
    metrics = metadata["evaluation_metrics"]
    print(
        "Model kaydedildi: "
        f"k={metadata['selected_k']}, silhouette={metrics['silhouette']:.3f}, "
        f"Davies–Bouldin={metrics['davies_bouldin']:.3f}, "
        f"stability ARI={metrics['stability_ari']:.3f}."
    )


if __name__ == "__main__":
    main()
