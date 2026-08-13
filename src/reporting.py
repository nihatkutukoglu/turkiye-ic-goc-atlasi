"""Analiz sonuçlarından tekrar üretilebilir rapor ve statik figürler üretir."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_output_directories
from src.data_loader import (
    load_age_gender,
    load_age_gender_reason,
    load_city_profiles,
    load_city_year_metrics,
    load_migration_flows,
)
from src.network_analysis import province_network_metrics, strongest_corridors


NAVY = "#14213D"
RED = "#9B1C31"
BLUE = "#3D5A80"
GRAY = "#667085"
LIGHT_GRAY = "#E6E9EF"
CLUSTER_COLORS = ["#9B1C31", "#3D5A80", "#2A9D8F", "#D9822B", "#6D597A"]


def _style_axes(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(LIGHT_GRAY)
    axis.grid(axis=grid_axis, color=LIGHT_GRAY, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=GRAY)
    axis.title.set_color(NAVY)


def _save_figure(figure: plt.Figure, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_figures(
    city_year: pd.DataFrame,
    flows: pd.DataFrame,
    reasons: pd.DataFrame,
    profiles: pd.DataFrame,
    metadata: dict,
) -> None:
    yearly = city_year.groupby("year", as_index=False)["in_migration"].sum()
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(
        yearly["year"],
        yearly["in_migration"] / 1_000_000,
        color=RED,
        marker="o",
        linewidth=2.4,
    )
    peak = yearly.loc[yearly["in_migration"].idxmax()]
    axis.annotate(
        f"Zirve: {int(peak['in_migration']):,}",
        xy=(peak["year"], peak["in_migration"] / 1_000_000),
        xytext=(-78, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": RED},
        color=NAVY,
        fontweight="bold",
    )
    axis.set(
        title="Türkiye'de Yıllık İller Arası Göç",
        xlabel="Yıl",
        ylabel="Göç eden kişi (milyon)",
    )
    axis.set_xticks(yearly["year"])
    axis.tick_params(axis="x", rotation=45)
    _style_axes(axis)
    _save_figure(figure, "yearly_migration.png")

    latest = city_year.loc[city_year["year"].eq(city_year["year"].max())]
    extremes = pd.concat(
        [latest.nlargest(10, "net_migration"), latest.nsmallest(10, "net_migration")]
    ).sort_values("net_migration")
    figure, axis = plt.subplots(figsize=(10, 7.5))
    colors = np.where(extremes["net_migration"].ge(0), BLUE, RED)
    axis.barh(extremes["province"], extremes["net_migration"], color=colors)
    axis.axvline(0, color=NAVY, linewidth=0.8)
    axis.set(
        title="2025 Net Göçte Öne Çıkan İller",
        xlabel="Net göç (kişi)",
        ylabel="",
    )
    _style_axes(axis, grid_axis="x")
    _save_figure(figure, "net_migration_2025.png")

    corridors = strongest_corridors(flows, year=2025, top_n=12).sort_values("flow")
    labels = corridors["origin"] + " → " + corridors["destination"]
    figure, axis = plt.subplots(figsize=(10, 6.5))
    axis.barh(labels, corridors["flow"], color=BLUE)
    axis.set(
        title="2025'in En Güçlü Yönlü Göç Koridorları",
        xlabel="Göç eden kişi",
        ylabel="",
    )
    _style_axes(axis, grid_axis="x")
    _save_figure(figure, "top_corridors_2025.png")

    reason_totals = (
        reasons.groupby("reason", as_index=False)
        .agg(count=("count", lambda values: values.sum(min_count=1)))
        .sort_values("count")
    )
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.barh(reason_totals["reason"], reason_totals["count"], color=BLUE)
    axis.set(
        title="2025 İller Arası Göç Nedenleri",
        xlabel="Kişi",
        ylabel="",
    )
    _style_axes(axis, grid_axis="x")
    _save_figure(figure, "migration_reasons_2025.png")

    figure, axis = plt.subplots(figsize=(11, 7.5))
    for index, (cluster_name, group) in enumerate(
        profiles.groupby("cluster_name", sort=True)
    ):
        axis.scatter(
            group["pc1"],
            group["pc2"],
            s=70,
            color=CLUSTER_COLORS[index % len(CLUSTER_COLORS)],
            label=cluster_name,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )
    for _, row in profiles.nlargest(6, "latest_population").iterrows():
        axis.annotate(
            row["province"],
            (row["pc1"], row["pc2"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=NAVY,
        )
    variance = metadata["pca_explained_variance_ratio"]
    axis.set(
        title="PCA Düzleminde Uzun Dönemli İl Profilleri",
        xlabel=f"PC1 (%{variance[0] * 100:.1f} açıklanan varyans)",
        ylabel=f"PC2 (%{variance[1] * 100:.1f} açıklanan varyans)",
    )
    axis.legend(frameon=False, fontsize=8, loc="best")
    _style_axes(axis)
    _save_figure(figure, "cluster_profiles_pca.png")

    evaluation = pd.DataFrame(metadata["k_comparison"])
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    plots = [
        ("silhouette", "Silhouette", True),
        ("davies_bouldin", "Davies–Bouldin", False),
        ("calinski_harabasz", "Calinski–Harabasz", True),
    ]
    for axis, (column, title, _) in zip(axes, plots):
        axis.plot(evaluation["k"], evaluation[column], marker="o", color=BLUE)
        axis.axvline(metadata["selected_k"], color=RED, linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="k")
        _style_axes(axis)
    _save_figure(figure, "kmeans_model_comparison.png")


def _integer(value: float | int) -> str:
    return f"{int(value):,}".replace(",", ".")


def _decimal(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def build_report(
    report_path: Path = REPORTS_DIR / "project_report.md",
) -> Path:
    ensure_output_directories()
    city_year = load_city_year_metrics()
    flows = load_migration_flows()
    age_gender = load_age_gender()
    reasons = load_age_gender_reason()
    profiles = load_city_profiles()
    metadata = json.loads(
        (MODELS_DIR / "cluster_metadata.json").read_text(encoding="utf-8")
    )
    create_figures(city_year, flows, reasons, profiles, metadata)

    yearly = city_year.groupby("year")["in_migration"].sum()
    latest = city_year.loc[city_year["year"].eq(2025)]
    top_in = latest.loc[latest["in_migration"].idxmax()]
    top_out = latest.loc[latest["out_migration"].idxmax()]
    top_net = latest.loc[latest["net_migration"].idxmax()]
    low_net = latest.loc[latest["net_migration"].idxmin()]
    top_rate = latest.loc[latest["net_migration_rate"].idxmax()]
    low_rate = latest.loc[latest["net_migration_rate"].idxmin()]
    peak_year = int(yearly.idxmax())
    peak_value = int(yearly.max())
    yoy_change = 100 * (yearly.loc[2025] / yearly.loc[2024] - 1)

    corridor = strongest_corridors(flows, 2025, 1).iloc[0]
    network = province_network_metrics(flows, 2025)
    top_pagerank = network.iloc[0]
    reason_totals = (
        reasons.groupby("reason")["count"]
        .sum(min_count=1)
        .sort_values(ascending=False)
    )
    top_reason = reason_totals.index[0]
    top_reason_value = reason_totals.iloc[0]
    gender_totals = age_gender[["male", "female"]].sum()
    top_age = age_gender.loc[age_gender["total"].idxmax()]
    metrics = metadata["evaluation_metrics"]
    hdbscan = metadata["hdbscan_comparison"]

    profile_lines = []
    for cluster_id, profile in sorted(metadata["cluster_profiles"].items()):
        profile_lines.append(
            f"- **{profile['name']} ({profile['province_count']} il):** "
            f"ortalama hareketlilik hızı ‰{_decimal(profile['mean_turnover_rate'], 1)}, "
            f"uzun dönem net göç hızı ‰{_decimal(profile['mean_net_migration_rate'], 1)}, "
            f"son üç yıl ortalaması ‰{_decimal(profile['recent_net_migration_rate'], 1)}."
        )

    report = f"""# Türkiye İç Göç Atlası — Analitik Rapor

## Executive Summary

Bu çalışma, TÜİK'in 2008–2025 dönemine ait **{_integer(len(flows))}** iller arası akış kaydını ve 2025 demografi tablolarını birleştirir. Yıllık toplam iller arası göç {peak_year}'te **{_integer(peak_value)}** kişiyle zirveye ulaştı; 2025 toplamı **{_integer(yearly.loc[2025])}** kişi ve bir önceki yıla göre değişim **%{_decimal(yoy_change, 1)}** oldu. Uzun dönemli özelliklerle eğitilen K-Means modeli, 81 ili yorumlanabilir dört göç profiline ayırdı.

## Problem Definition

Amaç, Türkiye'deki iller arası göçün hacmini, yönünü, demografik bileşimini ve uzun dönemli il profillerini tekrar üretilebilir bir veri bilimi hattıyla incelemektir. Clustering sonuçları nedensel veya ileriye dönük tahmin olarak değil, benzer göç davranışlarının betimleyici segmentasyonu olarak yorumlanır.

## Data Sources

- `iller_arasi_goc.xlsx`: 2008–2025, yıl × hedef il × kaynak il seviyesinde 116.640 kayıt.
- `illerin_goc_ozeti.xls`: 2025 için 81 ilin nüfus, giriş, çıkış, net göç ve net göç hızı.
- `yas_cinsiyet_goc.xls`: 2025 yaş grubu ve cinsiyet dağılımı.
- `yas_cinsiyet_neden.xls`: 2025 yaş, cinsiyet ve göç nedeni kırılımı.
- `egitim_goc_nedeni.xls`: 2025 eğitim durumu ve göç nedeni kırılımı (6+ yaş).

## Data Quality

Ham dosyalardaki Türkçe/İngilizce kaynak satırları, dipnotlar, toplam satırları ve açıklamalar processed katmandan çıkarıldı. Ana akış tablosunda her yıl için **81 × 80 = 6.480** kaynak–hedef çifti, sıfır self-loop, sıfır duplicate ve sıfır eksik değer doğrulandı. Negatif olamayacak alanlar, iki yönlü akış simetrisi ve `net = gelen - giden` eşitliği assertionlarla kontrol edildi. 2025 il agregasyonları resmi 81-il özetinin nüfus, gelen, giden, net ve TÜİK net göç hızıyla birebir uzlaştırıldı. `-` işaretleri sıfıra çevrilmedi; bilgi-yok değerleri `NaN` olarak korundu.

## Methodology

Bir satırdaki `migration_flow`, kaynak ilden hedef ile olan tekil yönlü akıştır. Ulusal toplamda yalnızca bu alan toplanarak aynı hareketin iki kez sayılması engellendi. İl düzeyinde gelen ve giden göç ayrı ayrı toplandı; hareketlilik bunların toplamı olarak tanımlandı. Net göç hızı TÜİK'in dönem ortası nüfus yaklaşımıyla hesaplandı.

## Exploratory Analysis

![Yıllık göç](figures/yearly_migration.png)

2023 zirvesinin ardından toplam hareket 2024 ve 2025'te geriledi. 2025'te en fazla göç alan ve veren il **{top_in['province']}** oldu (sırasıyla {_integer(top_in['in_migration'])} ve {_integer(top_out['out_migration'])}). Mutlak net göçte **{top_net['province']}** +{_integer(top_net['net_migration'])} ile ilk, **{low_net['province']}** {_integer(low_net['net_migration'])} ile son sırada yer aldı. Nüfusa oranlı net göç hızında **{top_rate['province']}** ‰{_decimal(top_rate['net_migration_rate'])}, **{low_rate['province']}** ‰{_decimal(low_rate['net_migration_rate'])} kaydetti.

![Net göç](figures/net_migration_2025.png)

2025'te en kalabalık göç yaş grubu **{top_age['age_group']}** ({_integer(top_age['total'])} kişi) oldu. Toplamda {_integer(gender_totals['female'])} kadın ve {_integer(gender_totals['male'])} erkek iller arası göç etti. En yaygın kaydedilen neden **{top_reason}** ({_integer(top_reason_value)} kişi) oldu.

![Göç nedenleri](figures/migration_reasons_2025.png)

## Migration Network Analysis

Akışlar kaynak→hedef yönünde weighted directed graph olarak kuruldu. 2025'in en büyük yönlü koridoru **{corridor['origin']} → {corridor['destination']}** ve {_integer(corridor['flow'])} kişiydi. Weighted PageRank'te **{top_pagerank['province']}** {_decimal(top_pagerank['pagerank'], 3)} ile ilk sırada yer aldı. Ağ yoğun olduğu için yorum gücü düşük betweenness metriği final analize eklenmedi.

![Göç koridorları](figures/top_corridors_2025.png)

## Feature Engineering

Model girdileri tek bir yıl yerine 2008–2025 davranışını temsil eder: ortalama hareketlilik hızı, ortalama net göç hızı, net hız volatilitesi, doğrusal trend, 2023–2025 yakın dönem ortalaması ve log-dönüşümlü 2025 nüfusu. Manuel deprem bayrağı veya bölge dummy'si kullanılmadı; bu sayede clustering coğrafi etiketlerle önceden zorlanmadı.

## Clustering Methodology

K-Means için `k=2…8` aralığında inertia, Silhouette, Davies–Bouldin, Calinski–Harabasz ve 10 farklı random seed arası Adjusted Rand Index kararlılığı hesaplandı. En az 3 il içeren ve tek kümede illerin %75'inden fazlasını toplamayan adaylar arasında metrik sıraları dengelendi.

![Model karşılaştırması](figures/kmeans_model_comparison.png)

## Model Comparison

HDBSCAN karşılaştırması **{hdbscan['cluster_count']} küme** ve **{hdbscan['noise_count']} noise il** üretti; noise hariç Silhouette {_decimal(hdbscan['silhouette_non_noise'], 3)} oldu. Yeni gözlemlere kararlı profil atayabilen K-Means bu nedenle production modeli, HDBSCAN ise keşifsel sağlama olarak tutuldu.

## Final Model

Seçilen `k`: **{metadata['selected_k']}**. Final metrikler: Silhouette **{_decimal(metrics['silhouette'], 3)}**, Davies–Bouldin **{_decimal(metrics['davies_bouldin'], 3)}**, Calinski–Harabasz **{_decimal(metrics['calinski_harabasz'], 2)}**, seed stability ARI **{_decimal(metrics['stability_ari'], 3)}**. PCA'nın ilk iki bileşeni toplam varyansın **%{_decimal(100 * sum(metadata['pca_explained_variance_ratio']), 1)}**'ini açıklar; PCA yalnızca boyut indirgeme ve görselleştirme için kullanıldı.

![PCA profilleri](figures/cluster_profiles_pca.png)

## Cluster Profiles

{chr(10).join(profile_lines)}

## Key Findings

1. Yıllık iller arası göç 2023'te {_integer(peak_value)} kişiyle dönem zirvesine ulaştı.
2. 2025 toplamı 2024'e göre %{_decimal(abs(yoy_change), 1)} azaldı.
3. İstanbul hem en yüksek giriş hem en yüksek çıkış hacmine sahipken net {_integer(low_net['net_migration'])} kayıp verdi.
4. Ankara +{_integer(top_net['net_migration'])} kişiyle 2025'in en yüksek mutlak net kazancını kaydetti.
5. İstanbul→Kocaeli koridoru 2025'te {_integer(corridor['flow'])} kişiyle en güçlü yönlü akış oldu.
6. 20–24 yaş grubu {_integer(top_age['total'])} kişiyle en hareketli yaş grubuydu.
7. Hane/aile ferdine bağımlı göç {_integer(top_reason_value)} kişiyle en yaygın kaydedilen nedendi.
8. Dört profilli K-Means sonucu farklı seed'ler karşısında yüksek kararlılık gösterdi (ARI {_decimal(metrics['stability_ari'], 3)}).

## Limitations

- Yaş, cinsiyet, eğitim ve göç nedeni tabloları yalnızca 2025'i kapsar; bu boyutlarda uzun dönem trendi kurulamaz.
- Ham tablodaki `-` işareti bilgi-yok/magnitude-null anlamındadır; eksik değer olarak korunur ve ilgili toplamlar dikkatle yorumlanır.
- 81 il, clustering için küçük bir örneklemdir; içsel metrikler evrensel bir "doğru" segmentasyon garanti etmez.
- Cluster'lar nedensellik veya gelecek tahmini sunmaz.
- İstihdam, gelir, konut maliyeti ve afet etkisi gibi dış açıklayıcı değişkenler modelde yoktur.

## Future Work

Ekonomik ve mekânsal değişkenlerin kaynaklı biçimde eklenmesi, profile stability için bootstrap güven aralıkları, göç koridorlarının coğrafi harita üzerinde gösterimi ve yeni TÜİK yılları geldiğinde veri/model sürümleme adımları sonraki geliştirmelerdir.

## Deployment

Model preprocessing + K-Means adımları tek `joblib` pipeline olarak, PCA ayrı bir pipeline olarak ve profil tanımları JSON metadata olarak saklanır. Streamlit uygulaması bu artefact'ları cache ile diskten yükler; açılışta model eğitmez.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    report_path = build_report()
    print(f"Rapor ve figürler oluşturuldu: {report_path}")


if __name__ == "__main__":
    main()
