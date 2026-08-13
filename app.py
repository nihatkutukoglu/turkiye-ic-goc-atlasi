"""Türkiye İç Göç Atlası Streamlit veri ürünü."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODELS_DIR
from src.data_loader import (
    load_age_gender,
    load_age_gender_reason,
    load_city_profiles,
    load_city_year_metrics,
    load_education_reason,
    load_migration_flows,
)
from src.feature_engineering import MODEL_FEATURES
from src.network_analysis import strongest_corridors


APP_DIR = Path(__file__).resolve().parent
TUIK_LOGO_PATH = APP_DIR / "assets" / "tuik_logo.png"
PRODUCT_VERSION = "1.0"
DATA_VERSION = "2025"


st.set_page_config(
    page_title="Türkiye İç Göç Atlası",
    page_icon="🇹🇷",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
<style>
    :root { --navy:#14213D; --burgundy:#9B1C31; --muted:#667085; --line:#E4E7EC; }
    .stApp { background:#F7F8FA; color:var(--navy); }
    [data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color:var(--navy); }
    /* Streamlit'in sabit üst araç çubuğu ilk satırı örtmesin. */
    .block-container { max-width:1440px; padding-top:4.5rem; padding-bottom:3rem; }
    h1, h2, h3 { color:var(--navy); letter-spacing:-0.02em; }
    h1 { font-weight:760; }
    .atlas-subtitle { color:var(--muted); font-size:1.05rem; margin-top:-0.7rem; margin-bottom:1.8rem; }
    .product-kicker { color:var(--burgundy); font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; font-weight:800; margin-bottom:.25rem; }
    .product-meta { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:-1rem; margin-bottom:1.55rem; }
    .product-badge { background:#FFFFFF; border:1px solid var(--line); border-radius:999px; color:#475467; font-size:.75rem; font-weight:650; padding:.3rem .65rem; }
    .kpi-card { background:#FFFFFF; border:1px solid var(--line); border-radius:12px; padding:18px 20px; min-height:112px; box-shadow:0 1px 2px rgba(16,24,40,.035); }
    .kpi-label { color:var(--muted); font-size:.78rem; letter-spacing:.04em; text-transform:uppercase; font-weight:650; }
    .kpi-value { color:var(--navy); font-size:1.55rem; line-height:1.2; font-weight:760; margin-top:8px; }
    .kpi-note { color:var(--muted); font-size:.78rem; margin-top:5px; }
    .profile-box { background:#FFFFFF; border-left:4px solid var(--burgundy); border-radius:8px; padding:16px 18px; margin:.5rem 0 1rem; }
    .method-box { background:#FFFFFF; border:1px solid var(--line); border-radius:10px; padding:14px 16px; color:#475467; }
    .guide-box { background:#EEF3F8; border-left:4px solid #3D5A80; border-radius:8px; padding:15px 18px; margin:.25rem 0 1.2rem; color:#344054; line-height:1.55; }
    .insight-box { background:#FFF8F0; border-left:4px solid #D9822B; border-radius:8px; padding:15px 18px; margin:.8rem 0 1.4rem; color:#344054; line-height:1.55; }
    .guide-title, .insight-title { color:var(--navy); font-weight:750; margin-bottom:5px; }
    .source-title { color:var(--muted); font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; font-weight:750; margin-bottom:.25rem; }
    .source-note { color:var(--muted); font-size:.76rem; line-height:1.45; margin-top:.35rem; }
    .product-footer { color:var(--muted); font-size:.76rem; line-height:1.6; padding:.2rem 0 1rem; }
    div[data-testid="stPlotlyChart"] { background:#FFFFFF; border:1px solid var(--line); border-radius:12px; padding:8px; }
    .stButton > button { background:var(--burgundy); color:white; border:0; border-radius:8px; font-weight:650; }
    .stButton > button:hover { background:#7D1728; color:white; border:0; }
    .stDownloadButton > button { border-color:#D0D5DD; color:var(--navy); border-radius:8px; font-weight:650; }
    @media (max-width: 700px) {
        .block-container { padding:4rem .9rem 2rem; }
        .kpi-card { min-height:96px; padding:14px; }
        .kpi-value { font-size:1.25rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "flows": load_migration_flows(),
        "city_year": load_city_year_metrics(),
        "age_gender": load_age_gender(),
        "age_reason": load_age_gender_reason(),
        "education_reason": load_education_reason(),
        "profiles": load_city_profiles(),
    }


@st.cache_resource(show_spinner=False)
def load_model_assets() -> tuple[object, dict]:
    pipeline = joblib.load(MODELS_DIR / "migration_cluster_pipeline.joblib")
    metadata = json.loads(
        (MODELS_DIR / "cluster_metadata.json").read_text(encoding="utf-8")
    )
    return pipeline, metadata


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}".replace(",", ".")


def fmt_decimal(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def guide(text: str, title: str = "Bu bölüm nasıl kullanılır?") -> None:
    st.markdown(
        f'<div class="guide-box"><div class="guide-title">{title}</div>{text}</div>',
        unsafe_allow_html=True,
    )


def insight(text: str, title: str = "Seçiminize göre kısa yorum") -> None:
    st.markdown(
        f'<div class="insight-box"><div class="insight-title">{title}</div>{text}</div>',
        unsafe_allow_html=True,
    )


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Excel'in Türkçe karakterleri doğru açması için BOM içeren CSV üretir."""

    return frame.to_csv(index=False).encode("utf-8-sig")


def chart_layout(figure: go.Figure, height: int = 460) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 24, "r": 24, "t": 62, "b": 32},
        font={"family": "Arial, sans-serif", "color": "#14213D"},
        title={"font": {"size": 18}},
        legend={"title": None},
    )
    return figure


def page_header() -> None:
    st.markdown(
        '<div class="product-kicker">Türkiye\'nin iç göç veri ürünü</div>',
        unsafe_allow_html=True,
    )
    st.title("Türkiye İç Göç Atlası")
    st.markdown(
        '<div class="atlas-subtitle">2008–2025 TÜİK verileriyle Türkiye\'nin iç göç dinamiklerini keşfedin.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="product-meta"><span class="product-badge">81 il</span>'
        f'<span class="product-badge">18 yıl</span><span class="product-badge">Veri sürümü {DATA_VERSION}</span>'
        f'<span class="product-badge">Ürün v{PRODUCT_VERSION}</span></div>',
        unsafe_allow_html=True,
    )


def page_footer() -> None:
    st.divider()
    st.markdown(
        f'<div class="product-footer"><b>Türkiye İç Göç Atlası · v{PRODUCT_VERSION}</b><br>'
        "Kaynak: TÜİK İç Göç İstatistikleri. Bağımsız analiz ürünüdür; "
        "TÜİK'in resmî uygulaması değildir.</div>",
        unsafe_allow_html=True,
    )


def overview_page(data: dict[str, pd.DataFrame]) -> None:
    guide(
        "Yıl sürgüsünü değiştirerek Türkiye genelindeki göç tablosunu inceleyin. "
        "Üstteki kartlar seçilen yılın özetini, alttaki grafikler zaman içindeki değişimi, "
        "net kazanan/kaybeden illeri ve en yoğun il çiftlerini gösterir."
    )
    city_year = data["city_year"]
    years = sorted(city_year["year"].unique(), reverse=True)
    selected_year = st.select_slider(
        "Analiz yılı",
        options=sorted(years),
        value=max(years),
    )
    selected = city_year.loc[city_year["year"].eq(selected_year)]
    st.download_button(
        "Seçili yılın il özetini indir",
        data=csv_bytes(selected),
        file_name=f"turkiye_ic_goc_il_ozeti_{selected_year}.csv",
        mime="text/csv",
        key="download_overview",
    )
    top_in = selected.loc[selected["in_migration"].idxmax()]
    top_out = selected.loc[selected["out_migration"].idxmax()]
    top_net = selected.loc[selected["net_migration"].idxmax()]
    low_net = selected.loc[selected["net_migration"].idxmin()]

    columns = st.columns(3)
    with columns[0]:
        kpi("Analiz edilen dönem", f"{min(years)}–{max(years)}", f"{len(years)} yıl")
    with columns[1]:
        kpi("Toplam göç", fmt_int(selected["in_migration"].sum()), str(selected_year))
    with columns[2]:
        kpi("En fazla göç alan", top_in["province"], fmt_int(top_in["in_migration"]))
    columns = st.columns(3)
    with columns[0]:
        kpi("En fazla göç veren", top_out["province"], fmt_int(top_out["out_migration"]))
    with columns[1]:
        kpi("En yüksek net göç", top_net["province"], f"+{fmt_int(top_net['net_migration'])}")
    with columns[2]:
        kpi("En düşük net göç", low_net["province"], fmt_int(low_net["net_migration"]))

    if selected_year > min(years):
        previous_total = city_year.loc[
            city_year["year"].eq(selected_year - 1), "in_migration"
        ].sum()
        current_total = selected["in_migration"].sum()
        annual_change = 100 * (current_total / previous_total - 1)
        direction = "arttı" if annual_change >= 0 else "azaldı"
        insight(
            f"{selected_year} yılında toplam <b>{fmt_int(current_total)}</b> kişi iller arasında göç etti. "
            f"Bu sayı bir önceki yıla göre <b>%{fmt_decimal(abs(annual_change))} {direction}</b>. "
            f"En yüksek net kazanç {top_net['province']} ilinde, en yüksek net kayıp "
            f"{low_net['province']} ilinde görüldü."
        )

    st.subheader("Uzun dönemli hareketlilik")
    yearly = city_year.groupby("year", as_index=False)["in_migration"].sum()
    figure = px.line(
        yearly,
        x="year",
        y="in_migration",
        markers=True,
        labels={"year": "Yıl", "in_migration": "Göç eden kişi"},
        title="Yıllara Göre Toplam İller Arası Göç",
    )
    figure.update_traces(line={"color": "#9B1C31", "width": 3}, marker={"size": 7})
    st.plotly_chart(chart_layout(figure), use_container_width=True)

    left, right = st.columns([1.25, 1])
    with left:
        extremes = pd.concat(
            [selected.nlargest(8, "net_migration"), selected.nsmallest(8, "net_migration")]
        ).sort_values("net_migration")
        extremes["yön"] = np.where(extremes["net_migration"] >= 0, "Net kazanç", "Net kayıp")
        figure = px.bar(
            extremes,
            x="net_migration",
            y="province",
            color="yön",
            orientation="h",
            color_discrete_map={"Net kazanç": "#3D5A80", "Net kayıp": "#9B1C31"},
            labels={"net_migration": "Net göç", "province": ""},
            title=f"Net Göçte Öne Çıkan İller · {selected_year}",
        )
        st.plotly_chart(chart_layout(figure, 540), use_container_width=True)
    with right:
        corridors = strongest_corridors(data["flows"], selected_year, 10)
        corridors["koridor"] = corridors["origin"] + " → " + corridors["destination"]
        figure = px.bar(
            corridors.sort_values("flow"),
            x="flow",
            y="koridor",
            orientation="h",
            color_discrete_sequence=["#3D5A80"],
            labels={"flow": "Kişi", "koridor": ""},
            title=f"En Güçlü Koridorlar · {selected_year}",
        )
        st.plotly_chart(chart_layout(figure, 540), use_container_width=True)

    st.markdown(
        '<div class="method-box"><b>Sayım notu:</b> Ulusal toplam, kaynak→hedef yönündeki tekil '
        '<code>migration_flow</code> alanından hesaplanır. Ters akış ayrı bir yolculuk olduğu için '
        'aynı hareket iki kez sayılmaz.</div>',
        unsafe_allow_html=True,
    )


def province_page(data: dict[str, pd.DataFrame], metadata: dict) -> None:
    guide(
        "Önce bir il ve yıl seçin. <b>Aldığı göç</b> başka illerden bu ile gelenleri, "
        "<b>verdiği göç</b> bu ilden başka illere gidenleri gösterir. "
        "<b>Net göç = aldığı göç − verdiği göç</b>; sonuç pozitifse il göç yoluyla nüfus kazanmış, "
        "negatifse nüfus kaybetmiştir."
    )
    city_year = data["city_year"]
    provinces = sorted(city_year["province"].unique())
    controls = st.columns([1, 1, 2])
    with controls[0]:
        province = st.selectbox("İl", provinces, index=provinces.index("İSTANBUL"))
    with controls[1]:
        year = st.selectbox("Yıl", sorted(city_year["year"].unique(), reverse=True))

    row = city_year.loc[
        city_year["province"].eq(province) & city_year["year"].eq(year)
    ].iloc[0]
    profile = data["profiles"].loc[data["profiles"]["province"].eq(province)].iloc[0]
    province_history = city_year.loc[city_year["province"].eq(province)].copy()
    with controls[2]:
        st.caption("Seçili ilin 2008–2025 geçmişi")
        st.download_button(
            "İl verisini indir",
            data=csv_bytes(province_history),
            file_name=f"{province.lower()}_goc_profili_2008_2025.csv",
            mime="text/csv",
            key="download_province",
        )

    columns = st.columns(5)
    values = [
        ("Aldığı göç", fmt_int(row["in_migration"]), str(year)),
        ("Verdiği göç", fmt_int(row["out_migration"]), str(year)),
        ("Net göç", fmt_int(row["net_migration"]), str(year)),
        ("Net göç hızı", f"{fmt_decimal(row['net_migration_rate'])}‰", "binde · TÜİK formülü"),
        (
            "Göç hareketliliği",
            f"%{fmt_decimal(row['turnover_rate'] / 10)}",
            "(gelen + giden) / nüfus",
        ),
    ]
    for column, (label, value, note) in zip(columns, values):
        with column:
            kpi(label, value, note)

    st.caption(
        f"Göç hareketliliği: {fmt_int(row['in_migration'] + row['out_migration'])} gelen-giden "
        f"hareketinin {fmt_int(row['population'])} kişilik il nüfusuna oranıdır. Bu değer, nüfusun "
        "tam olarak bu kadarlık bölümünün taşındığı anlamına gelmez; gelen ve giden hareketleri birlikte sayar."
    )

    balance_word = "net göç kazandı" if row["net_migration"] >= 0 else "net göç kaybetti"
    previous = city_year.loc[
        city_year["province"].eq(province) & city_year["year"].eq(year - 1)
    ]
    comparison_text = ""
    if not previous.empty:
        difference = row["net_migration"] - previous.iloc[0]["net_migration"]
        movement = "iyileşti" if difference >= 0 else "geriledi"
        comparison_text = (
            f" Net göç dengesi {year - 1}'e göre "
            f"<b>{fmt_int(abs(difference))} kişi {movement}</b>."
        )
    insight(
        f"{province}, {year} yılında <b>{fmt_int(row['in_migration'])}</b> kişi aldı, "
        f"<b>{fmt_int(row['out_migration'])}</b> kişi verdi ve sonuçta "
        f"<b>{fmt_int(abs(row['net_migration']))} kişi {balance_word}</b>.{comparison_text}"
    )

    description = metadata["cluster_profiles"][str(int(profile["cluster_id"]))]
    st.markdown(
        f"""<div class="profile-box"><b>Uzun dönemli göç profili</b><br>
        <span style="font-size:1.15rem;color:#9B1C31;font-weight:700;">{profile['cluster_name']}</span><br>
        <span style="color:#667085;">2008–2025 ortalama net göç hızı ‰{fmt_decimal(description['mean_net_migration_rate'])};
        son üç yıl ortalaması ‰{fmt_decimal(description['recent_net_migration_rate'])}.</span></div>""",
        unsafe_allow_html=True,
    )

    history = province_history
    trend = history.melt(
        id_vars="year",
        value_vars=["in_migration", "out_migration", "net_migration"],
        var_name="metric",
        value_name="value",
    )
    labels = {"in_migration": "Aldığı göç", "out_migration": "Verdiği göç", "net_migration": "Net göç"}
    trend["metric"] = trend["metric"].map(labels)
    figure = px.line(
        trend,
        x="year",
        y="value",
        color="metric",
        markers=True,
        color_discrete_sequence=["#3D5A80", "#D9822B", "#9B1C31"],
        labels={"year": "Yıl", "value": "Kişi", "metric": ""},
        title=f"{province} · 2008–2025 Göç Trendi",
    )
    st.plotly_chart(chart_layout(figure), use_container_width=True)
    st.caption(
        "Grafiği okuma: Mavi çizgi gelenleri, turuncu çizgi gidenleri, bordo çizgi ikisinin farkını gösterir. "
        "Bordo çizginin sıfırın altında olması net göç kaybına işaret eder."
    )

    flows = data["flows"].loc[data["flows"]["year"].eq(year)]
    incoming = flows.loc[flows["destination_province"].eq(province)].nlargest(8, "migration_flow").copy()
    incoming["connection"] = incoming["origin_province"] + " → " + province
    outgoing = flows.loc[flows["origin_province"].eq(province)].nlargest(8, "migration_flow").copy()
    outgoing["connection"] = province + " → " + outgoing["destination_province"]
    connections = pd.concat([incoming, outgoing])
    connections["type"] = ["Gelen"] * len(incoming) + ["Giden"] * len(outgoing)
    guide(
        f"<b>Mavi çubuklar</b>, {province}'a hangi illerden kaç kişinin geldiğini; "
        f"<b>bordo çubuklar</b>, {province}'dan hangi illere kaç kişinin gittiğini gösterir. "
        "Okun solundaki il kaynaktır, sağındaki il hedeftir. Çubuk uzadıkça o yöndeki göç sayısı artar.",
        title="Bu grafik nasıl okunur?",
    )
    figure = px.bar(
        connections.sort_values("migration_flow"),
        x="migration_flow",
        y="connection",
        color="type",
        orientation="h",
        color_discrete_map={"Gelen": "#3D5A80", "Giden": "#9B1C31"},
        labels={"migration_flow": "Kişi", "connection": "", "type": ""},
        title=f"{province}'un En Yoğun Göç Bağlantıları · {year}",
    )
    st.plotly_chart(chart_layout(figure, 580), use_container_width=True)

    top_incoming = incoming.iloc[0]
    top_outgoing = outgoing.iloc[0]
    incoming_share = 100 * top_incoming["migration_flow"] / row["in_migration"]
    outgoing_share = 100 * top_outgoing["migration_flow"] / row["out_migration"]
    insight(
        f"En büyük giriş bağlantısı <b>{top_incoming['origin_province']} → {province}</b> "
        f"({fmt_int(top_incoming['migration_flow'])} kişi; toplam gelenlerin %{fmt_decimal(incoming_share)}'i). "
        f"En büyük çıkış bağlantısı <b>{province} → {top_outgoing['destination_province']}</b> "
        f"({fmt_int(top_outgoing['migration_flow'])} kişi; toplam gidenlerin %{fmt_decimal(outgoing_share)}'i)."
    )


def flows_page(data: dict[str, pd.DataFrame]) -> None:
    guide(
        "Bu sayfa iller arasındaki hareketleri yönleriyle gösterir. <b>Kaynak il</b> insanların "
        "ayrıldığı, <b>hedef il</b> yerleştiği ildir. Varsayılan grafik en yoğun 15 yönü büyükten küçüğe "
        "sıralar. Daha az sonuç görmek için minimum kişi sayısını yükseltebilirsiniz."
    )
    flows = data["flows"]
    provinces = sorted(flows["origin_province"].unique())
    columns = st.columns(4)
    with columns[0]:
        year = st.selectbox("Yıl", sorted(flows["year"].unique(), reverse=True), key="flow_year")
    with columns[1]:
        origin = st.selectbox("Kaynak il", ["Tümü", *provinces])
    with columns[2]:
        destination = st.selectbox("Hedef il", ["Tümü", *provinces])
    year_flows = flows.loc[flows["year"].eq(year)]
    maximum = int(year_flows["migration_flow"].max())
    with columns[3]:
        minimum = st.number_input(
            "Minimum kişi sayısı",
            min_value=0,
            max_value=maximum,
            value=min(5_000, maximum),
            step=500,
            help="Yalnızca bu sayıdan daha büyük göç hareketleri değerlendirilir.",
        )

    filtered = year_flows.loc[year_flows["migration_flow"].ge(minimum)].copy()
    if origin != "Tümü":
        filtered = filtered.loc[filtered["origin_province"].eq(origin)]
    if destination != "Tümü":
        filtered = filtered.loc[filtered["destination_province"].eq(destination)]
    display_flows = filtered.nlargest(15, "migration_flow").copy()

    if display_flows.empty:
        st.warning("Bu filtrelerle gösterilecek akış bulunamadı. Minimum akış eşiğini düşürün.")
        return

    st.download_button(
        "Filtrelenmiş akışları indir",
        data=csv_bytes(filtered),
        file_name=f"goc_akislari_{year}.csv",
        mime="text/csv",
        key="download_flows",
    )

    display_flows["direction"] = (
        display_flows["origin_province"] + " → " + display_flows["destination_province"]
    )
    display_flows["person_label"] = display_flows["migration_flow"].map(fmt_int) + " kişi"
    ranked = display_flows.sort_values("migration_flow")
    figure = px.bar(
        ranked,
        x="migration_flow",
        y="direction",
        orientation="h",
        text="person_label",
        color_discrete_sequence=["#9B1C31"],
        labels={"migration_flow": "Kişi", "direction": ""},
        title=f"En Yoğun Göç Yönleri · {year}",
    )
    figure.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>%{x:,} kişi<extra></extra>",
    )
    figure.update_layout(showlegend=False, xaxis_range=[0, ranked["migration_flow"].max() * 1.18])
    st.plotly_chart(chart_layout(figure, 610), use_container_width=True)
    st.caption(
        f"Filtreye uyan {len(filtered):,} yönlü hareketten en yüksek hacimli "
        f"{len(display_flows)} tanesi gösteriliyor. Okun solu kaynak, sağı hedef ildir."
    )

    table = display_flows[
        ["origin_province", "destination_province", "migration_flow"]
    ].rename(
        columns={
            "origin_province": "Kaynak",
            "destination_province": "Hedef",
            "migration_flow": "Kişi",
        }
    )
    with st.expander("Rakamları tablo olarak göster"):
        st.dataframe(table, width="stretch", hide_index=True)

    advanced_flows = filtered.nlargest(20, "migration_flow")
    with st.expander("Gelişmiş akış diyagramını göster"):
        st.caption(
            "Bu görünüm iller arasındaki bağlantıları birlikte incelemek içindir. Çizgi kalınlığı kişi "
            "sayısını gösterir; çok sayıda il seçildiğinde görünüm doğal olarak yoğunlaşabilir."
        )
        nodes = sorted(
            set(advanced_flows["origin_province"]) | set(advanced_flows["destination_province"])
        )
        node_index = {node: index for index, node in enumerate(nodes)}
        sankey = go.Figure(
            go.Sankey(
                arrangement="snap",
                node={
                    "label": nodes,
                    "pad": 16,
                    "thickness": 16,
                    "color": "#D8DEE9",
                    "line": {"color": "#FFFFFF", "width": 0.5},
                },
                link={
                    "source": advanced_flows["origin_province"].map(node_index),
                    "target": advanced_flows["destination_province"].map(node_index),
                    "value": advanced_flows["migration_flow"],
                    "color": "rgba(155,28,49,0.28)",
                    "customdata": advanced_flows[["origin_province", "destination_province"]],
                    "hovertemplate": "%{customdata[0]} → %{customdata[1]}<br>%{value:,} kişi<extra></extra>",
                },
            )
        )
        sankey.update_layout(title=f"İlk {len(advanced_flows)} Akışın Bağlantı Görünümü · {year}")
        st.plotly_chart(chart_layout(sankey, 620), use_container_width=True)

    leader = display_flows.iloc[0]
    insight(
        f"Seçili filtrelerde en yoğun yönlü hareket <b>{leader['origin_province']} → "
        f"{leader['destination_province']}</b> hattında gerçekleşti: "
        f"<b>{fmt_int(leader['migration_flow'])} kişi</b>. Ana grafik yalnızca belirlediğiniz eşik üzerindeki "
        "en yüksek 15 yönü gösterir."
    )


def demographics_page(data: dict[str, pd.DataFrame]) -> None:
    guide(
        "Bu sayfa 2025 yılında göç eden kişilerin yaş, cinsiyet, eğitim ve beyan edilen göç "
        "nedenlerini özetler. Grafikler kişilerin neden göç ettiğini kanıtlamaz; TÜİK kayıtlarındaki "
        "dağılımı betimler. Çubukların veya hücrelerin üzerine gelerek kesin sayıları görebilirsiniz."
    )
    st.info("Bu bölümdeki demografi ve göç nedeni verileri yalnızca TÜİK 2025 kesitini gösterir.")
    age_gender = data["age_gender"].copy()
    age_gender["female_negative"] = -age_gender["female"]
    figure = go.Figure()
    figure.add_bar(
        y=age_gender["age_group"],
        x=age_gender["male"],
        name="Erkek",
        orientation="h",
        marker_color="#3D5A80",
        hovertemplate="%{y}<br>Erkek: %{x:,}<extra></extra>",
    )
    figure.add_bar(
        y=age_gender["age_group"],
        x=age_gender["female_negative"],
        name="Kadın",
        orientation="h",
        marker_color="#9B1C31",
        customdata=age_gender["female"],
        hovertemplate="%{y}<br>Kadın: %{customdata:,}<extra></extra>",
    )
    figure.update_layout(
        barmode="relative",
        title="Yaş ve Cinsiyete Göre İller Arası Göç",
        xaxis_title="Kişi",
        yaxis_title="Yaş grubu",
    )
    st.plotly_chart(chart_layout(figure, 570), use_container_width=True)

    largest_age = age_gender.loc[age_gender["total"].idxmax()]
    female_total = age_gender["female"].sum()
    male_total = age_gender["male"].sum()
    insight(
        f"En fazla göç hareketi <b>{largest_age['age_group']} yaş grubunda</b> görüldü "
        f"({fmt_int(largest_age['total'])} kişi). Toplam göç edenler içinde kadın sayısı "
        f"{fmt_int(female_total)}, erkek sayısı {fmt_int(male_total)}."
    )

    age_reason = data["age_reason"]
    reason_totals = (
        age_reason.groupby("reason", as_index=False)
        .agg(count=("count", lambda values: values.sum(min_count=1)))
        .sort_values("count")
    )
    figure = px.bar(
        reason_totals,
        x="count",
        y="reason",
        orientation="h",
        color_discrete_sequence=["#3D5A80"],
        labels={"count": "Kişi", "reason": ""},
        title="Göç Nedenleri",
    )
    st.plotly_chart(chart_layout(figure, 560), use_container_width=True)

    leading_reason = reason_totals.iloc[-1]
    insight(
        f"2025'te en sık kaydedilen göç nedeni <b>{leading_reason['reason']}</b> oldu "
        f"({fmt_int(leading_reason['count'])} kişi). “Bilinmeyen” ayrı bir kayıt kategorisidir; "
        "belirli bir göç nedeni olarak yorumlanmamalıdır."
    )

    left, right = st.columns(2)
    with left:
        gender_reason = (
            age_reason.groupby(["sex", "reason"], as_index=False)["count"]
            .sum(min_count=1)
        )
        top_reasons = reason_totals.nlargest(6, "count")["reason"]
        figure = px.bar(
            gender_reason.loc[gender_reason["reason"].isin(top_reasons)],
            x="count",
            y="reason",
            color="sex",
            barmode="group",
            orientation="h",
            color_discrete_map={"Erkek": "#3D5A80", "Kadın": "#9B1C31"},
            labels={"count": "Kişi", "reason": "", "sex": "Cinsiyet"},
            title="Cinsiyete Göre Başlıca Nedenler",
        )
        st.plotly_chart(chart_layout(figure, 520), use_container_width=True)
    with right:
        education = data["education_reason"].pivot_table(
            index="reason", columns="education_level", values="count", aggfunc="sum"
        )
        selected_reasons = education.sum(axis=1).nlargest(6).index
        shares = education.loc[selected_reasons].div(
            education.loc[selected_reasons].sum(axis=1), axis=0
        )
        figure = px.imshow(
            shares * 100,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "Eğitim düzeyi", "y": "Göç nedeni", "color": "%"},
            title="Eğitim × Neden Dağılımı (Satır %)",
        )
        st.plotly_chart(chart_layout(figure, 520), use_container_width=True)


def model_page(data: dict[str, pd.DataFrame], pipeline: object, metadata: dict) -> None:
    guide(
        "Bu bölüm illeri 'iyi' veya 'kötü' diye puanlamaz. Benzer uzun dönemli göç davranışlarına "
        "sahip illeri aynı <b>profil grubunda</b> toplar. Noktalar birbirine yaklaştıkça göç özellikleri "
        "daha benzerdir. Teknik metrikleri bilmeniz gerekmez; profil adları ve il dağılımı ana sonuçtur."
    )
    metrics = metadata["evaluation_metrics"]
    columns = st.columns(5)
    values = [
        ("Seçilen k", str(metadata["selected_k"]), "K-Means"),
        ("Silhouette", fmt_decimal(metrics["silhouette"], 3), "yüksek daha iyi"),
        ("Davies–Bouldin", fmt_decimal(metrics["davies_bouldin"], 3), "düşük daha iyi"),
        ("Calinski–Harabasz", fmt_decimal(metrics["calinski_harabasz"], 2), "yüksek daha iyi"),
        ("Stability ARI", fmt_decimal(metrics["stability_ari"], 3), "10 seed"),
    ]
    for column, value in zip(columns, values):
        with column:
            kpi(*value)

    evaluation = pd.DataFrame(metadata["k_comparison"])
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=evaluation["k"], y=evaluation["silhouette"], name="Silhouette", mode="lines+markers", line={"color": "#3D5A80"}))
    figure.add_trace(go.Scatter(x=evaluation["k"], y=evaluation["davies_bouldin"], name="Davies–Bouldin", mode="lines+markers", line={"color": "#9B1C31"}, yaxis="y2"))
    figure.update_layout(
        title="K-Means Model Karşılaştırması",
        xaxis={"title": "k"},
        yaxis={"title": "Silhouette"},
        yaxis2={"title": "Davies–Bouldin", "overlaying": "y", "side": "right"},
        shapes=[{"type": "line", "x0": metadata["selected_k"], "x1": metadata["selected_k"], "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#9B1C31", "dash": "dash"}}],
    )
    st.plotly_chart(chart_layout(figure), use_container_width=True)

    with st.expander("Model ölçümleri ne anlama geliyor?"):
        st.markdown(
            "- **k = 4:** Model illeri dört profile ayırdı.\n"
            "- **Silhouette:** Aynı profildeki illerin birbirine ne kadar benzediğini ölçer; yüksek olması tercih edilir.\n"
            "- **Davies–Bouldin:** Profillerin birbirinden ne kadar ayrıldığını ölçer; düşük olması tercih edilir.\n"
            "- **Calinski–Harabasz:** Profil içi benzerlik ile profiller arası ayrımı birlikte değerlendirir.\n"
            "- **Stability ARI:** Model farklı başlangıçlarla çalıştırıldığında sonucun ne kadar değişmediğini gösterir."
        )

    profiles = data["profiles"]
    variance = metadata["pca_explained_variance_ratio"]
    figure = px.scatter(
        profiles,
        x="pc1",
        y="pc2",
        color="cluster_name",
        hover_name="province",
        hover_data={
            "latest_population": ":,",
            "mean_turnover_rate": ":.1f",
            "mean_net_migration_rate": ":.1f",
            "recent_net_migration_rate": ":.1f",
            "pc1": False,
            "pc2": False,
            "cluster_name": False,
        },
        labels={"pc1": f"PC1 (%{variance[0] * 100:.1f})", "pc2": f"PC2 (%{variance[1] * 100:.1f})", "cluster_name": "Profil"},
        title="PCA Düzleminde Uzun Dönemli İl Profilleri",
        color_discrete_sequence=["#9B1C31", "#3D5A80", "#2A9D8F", "#D9822B"],
    )
    figure.update_traces(marker={"size": 10, "line": {"color": "white", "width": 0.8}})
    st.plotly_chart(chart_layout(figure, 590), use_container_width=True)
    st.caption(
        "Her nokta bir ili temsil eder. Aynı renkteki iller aynı göç profilindedir. Eksenler doğrudan "
        "göç sayısı değildir; birçok özelliğin iki boyutlu özetidir. Noktanın üzerine gelerek il ayrıntılarını görün."
    )

    summary = (
        profiles.groupby("cluster_name", as_index=False)
        .agg(
            il_sayisi=("province", "size"),
            ort_hareketlilik_hizi=("mean_turnover_rate", "mean"),
            ort_net_goc_hizi=("mean_net_migration_rate", "mean"),
            son_uc_yil_net_hizi=("recent_net_migration_rate", "mean"),
            ort_nufus=("latest_population", "mean"),
        )
        .round(2)
    )
    st.subheader("Göç profillerinin özeti")
    st.write(
        "Aşağıdaki tablo her profil grubunda kaç il bulunduğunu ve grubun ortalama göç davranışını gösterir. "
        "Pozitif net göç hızı genel olarak nüfus kazanımına, negatif değer göç yoluyla nüfus kaybına işaret eder."
    )
    st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Örnek değerlerle profil bul")
    st.markdown(
        '<div class="method-box"><b>İsteğe bağlı araç:</b> Aşağıdaki alanları değiştirerek örnek bir ilin hangi '
        'göç profiline daha yakın olacağını görebilirsiniz. Bu işlem geleceği tahmin etmez ve resmi bir sınıflandırma değildir.</div>',
        unsafe_allow_html=True,
    )
    defaults = profiles[MODEL_FEATURES].median()
    with st.form("profile_form"):
        first, second, third = st.columns(3)
        inputs: dict[str, float] = {}
        labels = metadata["feature_descriptions"]
        for index, feature in enumerate(MODEL_FEATURES):
            column = [first, second, third][index % 3]
            with column:
                inputs[feature] = st.number_input(
                    labels[feature],
                    value=float(defaults[feature]),
                    format="%.3f",
                )
        submitted = st.form_submit_button("Profili göster")
    if submitted:
        observation = pd.DataFrame([inputs])
        cluster_id = int(pipeline.predict(observation)[0])
        name = metadata["cluster_names"][str(cluster_id)]
        st.success(f"En yakın profil: {name}")

    hdbscan = metadata["hdbscan_comparison"]
    st.caption(
        f"Karşılaştırmalı HDBSCAN deneyi {hdbscan['cluster_count']} küme ve "
        f"{hdbscan['noise_count']} noise il üretti. Production atamaları K-Means artefact'ından gelir."
    )


def methodology_page(data: dict[str, pd.DataFrame]) -> None:
    guide(
        "Bu alan ürünün veri kapsamını, hesaplama yöntemlerini ve sınırlarını tek yerde toplar. "
        "İsterseniz temizlenmiş tabloları CSV olarak indirip kendi analizinizde kullanabilirsiniz."
    )

    columns = st.columns(3)
    with columns[0]:
        kpi("Kapsanan il", "81", "Türkiye geneli")
    with columns[1]:
        kpi("Ana dönem", "2008–2025", "18 yıl")
    with columns[2]:
        kpi("Yönlü akış", fmt_int(len(data["flows"])), "kaynak → hedef")

    sources_tab, method_tab, limits_tab = st.tabs(
        ["Veriler ve indirme", "Hesaplama yöntemi", "Kapsam ve sınırlar"]
    )

    with sources_tab:
        st.subheader("Kullanılan veri setleri")
        source_table = pd.DataFrame(
            [
                ["İller arası göç", "2008–2025", "Yıl × kaynak il × hedef il", "Akış, trend ve koridor"],
                ["İl-yıl göstergeleri", "2008–2025", "Yıl × il", "Gelen, giden, net göç ve hızlar"],
                ["Yaş ve cinsiyet", "2025", "Yaş grubu × cinsiyet", "Demografik dağılım"],
                ["Yaş ve göç nedeni", "2025", "Yaş × cinsiyet × neden", "Göç nedeni analizi"],
                ["Eğitim ve göç nedeni", "2025", "Eğitim × neden", "Eğitim-neden ilişkisi"],
                ["Şehir profilleri", "2008–2025", "İl", "Uzun dönemli davranış grupları"],
            ],
            columns=["Veri seti", "Kapsam", "Gözlem düzeyi", "Kullanım"],
        )
        st.dataframe(source_table, width="stretch", hide_index=True)

        st.subheader("Temizlenmiş verileri indir")
        downloads = [
            ("İller arası akışlar", data["flows"], "iller_arasi_goc_2008_2025.csv"),
            ("İl-yıl göstergeleri", data["city_year"], "il_yil_goc_gostergeleri.csv"),
            ("Yaş ve cinsiyet", data["age_gender"], "yas_cinsiyet_goc_2025.csv"),
            ("Yaş ve göç nedeni", data["age_reason"], "yas_goc_nedeni_2025.csv"),
            ("Eğitim ve göç nedeni", data["education_reason"], "egitim_goc_nedeni_2025.csv"),
            ("Şehir profilleri", data["profiles"], "sehir_goc_profilleri.csv"),
        ]
        download_columns = st.columns(3)
        for index, (label, frame, file_name) in enumerate(downloads):
            with download_columns[index % 3]:
                st.markdown(f"**{label}**  \n{fmt_int(len(frame))} satır")
                st.download_button(
                    "CSV indir",
                    data=csv_bytes(frame),
                    file_name=file_name,
                    mime="text/csv",
                    key=f"method_download_{index}",
                    use_container_width=True,
                )

    with method_tab:
        st.subheader("Temel hesaplamalar")
        st.markdown(
            "1. **Aldığı göç**, diğer 80 ilden seçili ile gelen yönlü akışların toplamıdır.\n"
            "2. **Verdiği göç**, seçili ilden diğer 80 ile giden yönlü akışların toplamıdır.\n"
            "3. **Net göç = aldığı göç − verdiği göç.**\n"
            "4. **Net göç hızı**, TÜİK'in dönem ortası nüfus yaklaşımıyla binde (‰) hesaplanır.\n"
            "5. **Göç hareketliliği**, gelen ve giden hareket toplamının nüfusa oranıdır; "
            "arayüzde yüzde (%) gösterilir.\n"
            "6. **Şehir profilleri**, 81 ilin 2008–2025 davranışını altı uzun dönemli özellikle "
            "karşılaştıran K-Means segmentasyonudur."
        )
        st.info(
            "Göç yönlüdür: İstanbul → Ankara ile Ankara → İstanbul iki farklı harekettir. "
            "Ulusal toplam hesaplanırken her yönlü kayıt yalnızca bir kez sayılır."
        )

    with limits_tab:
        st.subheader("Bu ürün neyi gösterir, neyi göstermez?")
        st.markdown(
            "- Ana göç akışları 2008–2025 dönemini kapsar; demografi ve göç nedeni kırılımları yalnızca 2025'tir.\n"
            "- Grafikler kayıtlı göç hareketlerini betimler; neden–sonuç ilişkisi kanıtlamaz.\n"
            "- Şehir profilleri gelecek tahmini, başarı sıralaması veya resmî sınıflandırma değildir.\n"
            "- Gelir, konut maliyeti, istihdam ve afet etkisi gibi dış değişkenler modele dahil değildir.\n"
            "- Kaynak tablolardaki bilgi bulunmayan değerler sıfır kabul edilmemiştir."
        )
        st.warning(
            "Karar verirken göstergeleri tek başına kullanmayın; ilgili yılın yerel koşulları ve başka "
            "resmî veri kaynaklarıyla birlikte değerlendirin."
        )


try:
    data = load_data()
    pipeline, metadata = load_model_assets()
except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
    st.error(
        "Gerekli veri veya model artefact'ları yüklenemedi. "
        "Kurulum adımlarını tamamlayıp modeli yeniden üretin."
    )
    st.exception(error)
    st.stop()

with st.sidebar:
    st.markdown("## Türkiye İç Göç Atlası")
    st.caption(f"Ürün v{PRODUCT_VERSION} · Veri sürümü {DATA_VERSION}")
    page = st.radio(
        "Bölüm",
        [
            "Genel Bakış",
            "İl Profili",
            "Göç Akışları",
            "Demografi",
            "Şehir Profilleri",
            "Veri & Metodoloji",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    with st.expander("Temel kavramlar"):
        st.markdown(
            "**Aldığı göç:** Başka illerden seçili ile gelen kişi sayısı.\n\n"
            "**Verdiği göç:** Seçili ilden başka illere giden kişi sayısı.\n\n"
            "**Net göç:** Aldığı göç eksi verdiği göç.\n\n"
            "**Net göç hızı:** Net göçün il nüfusuna oranlanmış hali (binde ‰).\n\n"
            "**Göç hareketliliği:** Gelen ve giden göç toplamının il nüfusuna oranı. Kartlarda daha "
            "kolay okunması için yüzde (%) gösterilir.\n\n"
            "**Göç koridoru:** Belirli bir kaynak ve hedef il arasındaki yönlü hareket.\n\n"
            "**Göç profili:** Uzun dönemli davranışı benzer illerin oluşturduğu grup."
        )
    st.divider()
    st.markdown('<div class="source-title">Veri kaynağı</div>', unsafe_allow_html=True)
    if TUIK_LOGO_PATH.exists():
        st.image(str(TUIK_LOGO_PATH), width=175)
    st.markdown(
        '<div class="source-note"><b>Türkiye İstatistik Kurumu (TÜİK)</b><br>'
        'İç Göç İstatistikleri<br>Ana akış: 2008–2025 · Demografi: 2025<br><br>'
        'Bu çalışma TÜİK verilerinden hazırlanmış bağımsız bir analizdir; TÜİK\'in resmî uygulaması değildir.</div>',
        unsafe_allow_html=True,
    )

page_header()
if page == "Genel Bakış":
    overview_page(data)
elif page == "İl Profili":
    province_page(data, metadata)
elif page == "Göç Akışları":
    flows_page(data)
elif page == "Demografi":
    demographics_page(data)
elif page == "Şehir Profilleri":
    model_page(data, pipeline, metadata)
else:
    methodology_page(data)

page_footer()
