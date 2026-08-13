"""Yönlü ve ağırlıklı iller arası göç ağı metrikleri."""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_migration_graph(flows: pd.DataFrame, year: int) -> nx.DiGraph:
    """Kaynak→hedef yönünde, kişi sayısı ağırlıklı bir ağ kurar."""

    selected = flows.loc[flows["year"].eq(year)]
    if selected.empty:
        raise ValueError(f"{year} yılı için akış bulunamadı.")
    graph = nx.from_pandas_edgelist(
        selected,
        source="origin_province",
        target="destination_province",
        edge_attr="migration_flow",
        create_using=nx.DiGraph,
    )
    nx.set_edge_attributes(
        graph,
        {(source, target): 1 / data["migration_flow"] for source, target, data in graph.edges(data=True)},
        "distance",
    )
    return graph


def province_network_metrics(flows: pd.DataFrame, year: int) -> pd.DataFrame:
    """Ağırlıklı giriş/çıkış gücü ve PageRank hesaplar."""

    graph = build_migration_graph(flows, year)
    pagerank = nx.pagerank(graph, weight="migration_flow")
    metrics = pd.DataFrame(
        {
            "province": list(graph.nodes),
            "weighted_in_strength": [
                graph.in_degree(node, weight="migration_flow") for node in graph.nodes
            ],
            "weighted_out_strength": [
                graph.out_degree(node, weight="migration_flow") for node in graph.nodes
            ],
            "pagerank": [pagerank[node] for node in graph.nodes],
        }
    )
    metrics["network_net_flow"] = (
        metrics["weighted_in_strength"] - metrics["weighted_out_strength"]
    )
    return metrics.sort_values("pagerank", ascending=False, ignore_index=True)


def strongest_corridors(
    flows: pd.DataFrame, year: int, top_n: int = 15
) -> pd.DataFrame:
    """En yüksek hacimli yönlü kaynak→hedef koridorlarını döndürür."""

    return (
        flows.loc[
            flows["year"].eq(year),
            ["origin_province", "destination_province", "migration_flow"],
        ]
        .nlargest(top_n, "migration_flow")
        .rename(
            columns={
                "origin_province": "origin",
                "destination_province": "destination",
                "migration_flow": "flow",
            }
        )
        .reset_index(drop=True)
    )
