"""Compare BM25 and Chroma dense retrieval on the BEIR SciFact test split."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
QRELS_PATH = DATA_DIR / "querl" / "test.tsv"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
CHROMA_DIR = PROJECT_ROOT / "Chroma_db_scifact"
COLLECTION_NAME = "scifact_test_benchmark"
K = 10
BATCH_SIZE = 64


def read_jsonl(path: Path) -> dict[str, dict]:
    """Read a JSONL file and return its rows keyed by the BEIR document/query ID."""
    rows = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            rows[str(row["_id"])] = row
    return rows


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Read BEIR's TSV relevance labels, skipping its header row."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as file:
        next(file, None)
        for line in file:
            query_id, corpus_id, score = line.rstrip("\n").split("\t")
            qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels


def tokenize(text: str) -> list[str]:
    """Lowercase text and keep alphanumeric terms for the BM25 baseline."""
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    """Return reciprocal rank of the first relevant result, or zero if absent."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def score_query(ranked_ids: list[str], labels: dict[str, int], k: int) -> dict[str, float]:
    """Calculate nDCG@k, Recall@k, and MRR@k for one query."""
    relevant = {doc_id for doc_id, score in labels.items() if score > 0}
    top_ids = ranked_ids[:k]

    dcg = sum(
        (2 ** labels.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_ids, start=1)
    )
    ideal_gains = sorted((2 ** score - 1 for score in labels.values()), reverse=True)[:k]
    ideal_dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))

    return {
        f"NDCG@{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
        f"Recall@{k}": len(set(top_ids) & relevant) / len(relevant) if relevant else 0.0,
        f"MRR@{k}": reciprocal_rank(top_ids, relevant),
    }


def mean_scores(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average each metric over the evaluated queries."""
    if not per_query:
        raise ValueError("No test queries with relevance labels were found.")
    metric_names = next(iter(per_query.values()))
    return {
        metric: sum(values[metric] for values in per_query.values()) / len(per_query)
        for metric in metric_names
    }


def run_bm25(corpus: dict[str, dict], queries: dict[str, dict], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Run an in-memory BM25 index over the full SciFact corpus."""
    corpus_ids = list(corpus)
    corpus_texts = [
        f"{corpus[doc_id].get('title', '')}\n{corpus[doc_id].get('text', '')}"
        for doc_id in corpus_ids
    ]
    bm25 = BM25Okapi([tokenize(text) for text in corpus_texts])

    per_query = {}
    for query_id, labels in qrels.items():
        query = queries[query_id]["text"]
        scores = bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:K]
        ranked_ids = [corpus_ids[index] for index in ranked_indices]
        per_query[query_id] = score_query(ranked_ids, labels, K)

    return mean_scores(per_query)


def make_svg(metrics: dict[str, dict[str, float]], path: Path) -> None:
    """Write a dependency-free grouped bar chart as an SVG file."""
    width, height = 900, 500
    left, right, top, bottom = 90, 30, 75, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    metric_names = list(next(iter(metrics.values())))
    systems = list(metrics)
    colors = ["#4472C4", "#ED7D31"]
    group_width = plot_width / len(metric_names)
    bar_width = 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">SciFact Test Retrieval Metrics</text>',
    ]

    for tick in range(6):
        value = tick / 5
        y = top + plot_height * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{value:.1f}</text>')

    for metric_index, metric in enumerate(metric_names):
        center = left + group_width * (metric_index + 0.5)
        for system_index, system in enumerate(systems):
            value = metrics[system][metric]
            bar_x = center + (system_index - (len(systems)-1)/2) * (bar_width + 8) - bar_width/2
            bar_height = plot_height * value
            bar_y = top + plot_height - bar_height
            parts.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[system_index]}"/>')
            parts.append(f'<text x="{bar_x + bar_width/2:.1f}" y="{bar_y-7:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{height-48}" text-anchor="middle" font-family="Arial" font-size="15">{metric}</text>')

    legend_x = width / 2 - 120
    for index, system in enumerate(systems):
        x = legend_x + index * 150
        parts.append(f'<rect x="{x}" y="{height-25}" width="16" height="16" fill="{colors[index]}"/>')
        parts.append(f'<text x="{x+23}" y="{height-12}" font-family="Arial" font-size="14">{system}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def run_dense(corpus: dict[str, dict], queries: dict[str, dict], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Index SciFact in a separate Chroma collection and evaluate test queries."""
    load_dotenv(PROJECT_ROOT / ".env")
    sys.path.insert(0, str(PROJECT_ROOT))
    from model.config import embeddings

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(vectorstore.get(include=["metadatas"])["ids"])
    missing_ids = [doc_id for doc_id in corpus if doc_id not in existing_ids]
    print(f"SciFact dense index: {len(existing_ids)} existing, {len(missing_ids)} to embed.")

    for start in range(0, len(missing_ids), BATCH_SIZE):
        batch_ids = missing_ids[start:start + BATCH_SIZE]
        batch_texts = [
            f"{corpus[doc_id].get('title', '')}\n{corpus[doc_id].get('text', '')}"
            for doc_id in batch_ids
        ]
        batch_metadatas = [{"corpus_id": doc_id} for doc_id in batch_ids]
        vectorstore.add_texts(batch_texts, metadatas=batch_metadatas, ids=batch_ids)
        print(f"Embedded and indexed {min(start + len(batch_ids), len(missing_ids))}/{len(missing_ids)} SciFact documents.")

    query_ids = list(qrels)
    query_texts = [queries[query_id]["text"] for query_id in query_ids]
    query_vectors = []
    for start in range(0, len(query_texts), BATCH_SIZE):
        query_vectors.extend(embeddings.embed_documents(query_texts[start:start + BATCH_SIZE]))

    per_query = {}
    for query_id, query_vector in zip(query_ids, query_vectors):
        results = vectorstore.similarity_search_by_vector(query_vector, k=K)
        ranked_ids = [doc.metadata["corpus_id"] for doc in results]
        per_query[query_id] = score_query(ranked_ids, qrels[query_id], K)

    return mean_scores(per_query)


def main() -> None:
    corpus = read_jsonl(DATA_DIR / "corpus.jsonl")
    queries = read_jsonl(DATA_DIR / "queries.jsonl")
    qrels = read_qrels(QRELS_PATH)

    missing_queries = set(qrels) - set(queries)
    missing_documents = {doc_id for labels in qrels.values() for doc_id in labels} - set(corpus)
    if missing_queries or missing_documents:
        raise ValueError(f"Dataset files do not align: {len(missing_queries)} missing queries, {len(missing_documents)} missing corpus documents.")

    print(f"SciFact test split: {len(corpus)} corpus docs, {len(qrels)} labeled queries.")
    results = {"BM25": run_bm25(corpus, queries, qrels)}
    results["Dense"] = run_dense(corpus, queries, qrels)

    EVALUATION_DIR.mkdir(exist_ok=True)
    metrics_path = EVALUATION_DIR / "scifact_test_metrics.json"
    chart_path = EVALUATION_DIR / "scifact_test_metrics.svg"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    make_svg(results, chart_path)

    print("\nMean test metrics:")
    for system, system_metrics in results.items():
        print(f"{system}: " + ", ".join(f"{metric}={value:.4f}" for metric, value in system_metrics.items()))
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
