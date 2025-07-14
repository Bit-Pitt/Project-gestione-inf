from fuzzywuzzy import fuzz
from math import log2
import os
import matplotlib.pyplot as plt

def compute_metrics(retrieved, golden):
    true_positives = 0
    for r in retrieved:
        for g in golden:
            if fuzz.ratio(r.lower(), g.lower()) >= 80:
                true_positives += 1
                break
    precision = true_positives / len(retrieved) if retrieved else 0
    recall = true_positives / len(golden) if golden else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    return precision, recall, f1

def compute_ndcg(retrieved, golden, k=10):
    golden_lower = [g.lower() for g in golden]
    gains = []
    for r in retrieved[:k]:
        match = any(fuzz.ratio(r.lower(), g) >= 80 for g in golden_lower)
        gains.append(1 if match else 0)
    dcg = sum(g / log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = [1] * min(len(golden), k)
    idcg = sum(g / log2(i + 2) for i, g in enumerate(ideal_gains))
    ndcg = dcg / idcg if idcg > 0 else 0
    print(f"NDCG@{k} = {ndcg:.3f}")
    return ndcg

def precision_recall_curve_interpolated(retrieved, golden, recall_levels=None):
    if recall_levels is None:
        recall_levels = [i/10 for i in range(11)]

    precisions = []
    recalls = []
    true_positives = 0
    golden_lower = [g.lower() for g in golden]
    matched_golden = set()
    for i, r in enumerate(retrieved, start=1):
        for g in golden_lower:
            if g not in matched_golden and fuzz.ratio(r.lower(), g) >= 80:
                true_positives += 1
                matched_golden.add(g)
                break
        precision = true_positives / i
        recall = true_positives / len(golden) if golden else 0
        precisions.append(precision)
        recalls.append(recall)

    interpolated_precisions = []
    for r_level in recall_levels:
        precisions_at_recall = [p for p, rec in zip(precisions, recalls) if rec >= r_level]
        interpolated_precisions.append(max(precisions_at_recall) if precisions_at_recall else 0.0)

    return recall_levels, interpolated_precisions

def average_precision_recall_curve_interpolated(retrieved, golden1, golden2, recall_levels=None):
    rec_levels1, prec1 = precision_recall_curve_interpolated(retrieved, golden1, recall_levels)
    rec_levels2, prec2 = precision_recall_curve_interpolated(retrieved, golden2, recall_levels)
    avg_prec = [(p1 + p2) / 2 for p1, p2 in zip(prec1, prec2)]
    return rec_levels1, avg_prec

def plot_average_precision_recall_curve_interpolated(results_dict, golden_standard_bm25, golden_standard_vsm):
    output_dir = "grafici"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8,6))

    for model, retrieved in results_dict.items():
        rec_levels, avg_prec = average_precision_recall_curve_interpolated(
            retrieved, golden_standard_bm25, golden_standard_vsm)
        plt.plot(rec_levels, avg_prec, marker='o', label=model)

    plt.xlabel('Recall medio')
    plt.ylabel('Precision media')
    plt.title('Curve Precision-Recall medie (tra i due Golden Standard)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(output_dir, "precision_recall_curve_media_interpolated.png")
    plt.savefig(filename)
    plt.close()
    print(f"Grafico salvato in: {filename}")

from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "NaePd5lzxh-rYkg1Aop3"))

def get_golden_standard(fields, queries, index_name):
    match_queries = [{"match": {f: queries[f]}} for f in fields]
    body = {
        "query": {"bool": {"should": match_queries, "minimum_should_match": 1}},
        "_source": ["title"],
        "size": 10
    }
    res = es.search(index=index_name, body=body)
    return [hit["_source"]["title"] for hit in res["hits"]["hits"]]
