from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import And
import os
import matplotlib.pyplot as plt
from math import log2
from elasticsearch import Elasticsearch 



# --- Ricerca Elasticsearch per Golden Standard ---
es = Elasticsearch("http://172.26.112.1:9200")  # Connessione al servizio

# group=OrGroup fa si che internamente il parser tratta ogni parola in OR
# And(clauses) ==> ogni singola query viene trattata come and
def build_and_query(ix, queries):
    clauses = []
    for field, text in queries.items():
        parser = QueryParser(field, schema=ix.schema, group=OrGroup)
        clauses.append(parser.parse(text))
    return And(clauses)


def get_golden_standard(fields, queries, index_name):
    match_queries = [{"match": {f: queries[f]}} for f in fields]
    body = {
        "query": {
            "bool": {
                "should": match_queries,
                "minimum_should_match": 1
            }
        },
        "_source": ["title"],
        "size": 10
    }
    res = es.search(index=index_name, body=body)
    return [hit['_source']['title'] for hit in res['hits']['hits']]



# --- Metriche (precision, recall, f1)  ---
def compute_metrics(retrieved, golden):
    true_positives = 0
    for r in retrieved:
        for g in golden:
            if r.lower() == g.lower():
                true_positives += 1
                break
    precision = true_positives / len(retrieved) if retrieved else 0
    recall = true_positives / len(golden) if golden else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1



def compute_ndcg(retrieved, golden, k=10):
    golden_lower = [g.lower() for g in golden]
    gains = []
    for r in retrieved[:k]:
        match = r.lower() in golden_lower
        gains.append(1 if match else 0)
    dcg = sum(g / log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = [1] * min(len(golden), k)
    idcg = sum(g / log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0



# --- Precision-Recall interpolated curve per un golden standard ---
def precision_recall_curve_interpolated(retrieved, golden, recall_levels=None):
    if recall_levels is None:
        recall_levels = [i/10 for i in range(11)]  # 0.0, 0.1, ..., 1.0

    precisions = []
    recalls = []
    true_positives = 0
    golden_lower = [g.lower() for g in golden]
    matched_golden = set()
    for i, r in enumerate(retrieved, start=1):
        for g in golden_lower:
            if g not in matched_golden and r.lower() == g:
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
        if precisions_at_recall:
            interpolated_precisions.append(max(precisions_at_recall))
        else:
            interpolated_precisions.append(0.0)

    return recall_levels, interpolated_precisions

# --- Media curve interpolata tra i due golden standard ---
def average_precision_recall_curve_interpolated(retrieved, golden1, golden2, recall_levels=None):
    rec_levels1, prec1 = precision_recall_curve_interpolated(retrieved, golden1, recall_levels)
    rec_levels2, prec2 = precision_recall_curve_interpolated(retrieved, golden2, recall_levels)
    avg_prec = [(p1 + p2) / 2 for p1, p2 in zip(prec1, prec2)]
    return rec_levels1, avg_prec


# Computa la r@precision
def compute_r_precision_at_k(retrieved, golden, k=3):
    retrieved_k = retrieved[:k]
    matched = 0
    golden_lower = [g.lower() for g in golden]

    for r in retrieved_k:
        if r.lower() in golden_lower:
            matched += 1

    return matched / k if k else 0


# --- Plot precision-recall interpolated curve media e salvataggio ---
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



