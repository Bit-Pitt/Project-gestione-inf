from whoosh.index import open_dir
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import And
import whoosh.scoring
from elasticsearch import Elasticsearch
from fuzzywuzzy import fuzz
import os
import matplotlib.pyplot as plt
from math import log2

'''
Script per inserire una query in modo dinamico
Il risultato delle query sarà stampato su terminale (per entrambi i golden standard)
golden_standard1: elastic search default (basato su bm25)
golden_standard2: elastic search con similarità modificata b,k1=0 così da somigliare maggiormente a vsm
Viene inoltre mostrato il grafico della precision ai diversi livelli di recall nella cartella "grafici"
'''



# --- Setup Whoosh ---
directory = os.path.join("Whoosh", "II_stdAnalyzer")
ix = open_dir(directory)

valid_fields = ["title", "year", "genre", "country", "plot"]

fields = input(f"In quali campi vuoi cercare? {valid_fields} (separati da virgola): ").strip().lower().split(",")
fields = [f.strip() for f in fields if f.strip() in valid_fields]

if not fields:
    print("❌ Nessun campo valido selezionato!")
    exit()

queries = {}
for field in fields:
    queries[field] = input(f"Inserisci la query per '{field}': ").strip()

def build_and_query(ix, queries):
    clauses = []
    for field, text in queries.items():
        parser = QueryParser(field, schema=ix.schema, group=OrGroup)
        clauses.append(parser.parse(text))
    return And(clauses)

final_query = build_and_query(ix, queries)

# --- Ricerca Whoosh BM25 ---
searcher_bm25 = ix.searcher()
results_bm25 = searcher_bm25.search(final_query, limit=10)

print("\n--- RISULTATI WHOOSH BM25 ---")
retrieved_bm25 = []
for hit in results_bm25:
    print(f"Titolo: {hit['title']} | Score: {hit.score}")
    retrieved_bm25.append(hit['title'])

searcher_bm25.close()

# --- Ricerca Whoosh TF-IDF ---
searcher_tfidf = ix.searcher(weighting=whoosh.scoring.TF_IDF())
results_tfidf = searcher_tfidf.search(final_query, limit=10)

print("\n--- RISULTATI WHOOSH TF-IDF ---")
retrieved_tfidf = []
for hit in results_tfidf:
    print(f"Titolo: {hit['title']} | Score: {hit.score}")
    retrieved_tfidf.append(hit['title'])

searcher_tfidf.close()

# --- Ricerca Elasticsearch per Golden Standard ---
es = Elasticsearch("http://172.26.112.1:9200")  # Cambia se serve

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

golden_standard_bm25 = get_golden_standard(fields, queries, "goldstandard_index")
golden_standard_vsm = get_golden_standard(fields, queries, "goldstandard_vsm")

print("\n--- RISULTATI ELASTICSEARCH (Golden Standard BM25) ---")
for title in golden_standard_bm25:
    print(f"Titolo: {title}")

print("\n--- RISULTATI ELASTICSEARCH (Golden Standard VSM) ---")
for title in golden_standard_vsm:
    print(f"Titolo: {title}")

# --- Metriche (precision, recall, f1) con fuzzy matching ---
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
    return dcg / idcg if idcg > 0 else 0

metrics_summary = {}

print()
for model_name, retrieved_docs in [("BM25", retrieved_bm25), ("TF-IDF", retrieved_tfidf)]:
    precisions = []
    recalls = []
    f1s = []
    ndcgs = []
    for gs_name, golden in [("Golden BM25", golden_standard_bm25), ("Golden VSM", golden_standard_vsm)]:
        p, r, f1 = compute_metrics(retrieved_docs, golden)
        ndcg = compute_ndcg(retrieved_docs, golden)
        print(f"Metriche per {model_name} vs {gs_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}, NDCG={ndcg:.3f}")
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ndcgs.append(ndcg)
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)
    avg_ndcg = sum(ndcgs) / len(ndcgs)
    metrics_summary[model_name] = (avg_p, avg_r, avg_f1, avg_ndcg)

print("\n--- SCORE MEDI SUI DUE GOLDEN STANDARD ---")
for model_name, (avg_p, avg_r, avg_f1, avg_ndcg) in metrics_summary.items():
    print(f"{model_name}: Precision media={avg_p:.3f}, Recall media={avg_r:.3f}, F1 media={avg_f1:.3f}, NDCG medio={avg_ndcg:.3f}")

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

# --- Chiamata finale per il plot ---
plot_average_precision_recall_curve_interpolated(
    {"BM25": retrieved_bm25, "TF-IDF": retrieved_tfidf},
    golden_standard_bm25,
    golden_standard_vsm
)
