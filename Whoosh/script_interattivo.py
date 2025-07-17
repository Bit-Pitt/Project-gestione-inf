from whoosh.index import open_dir
from whoosh.scoring import TF_IDF
from elasticsearch import Elasticsearch
import os
from utils import *

# --- Connessione a Elasticsearch ---
es = Elasticsearch("http://172.26.112.1:9200")
if not es.ping():
    print("❌ Errore di connessione a Elasticsearch!")
    exit()
else:
    print("✅ Connesso a Elasticsearch")

# --- Campi validi per la query ---
valid_fields = ["title", "year", "genre", "country", "plot"]
fields = input(f"In quali campi vuoi cercare? {valid_fields} (separati da virgola): ").strip().lower().split(",")
fields = [f.strip() for f in fields if f.strip() in valid_fields]

if not fields:
    print("❌ Nessun campo valido selezionato!")
    exit()

queries = {}
for field in fields:
    queries[field] = input(f"Inserisci la query per '{field}': ").strip()

# --- Apertura indici Whoosh ---
index_paths = {
    "STD": os.path.join("Whoosh", "II_stdAnalyzer"),
    "LEMM": os.path.join("Whoosh", "II_lemmatized"),
    "STEM": os.path.join("Whoosh", "II_stemmed")
}
indices = {name: open_dir(path) for name, path in index_paths.items()}

# --- Costruzione delle query per ogni indice ---
query_std = build_and_query(indices["STD"], queries)
query_lemm = build_and_query(indices["LEMM"], queries,mode="LEMM")
query_stem = build_and_query(indices["STEM"], queries,mode="STEM")

# --- Ricerche da effettuare ---
retrieved_results = {}

# BM25 standard
with indices["STD"].searcher() as searcher:
    results = searcher.search(query_std, limit=10)
    retrieved_results["BM25_STD"] = [hit['title'] for hit in results]

# VSM standard
with indices["STD"].searcher(weighting=TF_IDF()) as searcher:
    results = searcher.search(query_std, limit=10)
    retrieved_results["VSM_STD"] = [hit['title'] for hit in results]

# VSM lemmatizzato
with indices["LEMM"].searcher(weighting=TF_IDF()) as searcher:
    results = searcher.search(query_lemm, limit=10)
    retrieved_results["VSM_LEMM"] = [hit['title'] for hit in results]

# VSM stemmato
with indices["STEM"].searcher(weighting=TF_IDF()) as searcher:
    results = searcher.search(query_stem, limit=10)
    retrieved_results["VSM_STEM"] = [hit['title'] for hit in results]

# --- Golden standard da Elasticsearch ---
golden_standards = {
    "BM25_STD": get_golden_standard(fields, queries, "goldstandard_index"),
    "VSM_STD": get_golden_standard(fields, queries, "goldstandard_vsm"),
    "VSM_LEMM": get_golden_standard(fields, queries, "goldstandard_lemmatized"),
    "VSM_STEM": get_golden_standard(fields, queries, "goldstandard_stemmed")
}

# --- Valutazione ---
print("\n--- RISULTATI METRICHE ---")
metrics_summary = {}
combined_golden = list(set(golden_standards["BM25_STD"]) | set(golden_standards["VSM_STD"]))

for model_name, retrieved_docs in retrieved_results.items():
    if model_name in ['BM25_STD', 'VSM_STD']:
        p1, r1, f1_1 = compute_metrics(retrieved_docs, golden_standards["BM25_STD"])
        p2, r2, f1_2 = compute_metrics(retrieved_docs, golden_standards["VSM_STD"])
        ndcg1 = compute_ndcg(retrieved_docs, golden_standards["BM25_STD"])
        ndcg2 = compute_ndcg(retrieved_docs, golden_standards["VSM_STD"])
        
        p, r, f1 = (p1 + p2)/2, (r1 + r2)/2, (f1_1 + f1_2)/2
        ndcg = (ndcg1 + ndcg2)/2
        
        metrics_summary[model_name] = {"P": p, "R": r, "F1": f1, "NDCG": ndcg}
        print(f"{model_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}, NDCG={ndcg:.3f}")
    else:
        golden = golden_standards[model_name]
        p, r, f1 = compute_metrics(retrieved_docs, golden)
        ndcg = compute_ndcg(retrieved_docs, golden)
        metrics_summary[model_name] = {"P": p, "R": r, "F1": f1, "NDCG": ndcg}
        print(f"{model_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}, NDCG={ndcg:.3f}")

plot_interpolated_precision_curves(     #associ i modelli da plottare il golden collegato (per poter calcolare la precision ai lvl di recall)
    {
        "BM25_STD": retrieved_results["BM25_STD"],
        "VSM_STD": retrieved_results["VSM_STD"],
        "VSM_LEMM": retrieved_results["VSM_LEMM"],
        "VSM_STEM": retrieved_results["VSM_STEM"]
    },
    {
        "BM25_STD": combined_golden,
        "VSM_STD": combined_golden,
        "VSM_LEMM": golden_standards["VSM_LEMM"],
        "VSM_STEM": golden_standards["VSM_STEM"]
    }
)

