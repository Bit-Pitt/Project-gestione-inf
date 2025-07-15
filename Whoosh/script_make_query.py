from whoosh.index import open_dir
import whoosh.scoring
from elasticsearch import Elasticsearch
import os
from utils import *




'''
Script per inserire una query in modo dinamico
Il risultato delle query sarà stampato su terminale (per entrambi i golden standard)
golden_standard1: elastic search default (basato su bm25)
golden_standard2: elastic search con similarità modificata b,k1=0 così da somigliare maggiormente a vsm
Viene inoltre mostrato il grafico della precision ai diversi livelli di recall nella cartella "grafici"
'''



# --- Setup della query dell'II ---
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


golden_standard_bm25 = get_golden_standard(fields, queries, "goldstandard_index")
golden_standard_vsm = get_golden_standard(fields, queries, "goldstandard_vsm")

print("\n--- RISULTATI ELASTICSEARCH (Golden Standard BM25) ---")
for title in golden_standard_bm25:
    print(f"Titolo: {title}")

print("\n--- RISULTATI ELASTICSEARCH (Golden Standard VSM) ---")
for title in golden_standard_vsm:
    print(f"Titolo: {title}")

metrics_summary = {}

print()
for model_name, retrieved_docs in [("BM25", retrieved_bm25), ("TF-IDF", retrieved_tfidf)]:
    precisions = []
    recalls = []
    f1s = []
    ndcgs = []
    for gs_name, golden in [("Golden BM25", golden_standard_bm25), ("Golden VSM", golden_standard_vsm)]:
    # Questa valutazione viene fatta ogni modello da testare, per entrambi i golden std
        p, r, f1 = compute_metrics(retrieved_docs, golden)
        ndcg = compute_ndcg(retrieved_docs, golden)
        print(f"Metriche per {model_name} vs {gs_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}, NDCG={ndcg:.3f}")
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ndcgs.append(ndcg)
    #dati aggregati
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)
    avg_ndcg = sum(ndcgs) / len(ndcgs)
    metrics_summary[model_name] = (avg_p, avg_r, avg_f1, avg_ndcg)

print("\n--- SCORE MEDI SUI DUE GOLDEN STANDARD ---")
for model_name, (avg_p, avg_r, avg_f1, avg_ndcg) in metrics_summary.items():
    print(f"{model_name}: Precision media={avg_p:.3f}, Recall media={avg_r:.3f}, F1 media={avg_f1:.3f}, NDCG medio={avg_ndcg:.3f}")



# --- Chiamata finale per il plot ---
plot_average_precision_recall_curve_interpolated(
    {"BM25": retrieved_bm25, "TF-IDF": retrieved_tfidf},
    golden_standard_bm25,
    golden_standard_vsm
)
