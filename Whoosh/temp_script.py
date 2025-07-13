from whoosh.index import open_dir
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import And
import whoosh.scoring
from elasticsearch import Elasticsearch
from fuzzywuzzy import fuzz
import os

# --- Setup Whoosh ---
directory = os.path.join("Whoosh", "II_stdAnalyzer")
ix = open_dir(directory)

# Mappatura campi (come in PyLucene)
valid_fields = ["title", "year", "genre", "country", "plot"]

# --- Input campi e query ---
fields = input(f"In quali campi vuoi cercare? {valid_fields} (separati da virgola): ").strip().lower().split(",")
fields = [f.strip() for f in fields if f.strip() in valid_fields]

if not fields:
    print("❌ Nessun campo valido selezionato!")
    exit()

queries = {}
for field in fields:
    queries[field] = input(f"Inserisci la query per '{field}': ").strip()

# --- Funzione per costruire query AND su campi ---
def build_and_query(ix, queries):
    clauses = []
    for field, text in queries.items():
        parser = QueryParser(field, schema=ix.schema, group=OrGroup)
        clauses.append(parser.parse(text))
    return And(clauses)

final_query = build_and_query(ix, queries)

# --- Ricerca con Whoosh BM25 (default) ---
searcher_bm25 = ix.searcher()  # default BM25F
results_bm25 = searcher_bm25.search(final_query, limit=10)

print("\n--- RISULTATI WHOOSH BM25 ---")
retrieved_bm25 = []
for hit in results_bm25:
    print(f"Titolo: {hit['title']} | Score: {hit.score}")
    retrieved_bm25.append(hit['title'])

searcher_bm25.close()

# --- Ricerca con Whoosh TF-IDF ---
searcher_tfidf = ix.searcher(weighting=whoosh.scoring.TF_IDF())
results_tfidf = searcher_tfidf.search(final_query, limit=10)

print("\n--- RISULTATI WHOOSH TF-IDF ---")
retrieved_tfidf = []
for hit in results_tfidf:
    print(f"Titolo: {hit['title']} | Score: {hit.score}")
    retrieved_tfidf.append(hit['title'])

searcher_tfidf.close()

# --- Ricerca Elasticsearch per Golden Standard ---
es = Elasticsearch("http://172.26.112.1:9200")  # cambia se serve

def get_golden_standard(fields, queries, index_name):
    match_queries = [{"match": {f: queries[f]}} for f in fields]
    body = {
        "query": {
            "bool": {
                "should": match_queries,
                "minimum_should_match": 1  # Almeno una delle query deve matchare
            }
        },
        "_source": ["title"],
        "size": 10
    }
    res = es.search(index=index_name, body=body)
    return [hit['_source']['title'] for hit in res['hits']['hits']]

# Recupero i due golden standard
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

# Per memorizzare i risultati per calcolo media
metrics_summary = {}

# Calcolo metriche per ogni combinazione modello/golden standard
for model_name, retrieved_docs in [("BM25", retrieved_bm25), ("TF-IDF", retrieved_tfidf)]:
    precisions = []
    recalls = []
    f1s = []
    for gs_name, golden in [("Golden BM25", golden_standard_bm25), ("Golden VSM", golden_standard_vsm)]:
        p, r, f1 = compute_metrics(retrieved_docs, golden)
        print(f"\nMetriche per {model_name} vs {gs_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    # Media dei due golden standard
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)
    metrics_summary[model_name] = (avg_p, avg_r, avg_f1)

print("\n--- SCORE MEDI SUI DUE GOLDEN STANDARD ---")
for model_name, (avg_p, avg_r, avg_f1) in metrics_summary.items():
    print(f"{model_name}: Precision media={avg_p:.3f}, Recall media={avg_r:.3f}, F1 media={avg_f1:.3f}")
