import lucene
import os
from fuzzywuzzy import fuzz
from elasticsearch import Elasticsearch
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity
from java.nio.file import Paths
from math import log2
import matplotlib.pyplot as plt

lucene.initVM()

# === Percorso indice PyLucene ===
index_path = "./II_stdAnalyzer"  # path all'indice creato
directory = FSDirectory.open(Paths.get(index_path))
reader = DirectoryReader.open(directory)

# === Searcher con VSM e BM25 ===
searcher_vsm = IndexSearcher(reader)
searcher_vsm.setSimilarity(ClassicSimilarity())
searcher_bm25 = IndexSearcher(reader)
searcher_bm25.setSimilarity(BM25Similarity())

# === Campi ===
valid_fields = ["title", "year", "genre", "country", "plot"]
fields = input(f"In quali campi vuoi cercare? {valid_fields} (separati da virgola): ").strip().lower().split(",")
fields = [f.strip() for f in fields if f.strip() in valid_fields]
if not fields:
    print("❌ Nessun campo valido selezionato!")
    reader.close()
    exit()

queries = {field: input(f"Inserisci la query per '{field}': ").strip() for field in fields}

# === Costruzione Query Lucene ===
analyzer = StandardAnalyzer()
bq = BooleanQuery.Builder()
for field, query_string in queries.items():
    query = QueryParser(field, analyzer).parse(query_string)
    bq.add(query, BooleanClause.Occur.MUST)
final_query = bq.build()

# === Esecuzione Lucene ===
result_by_model = {}
for model_name, searcher in [("BM25", searcher_bm25), ("TF-IDF", searcher_vsm)]:
    hits = searcher.search(final_query, 10).scoreDocs
    result_by_model[model_name] = []
    print(f"\n--- RISULTATI PyLucene {model_name} ---")
    for hit in hits:
        doc = reader.storedFields().document(hit.doc)
        title = doc.get("title")
        print(f"Titolo: {title} | Score: {hit.score}")
        result_by_model[model_name].append(title)

# === Connessione Elasticsearch ===
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "NaePd5lzxh-rYkg1Aop3"))
if not es.ping():
    print("❌ Errore di connessione ad Elasticsearch!")
    reader.close()
    exit()

def get_golden_standard(fields, queries, index_name):
    match_queries = [{"match": {f: queries[f]}} for f in fields]
    body = {
        "query": {"bool": {"should": match_queries, "minimum_should_match": 1}},
        "_source": ["title"],
        "size": 10
    }
    res = es.search(index=index_name, body=body)
    return [hit["_source"]["title"] for hit in res["hits"]["hits"]]

golden_standard_bm25 = get_golden_standard(fields, queries, "goldstandard_index")
golden_standard_vsm = get_golden_standard(fields, queries, "goldstandard_vsm")

print("\n--- RISULTATI ELASTICSEARCH (Golden BM25) ---")
for title in golden_standard_bm25:
    print(f"Titolo: {title}")
print("\n--- RISULTATI ELASTICSEARCH (Golden VSM) ---")
for title in golden_standard_vsm:
    print(f"Titolo: {title}")

retrieved_tfidf = result_by_model["TF-IDF"]
retrieved_bm25 = result_by_model["BM25"]

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