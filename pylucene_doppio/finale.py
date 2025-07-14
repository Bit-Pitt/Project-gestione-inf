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

def compute_metrics(retrieved, golden):
    tp = sum(any(fuzz.ratio(r.lower(), g.lower()) >= 80 for g in golden) for r in retrieved)
    precision = tp / len(retrieved) if retrieved else 0
    recall = tp / len(golden) if golden else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def compute_ndcg(retrieved, golden, k=10):
    golden_lower = [g.lower() for g in golden]
    gains = [1 if any(fuzz.ratio(r.lower(), g) >= 80 for g in golden_lower) else 0 for r in retrieved[:k]]
    dcg = sum(g / log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1 / log2(i + 2) for i in range(min(len(golden), k)))
    return dcg / idcg if idcg > 0 else 0

metrics_summary = {}
for model_name, retrieved in result_by_model.items():
    p1, r1, f1_1 = compute_metrics(retrieved, golden_standard_bm25)
    p2, r2, f1_2 = compute_metrics(retrieved, golden_standard_vsm)
    avg_p, avg_r, avg_f1 = (p1 + p2) / 2, (r1 + r2) / 2, (f1_1 + f1_2) / 2
    ndcg = compute_ndcg(retrieved, golden_standard_bm25 + golden_standard_vsm)
    print(f"\n📊 Metriche medie per {model_name}")
    print(f"Precision: {avg_p:.3f}, Recall: {avg_r:.3f}, F1: {avg_f1:.3f}, NDCG: {ndcg:.3f}")
    metrics_summary[model_name] = (avg_p, avg_r, avg_f1)

def precision_recall_curve(retrieved, golden):
    tp, precisions, recalls, matched = 0, [], [], set()
    golden_lower = [g.lower() for g in golden]
    for i, r in enumerate(retrieved, 1):
        for g in golden_lower:
            if g not in matched and fuzz.ratio(r.lower(), g) >= 80:
                tp += 1
                matched.add(g)
                break
        precisions.append(tp / i)
        recalls.append(tp / len(golden) if golden else 0)
    return recalls, precisions

def plot_precision_recall(results, golden):
    plt.figure()
    for model, retrieved in results.items():
        recalls, precisions = precision_recall_curve(retrieved, golden_standard_bm25 + golden_standard_vsm)
        plt.plot(recalls, precisions, marker='o', label=model)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_precision_recall(result_by_model, golden_standard_bm25 + golden_standard_vsm)
