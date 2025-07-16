import lucene
import os
from elasticsearch import Elasticsearch
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity
from java.nio.file import Paths
from utils import *

lucene.initVM()

'''
Script per inserire una query in modo dinamico
Il risultato delle query sarà stampato su terminale (per entrambi i golden standard)
golden_standard1: elasticsearch default (BM25)
golden_standard2: elasticsearch con similarità VSM-like
Viene inoltre mostrato il grafico della precision ai diversi livelli di recall nella cartella "grafici"
'''

# --- Setup indice Lucene ---
directory = os.path.join("II_stdAnalyzer")  # o "Lucene_II_lemmatized"
index_dir = FSDirectory.open(Paths.get(directory))
reader = DirectoryReader.open(index_dir)

# --- Searcher VSM e BM25 ---
searcher_bm25 = IndexSearcher(reader)
searcher_bm25.setSimilarity(BM25Similarity())

searcher_tfidf = IndexSearcher(reader)
searcher_tfidf.setSimilarity(ClassicSimilarity())

# --- Input query ---
valid_fields = ["title", "year", "genre", "country", "plot"]
fields = input(f"In quali campi vuoi cercare? {valid_fields} (separati da virgola): ").strip().lower().split(",")
fields = [f.strip() for f in fields if f.strip() in valid_fields]

if not fields:
    print("❌ Nessun campo valido selezionato!")
    reader.close()
    exit()

queries = {}
for field in fields:
    queries[field] = input(f"Inserisci la query per '{field}': ").strip()

# --- Costruzione query PyLucene ---
analyzer = StandardAnalyzer()
bq = BooleanQuery.Builder()
for field, q in queries.items():
    parsed = QueryParser(field, analyzer).parse(q)
    bq.add(parsed, BooleanClause.Occur.MUST)
final_query = bq.build()

# --- Ricerca BM25 ---
print("\n--- RISULTATI PYLUCENE BM25 ---")
retrieved_bm25 = []
hits_bm25 = searcher_bm25.search(final_query, 10).scoreDocs
for hit in hits_bm25:
    doc = reader.storedFields().document(hit.doc)
    print(f"Titolo: {doc.get('title')} | Score: {hit.score}")
    retrieved_bm25.append(doc.get("title"))

# --- Ricerca TF-IDF ---
print("\n--- RISULTATI PYLUCENE TF-IDF ---")
retrieved_tfidf = []
hits_tfidf = searcher_tfidf.search(final_query, 10).scoreDocs
for hit in hits_tfidf:
    doc = reader.storedFields().document(hit.doc)
    print(f"Titolo: {doc.get('title')} | Score: {hit.score}")
    retrieved_tfidf.append(doc.get("title"))

# --- Golden standard da Elasticsearch ---
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "NaePd5lzxh-rYkg1Aop3"))
if not es.ping():
    print("❌ Connessione a Elasticsearch fallita")
    exit()

golden_standard_bm25 = get_golden_standard(fields, queries, "goldstandard_index")
golden_standard_vsm = get_golden_standard(fields, queries, "goldstandard_vsm")

print("\n--- RISULTATI ELASTICSEARCH (Golden Standard BM25) ---")
for title in golden_standard_bm25:
    print(f"Titolo: {title}")

print("\n--- RISULTATI ELASTICSEARCH (Golden Standard VSM) ---")
for title in golden_standard_vsm:
    print(f"Titolo: {title}")

# --- Metriche ---
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
    metrics_summary[model_name] = (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1s) / len(f1s),
        sum(ndcgs) / len(ndcgs)
    )

print("\n--- SCORE MEDI SUI DUE GOLDEN STANDARD ---")
for model_name, (avg_p, avg_r, avg_f1, avg_ndcg) in metrics_summary.items():
    print(f"{model_name}: Precision media={avg_p:.3f}, Recall media={avg_r:.3f}, F1 media={avg_f1:.3f}, NDCG medio={avg_ndcg:.3f}")

# --- Plot finale ---
plot_average_precision_recall_curve_interpolated(
    {"BM25": retrieved_bm25, "TF-IDF": retrieved_tfidf},
    golden_standard_bm25,
    golden_standard_vsm
)

reader.close()
