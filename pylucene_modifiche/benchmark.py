import os
from utils import *
from bench_plotting import *

from lucene import initVM
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity
from java.nio.file import Paths

initVM()

# --- Setup indice Lucene ---
index_path = os.path.join("II_stdAnalyzer")  # o "Lucene_II_lemmatized"
directory = FSDirectory.open(Paths.get(index_path))
reader = DirectoryReader.open(directory)

# --- Setup Searcher per BM25 e TF-IDF ---
searcher_bm25 = IndexSearcher(reader)
searcher_bm25.setSimilarity(BM25Similarity())
searcher_tfidf = IndexSearcher(reader)
searcher_tfidf.setSimilarity(ClassicSimilarity())

# --- Setup analyzer ---
analyzer = StandardAnalyzer()

# --- Query batch ---
query_data = [
    {"uin": 1, "query": {"plot": "action movie set in New York", "country": "United States", "genre": "action"}},
    {"uin": 2, "query": {"plot": "Curse* causes victims to kill* themselves", "genre": "horror"}},
    {"uin": 3, "query": {"plot": "fantasy or science fictions or action movie that talks about aliens"}},
    {"uin": 4, "query": {"plot": "a person unknowngly lives inside a massive tv shw"}},
    {"uin": 5, "query": {"plot": "young wizard school magic* battle dark* force* wand"}},
    {"uin": 6, "query": {"plot": "A boy discovers he has supernatural abilities and is drawn into a hidden world of ancient powers."}},
    {"uin": 7, "query": {"plot": "A group of unlikely heroes must destroy a powerful artifact while being hunted by ancient evil across vast mythical lands."}},
    {"uin": 8, "query": {"plot": "family dealing with major loss", "genre": "drama"}},
    {"uin": 9, "query": {"plot": "epic journey"}},
    {"uin": 10, "query": {"plot": "A character goes through major personal struggles and faces overwhelming challenges. Along the way, they uncover a deep truth about themselves or their past."}},
    {"uin": 11, "query": {"plot": "man with superpower"}},
    {"uin": 12, "query": {"plot": "detective investigates murders"}},
    {"uin": 13, "query": {"plot": "Horror movie with kill*", "country": "America United States"}},
    {"uin": 14, "query": {"title": "god"}},
    {"uin": 15, "query": {"plot": "A small town is terrorized by a mysterious creature that appears only at night."}}
]

# --- Dizionari per metriche ---
avg_curves = {"BM25": [], "TF-IDF": []}
precision_dict = {"BM25": [], "TF-IDF": []}
recall_dict = {"BM25": [], "TF-IDF": []}
ndcg_dict = {"BM25": [], "TF-IDF": []}
r_precision_dict = {"BM25": [], "TF-IDF": []}
f1_dict = {"BM25": [], "TF-IDF": []}

# --- Esecuzione per ogni query ---
for entry in query_data:
    queries = entry["query"]
    fields = list(queries.keys())

    # --- Costruzione della BooleanQuery Lucene ---
    bq_builder = BooleanQuery.Builder()
    for field, query_text in queries.items():
        parsed_query = QueryParser(field, analyzer).parse(query_text)
        bq_builder.add(parsed_query, BooleanClause.Occur.MUST)
    final_query = bq_builder.build()

    # --- Ricerca BM25 ---
    hits_bm25 = searcher_bm25.search(final_query, 10).scoreDocs
    retrieved_bm25 = [reader.storedFields().document(hit.doc).get("title") for hit in hits_bm25]

    # --- Ricerca TF-IDF ---
    hits_tfidf = searcher_tfidf.search(final_query, 10).scoreDocs
    retrieved_tfidf = [reader.storedFields().document(hit.doc).get("title") for hit in hits_tfidf]

    # --- Golden standards ---
    golden_bm25 = get_golden_standard(fields, queries, "goldstandard_index")
    golden_vsm = get_golden_standard(fields, queries, "goldstandard_vsm")

    # --- Curve interpolated ---
    _, avg_prec_bm25 = average_precision_recall_curve_interpolated(retrieved_bm25, golden_bm25, golden_vsm)
    _, avg_prec_tfidf = average_precision_recall_curve_interpolated(retrieved_tfidf, golden_bm25, golden_vsm)
    avg_curves["BM25"].append(avg_prec_bm25)
    avg_curves["TF-IDF"].append(avg_prec_tfidf)

    # --- Metriche ---
    for model_name, retrieved in [("BM25", retrieved_bm25), ("TF-IDF", retrieved_tfidf)]:
        p1, r1, f1_1 = compute_metrics(retrieved, golden_bm25)
        p2, r2, f1_2 = compute_metrics(retrieved, golden_vsm)
        ndcg1 = compute_ndcg(retrieved, golden_bm25)
        ndcg2 = compute_ndcg(retrieved, golden_vsm)
        rprec1 = compute_r_precision_at_k(retrieved, golden_bm25, k=3)
        rprec2 = compute_r_precision_at_k(retrieved, golden_vsm, k=3)

        precision_dict[model_name].append((p1 + p2) / 2)
        recall_dict[model_name].append((r1 + r2) / 2)
        f1_dict[model_name].append((f1_1 + f1_2) / 2)
        ndcg_dict[model_name].append((ndcg1 + ndcg2) / 2)
        r_precision_dict[model_name].append((rprec1 + rprec2) / 2)

# --- Plotting finale ---
plot_precision_recall_with_variance(avg_curves, "./grafici/precision_recall_curve_media_interpolated.png")
plot_per_query_precision_recall(avg_curves, "./grafici/subplot_per_query_precision_recall.png")
plot_querywise_precision_bar_chart(precision_dict, "./grafici/precision_per_query_bar_chart.png")
plot_querywise_recall_bar_chart(recall_dict, "./grafici/recall_per_query_bar_chart.png")
plot_querywise_ndcg_bar_chart(ndcg_dict, "./grafici/ndcg_per_query_bar_chart.png")
plot_querywise_rprecision_bar_chart(r_precision_dict, "./grafici/rprecision_diff_bar_chart.png")
plot_final_metric_summary_barplot(
    precision_dict,
    recall_dict,
    f1_dict,
    ndcg_dict,
    "./grafici/final_metric_summary_barplot.png"
)

print("=== R@3 BM25 ===")
print(r_precision_dict["BM25"])
print("=== R@3 TF-IDF ===")
print(r_precision_dict["TF-IDF"])
print("=== Differenza ===")
print([bm25 - tfidf for bm25, tfidf in zip(r_precision_dict["BM25"], r_precision_dict["TF-IDF"])])


reader.close()
