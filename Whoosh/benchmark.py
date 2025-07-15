from whoosh.index import open_dir
import whoosh.scoring
import os
from utils import *
from bench_plotting import * 

# --- Setup ---
ix = open_dir(os.path.join("Whoosh", "II_stdAnalyzer"))

# --- Queries ---
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

# Dizionari per salvare le evaluations
avg_curves = {"BM25": [], "TF-IDF": []}        # curve della precisione ai lvl di recall
precision_dict = {"BM25": [], "TF-IDF": []}   # precision media per ogni query
recall_dict = {"BM25": [], "TF-IDF": []}      # recall media per ogni query
ndcg_dict = {"BM25": [], "TF-IDF": []}         # dizionario per le ndcg
r_precision_dict = {"BM25": [], "TF-IDF": []}   # per R@prec
f1_dict = {"BM25": [], "TF-IDF": []}            # score di f1

# Popoliamo i dizionari che conterranno le varie metriche, come lista per ogni query
for entry in query_data:
    queries = entry["query"]    # dizionario delle query
    fields = list(queries.keys())
    final_query = build_and_query(ix, queries)

    # Ricerca BM25
    with ix.searcher() as searcher_bm25:
        results_bm25 = searcher_bm25.search(final_query, limit=10)
        retrieved_bm25 = [hit['title'] for hit in results_bm25]

    # Ricerca TF-IDF
    with ix.searcher(weighting=whoosh.scoring.TF_IDF()) as searcher_tfidf:
        results_tfidf = searcher_tfidf.search(final_query, limit=10)
        retrieved_tfidf = [hit['title'] for hit in results_tfidf]

    # Golden standards
    golden_bm25 = get_golden_standard(fields, queries, "goldstandard_index")
    golden_vsm = get_golden_standard(fields, queries, "goldstandard_vsm")

    # Curve interpolated per BM25 / TF-IDF
    _, avg_prec_bm25 = average_precision_recall_curve_interpolated(retrieved_bm25, golden_bm25, golden_vsm)
    _, avg_prec_tfidf = average_precision_recall_curve_interpolated(retrieved_tfidf, golden_bm25, golden_vsm)
    avg_curves["BM25"].append(avg_prec_bm25)
    avg_curves["TF-IDF"].append(avg_prec_tfidf)

    # Precision media per BM25
    precision_bm25_g1, recall_bm25_g1, f1_bm25_g1 = compute_metrics(retrieved_bm25, golden_bm25)
    precision_bm25_g2, recall_bm25_g2, f1_bm25_g2 = compute_metrics(retrieved_bm25, golden_vsm)
    precision_media_bm25 = (precision_bm25_g1 + precision_bm25_g2) / 2
    precision_dict["BM25"].append(precision_media_bm25)

    recall_media_bm25 = (recall_bm25_g1 + recall_bm25_g2) / 2
    recall_dict["BM25"].append(recall_media_bm25)

    # NDCG media per BM25 (media tra i due golden standard)
    ndcg_bm25_g1 = compute_ndcg(retrieved_bm25, golden_bm25)
    ndcg_bm25_g2 = compute_ndcg(retrieved_bm25, golden_vsm)
    ndcg_media_bm25 = (ndcg_bm25_g1 + ndcg_bm25_g2) / 2
    ndcg_dict["BM25"].append(ndcg_media_bm25)

    # Precision media per TF-IDF
    precision_tfidf_g1, recall_tfidf_g1, f1_tfidf_g1 = compute_metrics(retrieved_tfidf, golden_bm25)
    precision_tfidf_g2, recall_tfidf_g2, f1_tfidf_g2 = compute_metrics(retrieved_tfidf, golden_vsm)
    precision_media_tfidf = (precision_tfidf_g1 + precision_tfidf_g2) / 2
    precision_dict["TF-IDF"].append(precision_media_tfidf)

    recall_media_tfidf = (recall_tfidf_g1 + recall_tfidf_g2) / 2
    recall_dict["TF-IDF"].append(recall_media_tfidf)

    ndcg_tfidf_g1 = compute_ndcg(retrieved_tfidf, golden_bm25)
    ndcg_tfidf_g2 = compute_ndcg(retrieved_tfidf, golden_vsm)
    ndcg_media_tfidf = (ndcg_tfidf_g1 + ndcg_tfidf_g2) / 2
    ndcg_dict["TF-IDF"].append(ndcg_media_tfidf)

        # R@3 per BM25 (media sui due golden standard)
    r_bm25_g1 = compute_r_precision_at_k(retrieved_bm25, golden_bm25, k=3)
    r_bm25_g2 = compute_r_precision_at_k(retrieved_bm25, golden_vsm, k=3)
    r_media_bm25 = (r_bm25_g1 + r_bm25_g2) / 2
    r_precision_dict["BM25"].append(r_media_bm25)

    # R@3 per TF-IDF (media sui due golden standard)
    r_tfidf_g1 = compute_r_precision_at_k(retrieved_tfidf, golden_bm25, k=3)
    r_tfidf_g2 = compute_r_precision_at_k(retrieved_tfidf, golden_vsm, k=3)
    r_media_tfidf = (r_tfidf_g1 + r_tfidf_g2) / 2
    r_precision_dict["TF-IDF"].append(r_media_tfidf)

    #f1
    f1_media_bm25 = (f1_bm25_g1 + f1_bm25_g2) / 2
    f1_dict["BM25"].append(f1_media_bm25)
    f1_media_tfidf = (f1_tfidf_g1 + f1_tfidf_g2) / 2
    f1_dict["TF-IDF"].append(f1_media_tfidf)




# --- Plotting grafici tramite funzioni di "bench_plotting" ---
plot_precision_recall_with_variance(avg_curves, "grafici/precision_recall_curve_media_interpolated.png")
plot_per_query_precision_recall(avg_curves, "grafici/subplot_per_query_precision_recall.png")
plot_querywise_precision_bar_chart(precision_dict, "grafici/precision_per_query_bar_chart.png")
plot_querywise_recall_bar_chart(recall_dict, "grafici/recall_per_query_bar_chart.png")
plot_querywise_ndcg_bar_chart(ndcg_dict, "grafici/ndcg_per_query_bar_chart.png")
plot_querywise_rprecision_bar_chart(r_precision_dict, "grafici/rprecision_diff_bar_chart.png")
plot_final_metric_summary_barplot(
    precision_dict,
    recall_dict,
    f1_dict,
    ndcg_dict,
    "grafici/final_metric_summary_barplot.png"
)

