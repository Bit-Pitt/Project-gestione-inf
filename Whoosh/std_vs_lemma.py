from whoosh.index import open_dir
import whoosh.scoring
import os
from utils import *
from bench_plotting import *

# --- Setup: Apro i due indici---
ix_base = open_dir(os.path.join("Whoosh", "II_stdAnalyzer"))
ix_lemm = open_dir(os.path.join("Whoosh", "II_lemmatized"))

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

# --- DIZIONARI METRICHE ---
models = ["tf_idf_std", "tf_idf_lemma"]
avg_curves = {m: [] for m in models}
precision_dict = {m: [] for m in models}
recall_dict = {m: [] for m in models}
ndcg_dict = {m: [] for m in models}
r_precision_dict = {m: [] for m in models}
f1_dict = {m: [] for m in models}

# --- CICLO SU QUERY ---
for entry in query_data:
    queries = entry["query"]
    fields = list(queries.keys())

    # --- STANDARD ---
    query_base = build_and_query(ix_base, queries)
    with ix_base.searcher(weighting=whoosh.scoring.TF_IDF()) as searcher_base:
        results_base = searcher_base.search(query_base, limit=10)
        retrieved_base = [hit['title'] for hit in results_base]

    golden_base = get_golden_standard(fields, queries, "goldstandard_vsm")

    rec_levels, avg_prec = precision_recall_curve_interpolated(retrieved_base, golden_base)
    avg_curves["tf_idf_std"].append(avg_prec)

    prec, rec, f1 = compute_metrics(retrieved_base, golden_base)
    ndcg = compute_ndcg(retrieved_base, golden_base)
    r_prec = compute_r_precision_at_k(retrieved_base, golden_base, k=3)

    precision_dict["tf_idf_std"].append(prec)
    recall_dict["tf_idf_std"].append(rec)
    f1_dict["tf_idf_std"].append(f1)
    ndcg_dict["tf_idf_std"].append(ndcg)
    r_precision_dict["tf_idf_std"].append(r_prec)

    # --- LEMMATIZED ---
    query_lemm = build_and_query(ix_lemm, queries,mode="LEMM")
    with ix_lemm.searcher(weighting=whoosh.scoring.TF_IDF()) as searcher_lemm:
        results_lemm = searcher_lemm.search(query_lemm, limit=10)
        retrieved_lemm = [hit['title'] for hit in results_lemm]

    golden_lemm = get_golden_standard(fields, queries, "goldstandard_lemmatized")

    rec_levels, avg_prec = precision_recall_curve_interpolated(retrieved_lemm, golden_lemm)
    avg_curves["tf_idf_lemma"].append(avg_prec)

    prec, rec, f1 = compute_metrics(retrieved_lemm, golden_lemm)
    ndcg = compute_ndcg(retrieved_lemm, golden_lemm)
    r_prec = compute_r_precision_at_k(retrieved_lemm, golden_lemm, k=3)

    precision_dict["tf_idf_lemma"].append(prec)
    recall_dict["tf_idf_lemma"].append(rec)
    f1_dict["tf_idf_lemma"].append(f1)
    ndcg_dict["tf_idf_lemma"].append(ndcg)
    r_precision_dict["tf_idf_lemma"].append(r_prec)

# --- PLOTTING ---
output_dir = "Whoosh/std_vs_lemma"
os.makedirs(output_dir, exist_ok=True)

plot_precision_recall_with_variance(avg_curves, f"{output_dir}/precision_recall_curve_media_interpolated.png",model1="tf_idf_std",model2="tf_idf_lemma")
plot_per_query_precision_recall(avg_curves, f"{output_dir}/subplot_per_query_precision_recall.png",model1="tf_idf_std",model2="tf_idf_lemma")
plot_querywise_precision_bar_chart(precision_dict, f"{output_dir}/precision_per_query_bar_chart.png",model1="tf_idf_std",model2="tf_idf_lemma")
plot_querywise_recall_bar_chart(recall_dict, f"{output_dir}/recall_per_query_bar_chart.png",model1="tf_idf_std",model2="tf_idf_lemma")
plot_querywise_ndcg_bar_chart(ndcg_dict, f"{output_dir}/ndcg_per_query_bar_chart.png",model1="tf_idf_std",model2="tf_idf_lemma")
plot_querywise_rprecision_bar_chart(r_precision_dict, f"{output_dir}/rprecision_diff_bar_chart.png",model1="tf_idf_std",model2="tf_idf_lemma")
plot_final_metric_summary_barplot(
    precision_dict,
    recall_dict,
    f1_dict,
    ndcg_dict,
    f"{output_dir}/final_metric_summary_barplot.png",
    model1="tf_idf_std",
    model2="tf_idf_lemma"
)
