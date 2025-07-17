import os
from utils import *
from bench_plotting import *

from lucene import initVM
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import ClassicSimilarity
from java.nio.file import Paths

initVM()

# --- Setup Lucene ---
paths = {
    "tf_idf_std": os.path.join("II_stdAnalyzer"),
    "tf_idf_lemma": os.path.join("II_lemmatized")
}
readers = {k: DirectoryReader.open(FSDirectory.open(Paths.get(v))) for k, v in paths.items()}
searchers = {k: IndexSearcher(r) for k, r in readers.items()}
analyzer = StandardAnalyzer()
for s in searchers.values():
    s.setSimilarity(ClassicSimilarity())

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

# --- Dizionari metriche ---
models = ["tf_idf_std", "tf_idf_lemma"]
avg_curves = {m: [] for m in models}
precision_dict = {m: [] for m in models}
recall_dict = {m: [] for m in models}
ndcg_dict = {m: [] for m in models}
r_precision_dict = {m: [] for m in models}
f1_dict = {m: [] for m in models}

# --- Ciclo query ---
for entry in query_data:
    queries = entry["query"]
    fields = list(queries.keys())

    for model in models:
        bq_builder = BooleanQuery.Builder()
        for field, text in queries.items():
            parsed = QueryParser(field, analyzer).parse(text)
            bq_builder.add(parsed, BooleanClause.Occur.MUST)
        query = bq_builder.build()

        hits = searchers[model].search(query, 10).scoreDocs
        titles = [readers[model].storedFields().document(hit.doc).get("title") for hit in hits]

        # golden standard separato per ogni modello
        if model == "tf_idf_std":
            golden = get_golden_standard(fields, queries, "goldstandard_vsm")
        else:
            golden = get_golden_standard(fields, queries, "goldstandard_lemmatized")

        _, avg_prec = precision_recall_curve_interpolated(titles, golden)
        avg_curves[model].append(avg_prec)

        prec, rec, f1 = compute_metrics(titles, golden)
        ndcg = compute_ndcg(titles, golden)
        rprec = compute_r_precision_at_k(titles, golden, k=3)

        precision_dict[model].append(prec)
        recall_dict[model].append(rec)
        f1_dict[model].append(f1)
        ndcg_dict[model].append(ndcg)
        r_precision_dict[model].append(rprec)

# --- Plotting ---
output_dir = "./graph_std_vs_lemma"
os.makedirs(output_dir, exist_ok=True)

plot_precision_recall_with_variance(avg_curves, f"{output_dir}/precision_recall_curve_media_interpolated.png", model1="tf_idf_std", model2="tf_idf_lemma")
plot_per_query_precision_recall(avg_curves, f"{output_dir}/subplot_per_query_precision_recall.png", model1="tf_idf_std", model2="tf_idf_lemma")
plot_querywise_precision_bar_chart(precision_dict, f"{output_dir}/precision_per_query_bar_chart.png", model1="tf_idf_std", model2="tf_idf_lemma")
plot_querywise_recall_bar_chart(recall_dict, f"{output_dir}/recall_per_query_bar_chart.png", model1="tf_idf_std", model2="tf_idf_lemma")
plot_querywise_ndcg_bar_chart(ndcg_dict, f"{output_dir}/ndcg_per_query_bar_chart.png", model1="tf_idf_std", model2="tf_idf_lemma")
plot_querywise_rprecision_bar_chart(r_precision_dict, f"{output_dir}/rprecision_diff_bar_chart.png", model1="tf_idf_std", model2="tf_idf_lemma")
plot_final_metric_summary_barplot(
    precision_dict,
    recall_dict,
    f1_dict,
    ndcg_dict,
    f"{output_dir}/final_metric_summary_barplot.png",
    model1="tf_idf_std",
    model2="tf_idf_lemma"
)

# --- Cleanup ---
for r in readers.values():
    r.close()