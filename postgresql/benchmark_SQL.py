import os
import psycopg2
import matplotlib.pyplot as plt
from utils import (
    compute_metrics,
    compute_ndcg,
    compute_r_precision_at_k,
    average_precision_recall_curve_interpolated,
    get_golden_standard
)
from bench_plotting import (
    plot_precision_recall_with_variance,
    plot_per_query_precision_recall,
    plot_querywise_precision_bar_chart,
    plot_querywise_recall_bar_chart,
    plot_querywise_ndcg_bar_chart,
    plot_querywise_rprecision_bar_chart,
    plot_final_metric_summary_barplot
)

DB_CONFIG = {
    'dbname': 'filmsearch',
    'host': 'localhost',
    'port': '5432',
}

OUTPUT_DIR = "./grafici"
os.makedirs(OUTPUT_DIR, exist_ok=True)

query_texts_elastic = [
    {"title": "truman", "genre": "drama", "plot": "a man who lives in a society"},
    {"plot": "curse* causes victims to kill* themselves", "genre": "horror"},
    {"genre": "sci-fi", "plot": "fantasy or science fiction or action movie that talks about aliens"},
    {"plot": "a person unknowingly lives inside a massive tv show"},
    {"plot": "young wizard school magic* battle dark* force* wand"},
    {"plot": "a boy discovers he has supernatural abilities and is drawn into a hidden world of ancient powers"},
    {"plot": "a group of unlikely heroes must destroy a powerful artifact while being hunted by ancient evil across vast mythical lands"},
    {"plot": "a family dealing with major loss", "genre": "drama"},
    {"plot": "epic journey"},
    {"plot": "man discovers secret organization conspiracy truth"},
    {"plot": "man with superpower"},
    {"plot": "detective investigates murders"},
    {"plot": "horror movie with kill*", "country": "america united states"},
    {"title": "god"},
    {"plot": "a small town is terrorized by a mysterious creature that appears only at night"}
]


query_texts = [
    {"title": "truman", "genre": "drama", "plot": "man | lives | reality"},
    {"plot": "curse | causes | victims | kill*", "genre": "horror"},
    {"genre": "science | action | drama | fantasy", "plot": "aliens"},
    {"plot": "person | lives | inside | tv | show"},
    {"plot": "young | wizard | school | magic* | battle | dark* | force* | wand"},
    {"plot": "boy | discovers | supernatural | abilities | hidden | world | ancient | powers"},
    {"plot": "heroes | destroy | artifact | hunted | evil | mythical | lands"},
    {"plot": "family | loss", "genre": "drama"},
    {"plot": "epic | journey"},
    {"plot": "man | discovers | secret | organization | conspiracy | truth"},
    {"plot": "man | superpower"},
    {"plot": "detective | investigates | murders"},
    {"plot": "horror | movie | kill*", "country": "america | united | states"},
    {"title": "god"},
    {"plot": "town | terrorized | creature | night"}
]

FIELD_TO_TSV = {
    "title": "tsv_title",
    "genre": "tsv_genre",
    "plot":  "tsv_plot",
    "country": "tsv_country",
    "year": "tsv_year"
}

avg_curves = {"BM25": [], "TF-IDF": []}
precision_dict = {"BM25": [], "TF-IDF": []}
recall_dict = {"BM25": [], "TF-IDF": []}
f1_dict = {"BM25": [], "TF-IDF": []}
ndcg_dict = {"BM25": [], "TF-IDF": []}
r_precision_dict = {"BM25": [], "TF-IDF": []}

def build_tsquery(query_str):
    return query_str if '|' in query_str else " | ".join(query_str.strip().split())

def build_sql(query_dict):
    where_clauses = []
    vsm_rank_clauses = []
    bm25_rank_clauses = []
    where_params = []
    vsm_rank_params = []
    bm25_rank_params = []

    for field, value in query_dict.items():
        if field not in FIELD_TO_TSV:
            continue
        col = FIELD_TO_TSV[field]
        tsquery = build_tsquery(value)

        where_clauses.append(f"{col} @@ to_tsquery('english', %s)")
        where_params.append(tsquery)

        vsm_rank_clauses.append(f"ts_rank({col}, to_tsquery('english', %s))")
        vsm_rank_params.append(tsquery)

        bm25_rank_clauses.append(f"ts_rank_cd({col}, to_tsquery('english', %s))")
        bm25_rank_params.append(tsquery)

    where_str = " AND ".join(where_clauses)
    vsm_rank_str = " + ".join(vsm_rank_clauses)
    bm25_rank_str = " + ".join(bm25_rank_clauses)

    return where_str, vsm_rank_str, bm25_rank_str, where_params, vsm_rank_params, bm25_rank_params




def fetch_results(cursor, where, rank_expr, params, alias):
    query = f"""
        SELECT title, ({rank_expr}) AS {alias}
        FROM films
        WHERE {where}
        ORDER BY {alias} DESC
        LIMIT 10;
    """
    cursor.execute(query, params)
    return cursor.fetchall()


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    fields = list(FIELD_TO_TSV.keys())
    golden_bm25 = get_golden_standard(fields, query_texts_elastic, "goldstandard_index")
    golden_vsm = get_golden_standard(fields, query_texts_elastic, "goldstandard_vsm")
    '''
    #==============================DEBUG=========================
    print(f"Golden BM25 length: {len(golden_bm25)}")
    print(f"Golden VSM length: {len(golden_vsm)}")
    print(f"First golden BM25: {golden_bm25[0] if golden_bm25 else 'EMPTY'}")
    #==============================DEBUG=========================
    '''
    for idx, q in enumerate(query_texts):
        print(f"Query {idx+1}: {q}")
        where, vsm_rank, bm25_rank, where_params, vsm_params, bm25_params = build_sql(q)
        res_vsm = fetch_results(cursor, where, vsm_rank, where_params + vsm_params, "rank_vsm")
        res_bm25 = fetch_results(cursor, where, bm25_rank, where_params + bm25_params, "rank_bm25")

        '''
        #==============================DEBUG=========================
        print(f"to_tsquery results: {len(res_vsm)} items")
        print(f"to_tsquery_cd results: {len(res_bm25)} items")
        #==============================DEBUG=========================
        '''
        retrieved_vsm = [r[0] for r in res_vsm]
        retrieved_bm25 = [r[0] for r in res_bm25]

        _, avg_prec_bm25 = average_precision_recall_curve_interpolated(retrieved_bm25, golden_bm25[idx], golden_vsm[idx])
        _, avg_prec_vsm = average_precision_recall_curve_interpolated(retrieved_vsm, golden_bm25[idx], golden_vsm[idx])
        avg_curves["BM25"].append(avg_prec_bm25)
        avg_curves["TF-IDF"].append(avg_prec_vsm)

        for model, retrieved in [("BM25", retrieved_bm25), ("TF-IDF", retrieved_vsm)]:
            p1, r1, f1_1 = compute_metrics(retrieved, golden_bm25[idx])
            p2, r2, f1_2 = compute_metrics(retrieved, golden_vsm[idx])
            ndcg1 = compute_ndcg(retrieved, golden_bm25[idx])
            ndcg2 = compute_ndcg(retrieved, golden_vsm[idx])
            rprec1 = compute_r_precision_at_k(retrieved, golden_bm25[idx], 3)
            rprec2 = compute_r_precision_at_k(retrieved, golden_vsm[idx], 3)
            '''
            #==============================DEBUG=========================
            print(f"Precision BM25: {p1}, Recall BM25: {r1}")
            print(f"Precision VSM: {p2}, Recall VSM: {r2}")
            #==============================DEBUG=========================
            '''
            precision_dict[model].append((p1 + p2) / 2)
            recall_dict[model].append((r1 + r2) / 2)
            f1_dict[model].append((f1_1 + f1_2) / 2)
            ndcg_dict[model].append((ndcg1 + ndcg2) / 2)
            r_precision_dict[model].append((rprec1 + rprec2) / 2)

    # --- Plotting ---
    plot_precision_recall_with_variance(avg_curves, os.path.join(OUTPUT_DIR, "precision_recall_curve_media_interpolated.png"))
    plot_per_query_precision_recall(avg_curves, os.path.join(OUTPUT_DIR, "subplot_per_query_precision_recall.png"))
    plot_querywise_precision_bar_chart(precision_dict, os.path.join(OUTPUT_DIR, "precision_per_query_bar_chart.png"))
    plot_querywise_recall_bar_chart(recall_dict, os.path.join(OUTPUT_DIR, "recall_per_query_bar_chart.png"))
    plot_querywise_ndcg_bar_chart(ndcg_dict, os.path.join(OUTPUT_DIR, "ndcg_per_query_bar_chart.png"))
    plot_querywise_rprecision_bar_chart(r_precision_dict, os.path.join(OUTPUT_DIR, "rprecision_diff_bar_chart.png"))
    plot_final_metric_summary_barplot(
        precision_dict,
        recall_dict,
        f1_dict,
        ndcg_dict,
        os.path.join(OUTPUT_DIR, "final_metric_summary_barplot.png")
    )

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
