import psycopg2
from funzioniSQL import get_golden_standard, compute_metrics, compute_ndcg, plot_average_precision_recall_curve_interpolated

# === Campi usati da Elasticsearch per golden standard
fields_for_golden = ["title", "genre", "plot"]

# === Lista di 10 query per PostgreSQL e Elasticsearch
query_texts = [
    {"title": "family", "genre": "drama", "plot": "loss"},
    {"title": "war", "genre": "action", "plot": "hero"},
    {"title": "space", "genre": "sci-fi", "plot": "mission"},
    {"title": "love", "genre": "romance", "plot": "betrayal"},
    {"title": "science", "genre": "drama", "plot": "experiment"},
    {"title": "crime", "genre": "thriller", "plot": "detective"},
    {"title": "magic", "genre": "fantasy", "plot": "world"},
    {"title": "future", "genre": "sci-fi", "plot": "dystopia"},
    {"title": "journey", "genre": "adventure", "plot": "discovery"},
    {"title": "revenge", "genre": "drama", "plot": "justice"},
]

# === Connessione PostgreSQL
conn = psycopg2.connect(
    dbname='filmsearch',
    #user='admin',
    host='localhost',
    port='5432'
)
cursor = conn.cursor()

# === Risultati complessivi
results_dict = {}
all_golden_bm25 = []
all_golden_vsm = []

for query_dict in query_texts:
    tsquery = " & ".join(query_dict.values())
    print(f"\n🔍 Query: {tsquery}")

    # === Golden standard per questa query
    golden_bm25 = get_golden_standard(fields_for_golden, query_dict, "goldstandard_index")
    golden_vsm = get_golden_standard(fields_for_golden, query_dict, "goldstandard_vsm")
    golden = list(set(golden_bm25 + golden_vsm))
    all_golden_bm25.extend(golden_bm25)
    all_golden_vsm.extend(golden_vsm)

    # --- VSM-like
    cursor.execute("""
        SELECT title
        FROM films
        WHERE tsv @@ to_tsquery('english', %s)
        ORDER BY ts_rank(tsv, to_tsquery('english', %s)) DESC
        LIMIT 10;
    """, (tsquery, tsquery))
    retrieved_vsm = [row[0] for row in cursor.fetchall()]
    print("VSM-like:")
    compute_metrics(retrieved_vsm, golden)
    compute_ndcg(retrieved_vsm, golden)
    results_dict[f"VSM | {tsquery}"] = retrieved_vsm

    # --- BM25-like
    cursor.execute("""
        SELECT title
        FROM films
        WHERE tsv @@ to_tsquery('english', %s)
        ORDER BY ts_rank_cd(tsv, to_tsquery('english', %s)) DESC
        LIMIT 10;
    """, (tsquery, tsquery))
    retrieved_bm25 = [row[0] for row in cursor.fetchall()]
    print("BM25-like:")
    compute_metrics(retrieved_bm25, golden)
    compute_ndcg(retrieved_bm25, golden)
    results_dict[f"BM25 | {tsquery}"] = retrieved_bm25

# === Grafico finale
plot_average_precision_recall_curve_interpolated(results_dict, all_golden_bm25, all_golden_vsm)
