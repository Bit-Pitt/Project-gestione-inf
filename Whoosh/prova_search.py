from whoosh.index import open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, OrGroup
import whoosh
import os
from elasticsearch import Elasticsearch


# MIGLIORAMENTI POSSIBILI:
# - Aggiungere stemming e stopword removal per Whoosh (custom analyzer)



################################## 1. CONNESSIONE A ELASTICSEARCH #####################################

es = Elasticsearch("http://172.26.112.1:9200")

if es.ping():
    print("✅ Connessione riuscita a Elasticsearch!")
else:
    raise RuntimeError("❌ Errore di connessione a Elasticsearch. Assicurati che sia avviato.")


################################## 2. APERTURA INDICE WHOOSH ###########################################

whoosh_dir = os.path.join("Whoosh", "II_stdAnalyzer")
ix = open_dir(whoosh_dir)

# Crea un parser coerente con lo standard analyzer
parser = QueryParser("plot", schema=ix.schema, group=OrGroup)

# --- Test query ---
query_text = u"a guy who has no idea his life is being watched every moment"
parsed_query = parser.parse(query_text)


################################## 3. WHOOSH - BM25F ###################################################

searcher_bm25 = ix.searcher()
results_bm25 = searcher_bm25.search(parsed_query)

print("\n🔎 WHOOSH - BM25F (default)")
if not results_bm25:
    print("  ➤ Nessun risultato trovato.")
else:
    for hit in results_bm25:
        print(f"  - {hit['title']} (Score: {hit.score:.4f})")


################################## 4. WHOOSH - TF-IDF ##################################################

searcher_tfidf = ix.searcher(weighting=whoosh.scoring.TF_IDF())
results_tfidf = searcher_tfidf.search(parsed_query)

print("\n🔎 WHOOSH - TF-IDF (VSM)")
if not results_tfidf:
    print("  ➤ Nessun risultato trovato.")
else:
    for hit in results_tfidf:
        print(f"  - {hit['title']} (Score: {hit.score:.4f})")


################################## 5. ELASTICSEARCH - BM25 (default index) ##############################

es_query = {
    "size": 10,
    "query": {
        "match": {
            "plot": {
                "query": query_text,
                "operator": "or"
            }
        }
    }
}

res_es = es.search(index="goldstandard_index", body=es_query)

print("\n🔎 ELASTICSEARCH - BM25 (Indice: goldstandard_index)")
hits = res_es["hits"]["hits"]
if not hits:
    print("  ➤ Nessun risultato trovato.")
else:
    for hit in hits:
        print(f"  - {hit['_source']['title']} (Score: {hit['_score']:.4f})")


################################## 6. ELASTICSEARCH - TF-IDF-like #######################################

# Assicurati di avere già creato questo indice con BM25 modificato
vsm_index = "goldstandard_vsm"
if es.indices.exists(index=vsm_index):
    res_vsm = es.search(index=vsm_index, body=es_query)
    print(f"\n🔎 ELASTICSEARCH - TF-IDF-like (Indice: {vsm_index})")
    vsm_hits = res_vsm["hits"]["hits"]
    if not vsm_hits:
        print("  ➤ Nessun risultato trovato.")
    else:
        for hit in vsm_hits:
            print(f"  - {hit['_source']['title']} (Score: {hit['_score']:.4f})")
else:
    print(f"\n⚠️ Indice '{vsm_index}' non trovato. Salta confronto TF-IDF-like in Elasticsearch.")



