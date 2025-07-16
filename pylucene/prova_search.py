import lucene
import os
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search.similarities import BM25Similarity, ClassicSimilarity
from org.apache.lucene.store import FSDirectory
from elasticsearch import Elasticsearch


# MIGLIORAMENTI POSSIBILI:
# - Aggiungere stemming e stopword removal per Whoosh (custom analyzer)

lucene.initVM()

################################## 1. CONNESSIONE A ELASTICSEARCH #####################################

# Connessione a Elasticsearch
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "NaePd5lzxh-rYkg1Aop3"))

if es.ping():
    print("✅ Connessione riuscita a Elasticsearch!")
else:
    print("❌ Errore di connessione a Elasticsearch!")


################################## 2. APERTURA INDICE WHOOSH ###########################################

index_dir = "./II_stdAnalyzer"  # path all'indice creato
directory = FSDirectory.open(Paths.get(index_dir))
reader = DirectoryReader.open(directory)
print(f"L'indice contiene {reader.numDocs()} documenti.")

# === Query Parser ===
analyzer = StandardAnalyzer()
query_str = "a guy who has no idea his life is being watched every moment"
parser = QueryParser("plot", analyzer)
query = parser.parse(query_str)


################################## 3. WHOOSH - BM25F ###################################################

searcher_bm25 = IndexSearcher(reader)
searcher_bm25.setSimilarity(BM25Similarity())
hits_bm25 = searcher_bm25.search(query, 10).scoreDocs

print("\n🔎 PyLucene - BM25")
if not hits_bm25:
    print("  ➤ Nessun risultato trovato.")
else:
    for hit in hits_bm25:
        doc = searcher_bm25.storedFields().document(hit.doc)
        print(f"  - {doc.get('title')} (Score: {hit.score:.4f})")


################################## 4. WHOOSH - TF-IDF ##################################################
searcher_tfidf = IndexSearcher(reader)
searcher_tfidf.setSimilarity(ClassicSimilarity())
hits_tfidf = searcher_tfidf.search(query, 10).scoreDocs

print("\n🔎 PyLucene - TF-IDF (ClassicSimilarity)")
if not hits_tfidf:
    print("  ➤ Nessun risultato trovato.")
else:
    for hit in hits_tfidf:
        doc = searcher_tfidf.storedFields().document(hit.doc)
        print(f"  - {doc.get('title')} (Score: {hit.score:.4f})")


################################## 5. ELASTICSEARCH - BM25 (default index) ##############################

es_query = {
    "size": 10,
    "query": {
        "match": {
            "plot": {
                "query": query_str,
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



