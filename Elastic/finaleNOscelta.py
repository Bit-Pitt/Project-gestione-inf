import lucene
import os
import requests
from fuzzywuzzy import fuzz
from elasticsearch import Elasticsearch
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity
from java.nio.file import Paths

lucene.initVM()

# 📂 Percorso dell'indice PyLucene
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, "index")

# 📖 Apri l'indice
directory = FSDirectory.open(Paths.get(index_path))
reader = DirectoryReader.open(directory)

# 🔍 Crea i due motori di ricerca PyLucene
searcher_vsm = IndexSearcher(reader)
searcher_vsm.setSimilarity(ClassicSimilarity())  # VSM
searcher_bm25 = IndexSearcher(reader)
searcher_bm25.setSimilarity(BM25Similarity())   # BM25

# 📝 Mappatura dei campi
field_mapping_NOGS = {
    "title": "Title",
    "year": "Year",
    "genre": "Genre",
    "country": "Country",
    "plot": "Plot"
}

# Mappatura dei campi per il Golden Standard
field_mapping_GS = {
    "title": "title",
    "year": "year",
    "genre": "genre",
    "country": "country",
    "plot": "plot"
}

# 🎯 Input utente
model_choice = input("Quale modello di ricerca vuoi utilizzare? [vsm/bm25]: ").strip().lower()
if model_choice == "vsm":
    searcher = searcher_vsm
elif model_choice == "bm25":
    searcher = searcher_bm25
else:
    print("❌ Modello di ricerca non valido! Usa 'vsm' o 'bm25'")
    reader.close()
    exit()

valid_fields = list(field_mapping_NOGS.keys())
field = input(f"In che campo vuoi cercare? {valid_fields}: ").strip().lower()
if field not in field_mapping_NOGS:
    print("❌ Campo non valido! Usa uno tra:", valid_fields)
    reader.close()
    exit()
query_string = input(f"Inserisci la query per '{field}': ").strip()

'''
# 📝 Correzione ortografica della query (usando fuzzywuzzy)
def correct_spelling(query_string):
    """Corregge la query cercando la migliore corrispondenza tramite fuzzy matching"""
    analyzer = StandardAnalyzer()
    query_parser = QueryParser(field_mapping_NOGS["title"], analyzer)  # Assumiamo che la ricerca sia sul titolo
    query = query_parser.parse(query_string)  # Crea la query Lucene

    # Esegui la ricerca per ottenere i primi 10 risultati
    top_docs = searcher_vsm.search(query, 10)

    # Trova la migliore corrispondenza tra i risultati
    best_match = query_string  # Default: la query originale
    best_score = 0
    for hit in top_docs.scoreDocs:
        doc = reader.storedFields().document(hit.doc)
        title = doc.get("Title")
        score = fuzz.ratio(query_string.lower(), title.lower())  # Confronto fuzzy
        if score > best_score:
            best_score = score
            best_match = title

    return best_match if best_score >= 80 else query_string  # Restituisce la migliore corrispondenza se il punteggio è sufficiente
'''
''' query_string = correct_spelling(query_string)  # Applica la correzione alla query '''

# 🔍 Creazione dell'analizzatore e del parser
analyzer = StandardAnalyzer()
query_parser = QueryParser(field_mapping_NOGS[field], analyzer)
query = query_parser.parse(query_string)

# 🎯 Esegui la ricerca con PyLucene
top_docs = searcher.search(query, 10)

# Inizializzo vettori per usarli dopo nel GoldenStandard
retrieved_docs = []

print("📌 **BM25 (Best Matching 25)**")
for hit in top_docs.scoreDocs:
    doc = reader.storedFields().document(hit.doc)
    title = doc.get("Title")
    retrieved_docs.append(title)
    print(f"🎬 Titolo: {doc.get('Title')} | ⭐ Score: {hit.score}")
print("-" * 50)

# Mi connetto ad ElasticSearch per GoldenStandard
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "NaePd5lzxh-rYkg1Aop3"))
if not es.ping():
    print("❌ Errore di connessione ad Elasticsearch!")
    reader.close()
    exit()

def getGoldenStandard(field, query_string):
    try:
        search_body = {
        "query": {
            "match": {
                field_mapping_GS[field]: query_string
            }
        },
        "_source": [field_mapping_GS["title"]], # La risposta contiene solo il titolo
        "size": 10  # Limitato ai primi 10 risultati
        }
        response = es.search(index="films", body=search_body)
        GS = [hit['_source']['title'].lower().strip() for hit in response['hits']['hits']]
        return GS
    except Exception as e:
        print(f"Errore durante la ricerca del Golden Standard: {e}")
        return []

golden_standard = getGoldenStandard(field, query_string)

# Ritorno i risultati del Golden Standard
print(f"\n🔍 **Golden Standard da Elasticsearch per {query_string}**")
for title in golden_standard:
    print(f"🎬 Titolo: {title}")
print("-" * 50)

# 📊 **Calcolo Precision, Recall e F1-score**
def compute_metrics(retrieved, golden):
    """Calcola Precision, Recall e F1-score con fuzzy matching"""
    true_positives = 0
    for retrieved_title in retrieved:
        for golden_title in golden:
            if fuzz.ratio(retrieved_title.lower(), golden_title.lower()) >= 80:  # Se la similarità è alta
                true_positives += 1
                break  # Evita di contare lo stesso retrieved più volte

    retrieved_count = len(retrieved)
    golden_count = len(golden)

    precision = true_positives / retrieved_count if retrieved_count > 0 else 0
    recall = true_positives / golden_count if golden_count > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# 🔢 **Calcola metriche**
precision, recall, f1 = compute_metrics(retrieved_docs, golden_standard)

# 📊 **Stampa metriche**
print("\n📈 **Performance dei Modelli**")
print("📌 **VSM (Vector Space Model)**")
print(f"🎯 Precision: {precision:.3f}")
print(f"🎯 Recall: {recall:.3f}")
print(f"🎯 F1-score: {f1:.3f}")
print("-" * 50)
