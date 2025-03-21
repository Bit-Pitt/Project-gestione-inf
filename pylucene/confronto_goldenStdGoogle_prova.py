import lucene
import os
import requests
from fuzzywuzzy import fuzz
import re
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, TopDocs
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity
from java.nio.file import Paths

# 🚀 Avvia la JVM
lucene.initVM()

# 📂 Percorso dell'indice
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, "index")

# 📖 Apri l'indice
directory = FSDirectory.open(Paths.get(index_path))
reader = DirectoryReader.open(directory)

# 🔍 Crea i due motori di ricerca
searcher_vsm = IndexSearcher(reader)
searcher_vsm.setSimilarity(ClassicSimilarity())  # 🔥 Forza VSM

searcher_bm25 = IndexSearcher(reader)
searcher_bm25.setSimilarity(BM25Similarity())  # 🔥 Usa BM25

# 📝 Mappatura dei campi
field_mapping = {
    "title": "Title",
    "year": "Year",
    "genre": "Genre",
    "country": "Country",
    "plot": "Plot"
}

# 🎯 Input utente
valid_fields = list(field_mapping.keys())
field = input(f"In che campo vuoi cercare? {valid_fields}: ").strip().lower()

if field not in field_mapping:
    print("❌ Campo non valido! Usa uno tra:", valid_fields)
    reader.close()
    exit()

query_string = input(f"Inserisci la query per '{field}': ").strip()

# 📝 Correzione ortografica della query (usando fuzzywuzzy)
def correct_spelling(query_string):
    """Corregge la query cercando la migliore corrispondenza tramite fuzzy matching"""
    analyzer = StandardAnalyzer()
    query_parser = QueryParser(field_mapping["title"], analyzer)  # Assumiamo che la ricerca sia sul titolo
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

query_string = correct_spelling(query_string)  # Applica la correzione alla query

# 🔍 Creazione dell'analizzatore e del parser
analyzer = StandardAnalyzer()
query_parser = QueryParser(field_mapping[field], analyzer)
query = query_parser.parse(query_string)

# 🎯 Esegui la ricerca con entrambi i modelli
top_docs_vsm = searcher_vsm.search(query, 10)
top_docs_bm25 = searcher_bm25.search(query, 10)

# 📊 Confronto dei risultati
print(f"\n🔍 Risultati per '{query_string}' nel campo '{field}':\n")

# Inizializzo vettori per usarli dopo nel GoldenStandard
vsm_results = []
bm25_results = []

print("📌 **VSM (Vector Space Model)**")
for hit in top_docs_vsm.scoreDocs:
    doc = reader.storedFields().document(hit.doc)
    title = doc.get("Title")
    vsm_results.append(title)
    print(f"🎬 Titolo: {doc.get('Title')} | ⭐ Score: {hit.score}")
print("-" * 50)

print("📌 **BM25 (Best Matching 25)**")
for hit in top_docs_bm25.scoreDocs:
    doc = reader.storedFields().document(hit.doc)
    title = doc.get("Title")
    bm25_results.append(title)
    print(f"🎬 Titolo: {doc.get('Title')} | ⭐ Score: {hit.score}")
print("-" * 50)

# 🔍 **Ottieni il golden standard da Google**
API_KEY = "AIzaSyBBRiMjlE0q7x8z9pUFSlt_7dAWa1gZaHQ"
CX_ID = "c1804e638497b4ba6"

def get_google_results(query):
    """Interroga l'API di Google e restituisce i primi 10 risultati"""
    query_with_film = f"{query} site:imdb.com OR site:rottentomatoes.com OR site:metacritic.com"
    url = f"https://www.googleapis.com/customsearch/v1?q={query_with_film}&key={API_KEY}&cx={CX_ID}"
    response = requests.get(url).json()

    # Controlliamo cosa ritorna Google
    if "items" not in response:
        print("Nessun risultato trovato da Google.")
        return []

    google_results = [item["title"].lower().strip() for item in response.get("items", [])]
    return google_results

def clean_title(title):
    """Rimuove dettagli extra dai titoli come anno tra parentesi, 'imdb', 'wikipedia', etc."""
    title = re.sub(r'\(.*\)', '', title)  # Rimuove tutto ciò che è tra parentesi
    title = re.sub(r' - .*', '', title)  # Rimuove '- imdb', '- wikipedia', etc.
    return title.strip()

golden_standard = get_google_results(query_string)
golden_standard_cleaned = [clean_title(title) for title in golden_standard]  # Pulisce i titoli del golden standard

print(f"\n🏅 **Golden Standard (Google Top Results) per '{query_string}':**")
for title in golden_standard_cleaned:
    print(f"✅ {title}")
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
precision_vsm, recall_vsm, f1_vsm = compute_metrics(vsm_results, golden_standard_cleaned)
precision_bm25, recall_bm25, f1_bm25 = compute_metrics(bm25_results, golden_standard_cleaned)

# 📊 **Stampa metriche**
print("\n📈 **Performance dei Modelli**")
print("📌 **VSM (Vector Space Model)**")
print(f"🎯 Precision: {precision_vsm:.3f}")
print(f"🎯 Recall: {recall_vsm:.3f}")
print(f"🎯 F1-score: {f1_vsm:.3f}")
print("-" * 50)

print("📌 **BM25 (Best Matching 25)**")
print(f"🎯 Precision: {precision_bm25:.3f}")
print(f"🎯 Recall: {recall_bm25:.3f}")
print(f"🎯 F1-score: {f1_bm25:.3f}")
print("-" * 50)

# 🔚 Chiudi il reader
reader.close()
