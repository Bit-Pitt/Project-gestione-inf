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
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause

#🚀 Avvia la JVM
lucene.initVM()

# 📂 Percorso dell'indice
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, "index")

# 📖 Apri l'indice
directory = FSDirectory.open(Paths.get(index_path))
reader = DirectoryReader.open(directory)

# 🔍 Crea i motori di ricerca con VSM e BM25
searcher_vsm = IndexSearcher(reader)
searcher_vsm.setSimilarity(ClassicSimilarity())  # 🔥 VSM

searcher_bm25 = IndexSearcher(reader)
searcher_bm25.setSimilarity(BM25Similarity())  # 🔥 BM25

# 📝 Mappatura dei campi
field_mapping = {
    "title": "Title",
    "year": "Year",
    "genre": "Genre",
    "country": "Country",
    "plot": "Plot"
}

# 🔥 Scegli il modello di ricerca
model_choice = input("Vuoi usare 'vsm' o 'bm25'? ").strip().lower()
if model_choice == "vsm":
    searcher = searcher_vsm
elif model_choice == "bm25":
    searcher = searcher_bm25
else:
    print("❌ Modello non valido! Usa 'vsm' o 'bm25'")
    reader.close()
    exit()

# 📌 Chiedi i campi di ricerca
valid_fields = list(field_mapping.keys())
fields = input(f"In quali campi vuoi cercare? {valid_fields} (separati da virgola): ").strip().lower().split(",")
fields = [f.strip() for f in fields if f.strip() in valid_fields]

if not fields:
    print("❌ Nessun campo valido selezionato!")
    reader.close()
    exit()

# 🔍 Costruisci la query
analyzer = StandardAnalyzer()
bq = BooleanQuery.Builder()

for field in fields:
    query_string = input(f"Inserisci la query per '{field}': ").strip()
    query_parser = QueryParser(field_mapping[field], analyzer)
    query = query_parser.parse(query_string)
    bq.add(query, BooleanClause.Occur.MUST)

final_query = bq.build()
'''
# 🎯 Esegui la ricerca
top_docs = searcher.search(final_query, 10)

# 📊 Visualizza i risultati
print("\n🔍 Risultati della ricerca:\n")
retrieved_results = []

for hit in top_docs.scoreDocs:
    doc = reader.storedFields().document(hit.doc)
    title = doc.get("Title")
    retrieved_results.append(title)
    print(f"🎬 Titolo: {title} | ⭐ Score: {hit.score}")
print("-" * 50)

# 🔍 **Ottieni il golden standard da Google**
API_KEY = "AIzaSyBBRiMjlE0q7x8z9pUFSlt_7dAWa1gZaHQ"
CX_ID = "c1804e638497b4ba6"

def get_google_results(query):
    """Interroga l'API di Google e restituisce i primi 10 risultati"""
    query_with_film = query + "movie"
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
'''
print (final_query)
# 🔚 Chiudi il reader
reader.close()
