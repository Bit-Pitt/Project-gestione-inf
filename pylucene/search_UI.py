import lucene
import os
import sys
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity
from java.nio.file import Paths

# Inizializza la JVM di Lucene
lucene.initVM()

# Percorso dell'indice
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, "index")

directory = FSDirectory.open(Paths.get(index_path))
reader = DirectoryReader.open(directory)

# Crea i motori di ricerca per VSM e BM25
searcher_vsm = IndexSearcher(reader)
searcher_vsm.setSimilarity(ClassicSimilarity())

searcher_bm25 = IndexSearcher(reader)
searcher_bm25.setSimilarity(BM25Similarity())

# Mappatura dei campi
field_mapping = {
    "title": "Title",
    "year": "Year",
    "genre": "Genre",
    "country": "Country",
    "plot": "Plot"
}

# Legge i parametri dalla UI
if len(sys.argv) < 3:
    print("Errore: Parametri insufficienti.")
    sys.exit(1)

model_choice = sys.argv[1].strip().lower()
query_string = sys.argv[2].strip()

# Seleziona il motore di ricerca
if model_choice == "vsm":
    searcher = searcher_vsm
elif model_choice == "bm25":
    searcher = searcher_bm25
else:
    print("Errore: Modello non valido.")
    sys.exit(1)

# Parsing della query
query_parts = query_string.split(", ")
fields = {}
for part in query_parts:
    if ":" in part:
        key, value = part.split(":", 1)
        if key in field_mapping:
            fields[key] = value.strip()

if not fields:
    print("Errore: Nessun campo valido nella query.")
    sys.exit(1)

# Creazione della query
analyzer = StandardAnalyzer()
bq = BooleanQuery.Builder()

for field, value in fields.items():
    query_parser = QueryParser(field_mapping[field], analyzer)
    query = query_parser.parse(value)
    bq.add(query, BooleanClause.Occur.MUST)

final_query = bq.build()

# Esegui la ricerca
num_results = 10
top_docs = searcher.search(final_query, num_results)
retrieved_results = []

for hit in top_docs.scoreDocs:
    doc = reader.storedFields().document(hit.doc)
    title = doc.get("Title")
    score = hit.score
    retrieved_results.append((title, score))

# Funzione per calcolare il Golden Standard

def get_golden_standard(searcher, query, top_n=50, score_threshold=5.0):
    top_docs = searcher.search(query, top_n)
    golden_standard = []
    for hit in top_docs.scoreDocs:
        if hit.score >= score_threshold:
            doc = reader.storedFields().document(hit.doc)
            golden_standard.append(doc.get("Title").lower().strip())
    return golden_standard

# Ottieni il Golden Standard
golden_standard = get_golden_standard(searcher, final_query)

# Funzione per calcolare precision, recall e F1-score
def compute_metrics(retrieved, golden):
    retrieved_set = set([r[0].lower().strip() for r in retrieved])
    golden_set = set(golden)
    true_positives = len(retrieved_set & golden_set)
    retrieved_count = len(retrieved_set)
    golden_count = len(golden_set)
    precision = true_positives / retrieved_count if retrieved_count > 0 else 0
    recall = true_positives / golden_count if golden_count > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Calcola le metriche
precision, recall, f1 = compute_metrics(retrieved_results, golden_standard)

# Stampa i risultati in formato compatibile con la UI
output = []
output.append("\n🔍 Risultati della ricerca:\n")
for title, score in retrieved_results:
    output.append(f"🎬 {title} | ⭐ Score: {score:.2f}")
output.append("-" * 50)
output.append("\n🏅 Golden Standard:")
for title in golden_standard:
    output.append(f"✅ {title}")
output.append("-" * 50)
output.append("\n📈 Performance del Modello")
output.append(f"🎯 Precision: {precision:.3f}")
output.append(f"🎯 Recall: {recall:.3f}")
output.append(f"🎯 F1-score: {f1:.3f}")
output.append("-" * 50)

# Stampa l'output in modo che Tkinter possa leggerlo
print("\n".join(output))

# Chiudi il reader
reader.close()