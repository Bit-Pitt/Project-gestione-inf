import lucene
import os
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity
from org.apache.lucene.search import BooleanQuery
from java.nio.file import Paths

# 🚀 Avvia la JVM
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

# 🔍 **Golden Standard migliorato**
def get_golden_standard(searcher, query, top_n=50, score_threshold=5.0):
    top_docs = searcher.search(query, top_n)
    golden_standard = []
    for hit in top_docs.scoreDocs:
        if hit.score >= score_threshold:
            doc = reader.storedFields().document(hit.doc)
            title = doc.get("Title").lower().strip()
            golden_standard.append(title)
    return golden_standard

# 🏅 **Ottieni il Golden Standard**
golden_standard = get_golden_standard(searcher, final_query)
print("\n🏅 **Golden Standard:**")
for title in golden_standard:
    print(f"✅ {title}")
print("-" * 50)

# 📊 **Calcolo Precision, Recall e F1-score**
def compute_metrics(retrieved, golden):
    retrieved_set = set([r.lower().strip() for r in retrieved])
    golden_set = set([g.lower().strip() for g in golden])
    true_positives = len(retrieved_set & golden_set)
    retrieved_count = len(retrieved_set)
    golden_count = len(golden_set)
    precision = true_positives / retrieved_count if retrieved_count > 0 else 0
    recall = true_positives / golden_count if golden_count > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# 🔢 **Calcola metriche**
precision, recall, f1 = compute_metrics(retrieved_results, golden_standard)

# 📊 **Stampa metriche**
print("\n📈 **Performance del Modello**")
print(f"🎯 Precision: {precision:.3f}")
print(f"🎯 Recall: {recall:.3f}")
print(f"🎯 F1-score: {f1:.3f}")
print("-" * 50)

# 🔚 Chiudi il reader
reader.close()
