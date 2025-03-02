import lucene
import os
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
index_path = os.path.join(current_dir, "index") # con index cartella dell'indice

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

# 🔍 Creazione dell'analizzatore e del parser
analyzer = StandardAnalyzer()
query_parser = QueryParser(field_mapping[field], analyzer)
query = query_parser.parse(query_string)

# 🎯 Esegui la ricerca con entrambi i modelli
top_docs_vsm = searcher_vsm.search(query, 10)
top_docs_bm25 = searcher_bm25.search(query, 10)

# 📊 Confronto dei risultati
print(f"\n🔍 Risultati per '{query_string}' nel campo '{field}':\n")

print("📌 **VSM (Vector Space Model)**")
for hit in top_docs_vsm.scoreDocs:
    doc = reader.storedFields().document(hit.doc)
    print(f"🎬 Titolo: {doc.get('Title')} | ⭐ Score: {hit.score}")
print("-" * 50)

print("📌 **BM25 (Best Matching 25)**")
for hit in top_docs_bm25.scoreDocs:
    doc = reader.storedFields().document(hit.doc)
    print(f"🎬 Titolo: {doc.get('Title')} | ⭐ Score: {hit.score}")
print("-" * 50)

# 🔚 Chiudi il reader
reader.close()
