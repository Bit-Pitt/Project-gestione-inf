from lucene import initVM
from org.apache.lucene.document import Document, Field
from org.apache.lucene.document import TextField
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths
import pandas as pd

# Inizializza la JVM
initVM()

# Carica il dataset CSV
df = pd.read_csv("films.csv")

# Crea l'analyzer
analyzer = StandardAnalyzer()

# Config writer
config = IndexWriterConfig(analyzer)

# Directory indice (modifica se vuoi salvarlo altrove)
index_dir = FSDirectory.open(Paths.get("../II_stdAnalyzer"))  # nuova cartella

# Crea l'IndexWriter
writer = IndexWriter(index_dir, config)

# Aggiungi i documenti all'indice (nomi campi in minuscolo!)
for _, row in df.iterrows():
    doc = Document()
    doc.add(Field("title", row["Title"], TextField.TYPE_STORED))
    doc.add(Field("year", str(row["Year"]), TextField.TYPE_STORED))
    doc.add(Field("genre", row["Genre"], TextField.TYPE_STORED))
    doc.add(Field("country", row["Country"], TextField.TYPE_STORED))
    doc.add(Field("plot", row["Plot"], TextField.TYPE_STORED))
    writer.addDocument(doc)

# Commit e chiusura
writer.commit()
writer.close()

print("✅ Indicizzazione completata con nomi campo minuscoli!")
