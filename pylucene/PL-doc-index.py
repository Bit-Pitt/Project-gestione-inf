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

# Crea la configurazione dell'index writer
config = IndexWriterConfig(analyzer)

# Definisci la directory dove salvare l'indice
index_dir = FSDirectory.open(Paths.get("/Users/nitroroot/Desktop/progettoIR/index"))

# Crea l'IndexWriter
writer = IndexWriter(index_dir, config)

# Aggiungi i documenti all'indice
for index, row in df.iterrows():
    doc = Document()
    doc.add(Field("Title", row["Title"], TextField.TYPE_STORED))
    doc.add(Field("Year", str(row["Year"]), TextField.TYPE_STORED))
    doc.add(Field("Genre", row["Genre"], TextField.TYPE_STORED))
    doc.add(Field("Country", row["Country"], TextField.TYPE_STORED))
    doc.add(Field("Plot", row["Plot"], TextField.TYPE_STORED))
    
    writer.addDocument(doc)

# Commit e chiusura
writer.commit()
writer.close()

print("Indicizzazione completata!")
