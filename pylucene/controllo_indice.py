import lucene
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from java.nio.file import Paths

# Percorso dell'indice
index_path = "/Users/nitroroot/Desktop/progettoIR/index"

# Avvia la JVM di Lucene
lucene.initVM()

# Apri l'indice
directory = FSDirectory.open(Paths.get(index_path))
if DirectoryReader.indexExists(directory):
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)

    print(f"📁 Numero di documenti indicizzati: {reader.numDocs()}")  # Conta i documenti
    
    # Leggiamo i primi 5 documenti indicizzati
    for i in range(min(5, reader.numDocs())):
        doc = reader.storedFields().document(i)  # ✅ Questo metodo funziona con PyLucene!
        print(f"🎬 Titolo: {doc.get('Title')}")
        print(f"📅 Anno: {doc.get('Year')}")
        print(f"🎭 Genere: {doc.get('Genre')}")
        print(f"🌍 Nazionalità: {doc.get('Country')}")
        print(f"📖 Trama: {doc.get('Plot')}\n")

    reader.close()
else:
    print("❌ L'indice non esiste o è vuoto.")
