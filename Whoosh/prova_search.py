from whoosh.index import open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, OrGroup  
import whoosh
import os, os.path

#QUI FAI UN RIASSUNTO DI QUELLO CHE HAI FATTO
#appunti: sembra migliore il BM25F per cui poi prova a migliorare il tf_idf() in qualche modo, idealmente puoi cambiare la fase di pre-processing
#       anche se questo non è direttamente collegato al tf_idf() e poi vedi cosa dice chatgpt per migliorare questo modello!


#Apertura dell'indice
directory = os.path.join("Whoosh", "II_stdAnalyzer")                #Soluzione portabile
ix = open_dir(directory)

#Creazione del SE con modello BM25F  (deafult)
searcher = ix.searcher()            

#print(list(searcher.lexicon("plot")))            #stampa tutti i token del field "context"
#print(list(searcher.lexicon("title")))               #stampa tutti i token del field "title"


#Creazione del parser della query con stdAnalyzer (per essere coerenti), e "plot" come campo default
parser = QueryParser("plot", schema=ix.schema,group=OrGroup) 
          


query = parser.parse(u" a guy who has no idea his life is being watched every moment ")           #Qui viene fatto il pre-processing tramite std analyzer  (coerenza con II)                    
results = searcher.search(query)                            

print("\n\nRISULTATO DELLA QUERY:   (modello BM25F), stdAnalyzer")      
if len(results) == 0:
    print("Empty result!!")
else:
    for hit in results:
        print(f"Documento: {hit["title"]}, Score: {hit.score}")


searcher = ix.searcher(weighting=whoosh.scoring.TF_IDF())                       #QUI STO USANDO IL VSM
query = parser.parse(u" reven* AND genre:thriller AND country:United States AND plot:*London* ") 
results = searcher.search(query)
print("\n\nRISULTATO DELLA QUERY:   (modello VSM), stdAnalyzer")
if len(results) == 0:
    print("Empty result!!")
else:
    for hit in results:
        print(f"Documento: {hit["title"]}, Score: {hit.score}")
