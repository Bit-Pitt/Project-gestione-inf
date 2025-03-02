from whoosh.index import open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, OrGroup  
import whoosh
import os, os.path



directory = os.path.join("Whoosh", "II_stdAnalyzer")                #Soluzione portabile
ix = open_dir(directory)

searcher = ix.searcher()            #Whoosh utilizza di default il modello di BM25F 

#print(list(searcher.lexicon("plot")))            #stampa tutti i token del field "context"
#print(list(searcher.lexicon("title")))               #stampa tutti i token del field "title"


parser = QueryParser("plot", schema=ix.schema,group=OrGroup)          #in questo modo il field "plot" sarà di default  


query = parser.parse(u"godfather  mom mafia")           #Qui viene fatto il pre-processing tramite std analyzer  (coerenza con II)                    
results = searcher.search(query)

print("\n\nRISULTATO DELLA QUERY:   (modello BM25F), stdAnalyzer")
if len(results) == 0:
    print("Empty result!!")
else:
    for hit in results:
        print(f"Documento: {hit}, Score: {hit.score}")


searcher = ix.searcher(weighting=whoosh.scoring.TF_IDF())
query = parser.parse(u"godfather  mom  mafia") 
results = searcher.search(query)
print("\n\nRISULTATO DELLA QUERY:   (modello VSM), stdAnalyzer")
if len(results) == 0:
    print("Empty result!!")
else:
    for hit in results:
        print(f"Documento: {hit}, Score: {hit.score}")
