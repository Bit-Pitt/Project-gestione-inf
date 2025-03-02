from whoosh.index import open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
import os, os.path

ix = open_dir("indexdir")

searcher = ix.searcher()

#print(list(searcher.lexicon("plot")))            #stampa tutti i token del field "context"
#print(list(searcher.lexicon("title")))               #stampa tutti i token del field "title"


parser = QueryParser("plot", schema=ix.schema)

query = parser.parse(u"godfather")
                
results = searcher.search(query)

print("\n\nRISULTATO DELLA QUERY:")
if len(results) == 0:
    print("Empty result!!")
else:
    for x in results:
        print(x)
        