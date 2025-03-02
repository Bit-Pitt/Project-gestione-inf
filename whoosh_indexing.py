from whoosh.index import create_in
from whoosh.fields import *
import os, os.path

schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)        #SOLO "text" subirà pre-processing
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
ix = create_in("indexdir", schema)

writer = ix.writer()          

    # whoosh usa l'analyzer inglese di default

writer.add_document(title=u"First document", path=u"/a", content=u"This is the first documents we've added!")
writer.add_document(title=u"Second document", path=u"/b", content=u"The second one is even more interesting!")
writer.commit()