from whoosh.index import create_in
from whoosh.fields import *
import os, os.path
import pandas as pd
from elasticsearch import Elasticsearch

# Codice per la creazione dell'indice / degli indici.


#Creazione del primo II di Whoosh, utilizzo dello standard Analyzer

    
'''
schema = Schema(
    title=TEXT(stored=True),        #SOLO "TEXT" subirà pre-processing
    year=ID(stored=True),
    genre=KEYWORD(stored=True, commas=True), 
    country=KEYWORD(stored=True),  
    plot=TEXT(stored=False),                                                                                   
)
'''
'''
schema = Schema(
    title=TEXT(stored=True),        
    year=TEXT(stored=True),
    genre=TEXT(stored=True), 
    country=TEXT(stored=True),  
    plot=TEXT(stored=False),                                                                                   
)

directory = os.path.join("Whoosh", "II_stdAnalyzer")            #Soluzione portabile
if not os.path.exists(directory):
    os.makedirs(directory)     

ix = create_in(directory, schema)

writer = ix.writer()          


# Carica il dataset CSV
file = os.path.join("dataset-constr", "films.csv") 
df = pd.read_csv(file)



# Aggiungi i documenti all'indice specificando i vari campi
for indice, films in df.iterrows():
    writer.add_document(    title=films["Title"], 
                            year= str(films["Year"]),
                            genre=films["Genre"],
                            country=films["Country"],
                            plot=films["Plot"]
                               )
 
writer.commit()


print("Indicizzazione primo II completata!")

'''


#Creazione del'II del Gold Standard


# Connessione al servizio di  Elasticsearch 
# NECESSITA CHE IL SERVIZIO SIA ATTIVO 
# vai nella cartella "D:\Downloads\elasticsearch-9.0.3-windows-x86_64\elasticsearch-9.0.3\bin> .\elasticsearch.bat"
# estratta la cartella avvi il servizio ".\elasticsearch.bat"

from whoosh.index import create_in
from whoosh.fields import *
import os
import pandas as pd
from elasticsearch import Elasticsearch

# --- Connessione a Elasticsearch ---
es = Elasticsearch("http://172.26.112.1:9200")

if es.ping():
    print(" Connessione riuscita a Elasticsearch!")
else:
    print(" Errore di connessione a Elasticsearch!")

#es.indices.delete(index="goldstandard_index")
#es.indices.delete(index="goldstandard_vsm")

# --- Primo indice: BM25 (default) ---


bm25_index_name = "goldstandard_index"
bm25_mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "standard_analyzer": {
                    "type": "standard"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "title":   {"type": "text", "analyzer": "standard"},
            "year":    {"type": "text"},
            "genre":   {"type": "text"},
            "country": {"type": "text"},
            "plot":    {"type": "text", "analyzer": "standard"}
        }
    }
}
indicizza_bm = True
if not es.indices.exists(index=bm25_index_name):
    es.indices.create(index=bm25_index_name, body=bm25_mapping)
    print(f" Indice '{bm25_index_name}' creato.")
else:
    indicizza_bm = False
    print(f" L'indice '{bm25_index_name}' esiste già.")


# --- Secondo indice: TF-IDF/VSM ---
vsm_index_name = "goldstandard_vsm"
vsm_mapping = {
    "settings": {
        "similarity": {
            "my_similarity": {
                "type": "BM25",    # PARAMETRI IMPOSTATI PER RENDERE LA SIMILARITA' SIMILE AL VSM
                "b": 0.0,        # No length normalization
                "k1": 1.0        # TF scaling similar to classic TF-IDF
            }
        },
        "analysis": {
            "analyzer": {
                "standard_analyzer": {
                    "type": "standard"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "title":   {"type": "text", "similarity": "my_similarity", "analyzer": "standard"},
            "year":    {"type": "text", "similarity": "my_similarity"},
            "genre":   {"type": "text", "similarity": "my_similarity"},
            "country": {"type": "text", "similarity": "my_similarity"},
            "plot":    {"type": "text", "similarity": "my_similarity", "analyzer": "standard"}
        }
    }
}

indicizza_vsm=True
if not es.indices.exists(index=vsm_index_name):
    es.indices.create(index=vsm_index_name, body=vsm_mapping)
    print(f" Indice '{vsm_index_name}' creato con TF-IDF (classic similarity).")
else:
    indicizza_vsm=False
    print(f" L'indice '{vsm_index_name}' esiste già.")


# --- Carica dataset CSV ---
file = os.path.join("dataset-constr", "films.csv")
df = pd.read_csv(file)

# --- Indicizzazione su entrambi gli indici ---
for _, film in df.iterrows():
    doc = {
        "title": film["Title"],
        "year": str(film["Year"]),
        "genre": film["Genre"],
        "country": film["Country"],
        "plot": film["Plot"]
    }
    # indicizza su entrambi
    if indicizza_bm:
        es.index(index=bm25_index_name, document=doc)
    if indicizza_vsm:
        es.index(index=vsm_index_name, document=doc)

if not indicizza_bm:
    print("Indicizzazione non effettuate per bm (già presente)")
if not indicizza_bm:
    print("Indicizzazione non effettuate per vsm (già presente)")

print(" Indicizzazione completata su entrambi gli indici Elasticsearch!")
