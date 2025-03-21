from elasticsearch import Elasticsearch, helpers
import json

# Connessione a Elasticsearch
es = Elasticsearch("http://localhost:9200", http_auth=("elastic", "NaePd5lzxh-rYkg1Aop3"))

# Nome dell'indice
index_name = "films"

# Caricamento dei dati
with open("films.json", encoding="utf-8") as file:
    movies = json.load(file)

# Preparazione dei dati per l'importazione in Elasticsearch
actions = [
    {
        "_op_type": "index",  # Operazione di indicizzazione
        "_index": index_name,
        "_source": movie
    }
    for movie in movies
]

# Importazione dei dati con bulk
helpers.bulk(es, actions)

print(f"✅ Caricati {len(movies)} film in Elasticsearch!")
# ✅ Caricati 250 film in Elasticsearch!
