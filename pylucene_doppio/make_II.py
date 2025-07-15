from elasticsearch import Elasticsearch, helpers
import os
import pandas as pd

# Connessione a Elasticsearch
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "NaePd5lzxh-rYkg1Aop3"))

if es.ping():
    print("✅ Connessione riuscita a Elasticsearch!")
else:
    print("❌ Errore di connessione a Elasticsearch!")

# === Indice BM25 ===
bm25_index_name = "goldstandard_index"
bm25_mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "standard_analyzer": { "type": "standard" }
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

# === Indice VSM ===
vsm_index_name = "goldstandard_vsm"
vsm_mapping = {
    "settings": {
        "similarity": {
            "my_similarity": {
                "type": "BM25",
                "b": 0.0,
                "k1": 1.0
            }
        },
        "analysis": {
            "analyzer": {
                "standard_analyzer": { "type": "standard" }
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

# === Elimina e ricrea indici ===
def ricrea_indice(nome, mapping):
    if es.indices.exists(index=nome):
        es.indices.delete(index=nome)
        print(f"🗑️  Indice '{nome}' eliminato.")
    es.indices.create(index=nome, body=mapping)
    print(f"✅ Indice '{nome}' creato.")

ricrea_indice(bm25_index_name, bm25_mapping)
ricrea_indice(vsm_index_name, vsm_mapping)

# === Caricamento CSV ===
file = "/Users/nitroroot/Desktop/Project-gestione-inf/pylucene_doppio/costruzione_dataset/films.csv"
df = pd.read_csv(file)
print(f"📄 Caricati {len(df)} film dal CSV.")

# === Costruzione lista per helpers.bulk() ===
def crea_actions(index_name):
    return [
        {
            "_op_type": "index",
            "_index": index_name,
            "_source": {
                "title": row["Title"],
                "year": str(row["Year"]),
                "genre": row["Genre"],
                "country": row["Country"],
                "plot": row["Plot"]
            }
        }
        for _, row in df.iterrows()
    ]

# === Indicizzazione con bulk ===
helpers.bulk(es, crea_actions(bm25_index_name))
helpers.bulk(es, crea_actions(vsm_index_name))

# === Refresh per rendere subito visibili i documenti ===
es.indices.refresh(index=bm25_index_name)
es.indices.refresh(index=vsm_index_name)

print("✅ Indicizzazione completata su entrambi gli indici!")
