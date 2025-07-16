import os
import shutil
import pandas as pd
import spacy
from lucene import initVM
from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths
from elasticsearch import Elasticsearch, helpers

# === Inizializzazione Lucene ===
initVM()
analyzer = StandardAnalyzer()
config_std = IndexWriterConfig(analyzer)
config_lemm = IndexWriterConfig(analyzer)

# === Lemmatizzazione spaCy ===
nlp = spacy.load("en_core_web_sm")
def lemmatize_text(text):
    if pd.isna(text):
        return ""
    doc = nlp(str(text))
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

# === Caricamento e pulizia CSV ===
df = pd.read_csv("./costruzione_dataset/films.csv")
print(f"📄 Film caricati: {len(df)}")
df = df.drop_duplicates(subset=["Title"])
print(f"🧹 Film dopo deduplicazione: {len(df)}")

# === Percorsi indici ===
std_path = "II_stdAnalyzer"
lemm_path = "II_lemmatized"
for path in [std_path, lemm_path]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# === Creazione Indice Lucene Standard ===
writer_std = IndexWriter(FSDirectory.open(Paths.get(std_path)), config_std)
for _, row in df.iterrows():
    doc = Document()
    doc.add(Field("title", row["Title"], TextField.TYPE_STORED))
    doc.add(Field("year", str(row["Year"]), TextField.TYPE_STORED))
    doc.add(Field("genre", row["Genre"], TextField.TYPE_STORED))
    doc.add(Field("country", row["Country"], TextField.TYPE_STORED))
    doc.add(Field("plot", row["Plot"], TextField.TYPE_STORED))
    writer_std.addDocument(doc)
writer_std.commit()
writer_std.close()
print("📁 Indice Lucene standard creato")

# === Creazione Indice Lucene Lemmatizzato ===
df_lemm = df.copy()
df_lemm["Title"] = df_lemm["Title"].apply(lemmatize_text)
df_lemm["Genre"] = df_lemm["Genre"].apply(lemmatize_text)
df_lemm["Country"] = df_lemm["Country"].apply(lemmatize_text)
df_lemm["Plot"] = df_lemm["Plot"].apply(lemmatize_text)

writer_lemm = IndexWriter(FSDirectory.open(Paths.get(lemm_path)), config_lemm)
for _, row in df_lemm.iterrows():
    doc = Document()
    doc.add(Field("title", row["Title"], TextField.TYPE_STORED))
    doc.add(Field("year", str(row["Year"]), TextField.TYPE_STORED))
    doc.add(Field("genre", row["Genre"], TextField.TYPE_STORED))
    doc.add(Field("country", row["Country"], TextField.TYPE_STORED))
    doc.add(Field("plot", row["Plot"], TextField.TYPE_STORED))
    writer_lemm.addDocument(doc)
writer_lemm.commit()
writer_lemm.close()
print("📁 Indice Lucene lemmatizzato creato")

# === Connessione Elasticsearch ===
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "NaePd5lzxh-rYkg1Aop3"))

bm25_index = "goldstandard_index"
vsm_index = "goldstandard_vsm"
lemm_index = "goldstandard_lemmatized"

def create_es_index(index_name, mapping):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=mapping)
    print(f"🔁 Indice Elasticsearch '{index_name}' creato")

bm25_mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "standard_analyzer": {"type": "standard"}
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

vsm_mapping = {
    "settings": {
        "similarity": {
            "my_similarity": {"type": "BM25", "b": 0.0, "k1": 1.0}
        },
        "analysis": {
            "analyzer": {
                "standard_analyzer": {"type": "standard"}
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

create_es_index(bm25_index, bm25_mapping)
create_es_index(vsm_index, vsm_mapping)
create_es_index(lemm_index, vsm_mapping)

def crea_actions(index_name, dataframe):
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
        for _, row in dataframe.iterrows()
    ]

helpers.bulk(es, crea_actions(bm25_index, df))
helpers.bulk(es, crea_actions(vsm_index, df))
helpers.bulk(es, crea_actions(lemm_index, df_lemm))
es.indices.refresh(index=bm25_index)
es.indices.refresh(index=vsm_index)
es.indices.refresh(index=lemm_index)
print("✅ Indicizzazione completata su: Lucene (standard e lemmatizzato) + Elasticsearch (BM25, VSM, Lemmatizzato)")
