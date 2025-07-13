import lucene
import os
import requests
from fuzzywuzzy import fuzz
from elasticsearch import Elasticsearch
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import ClassicSimilarity, BM25Similarity
from java.nio.file import Paths

lucene.initVM()

# 📂 Percorso dell'indice PyLucene
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, "index")

# 📖 Apri l'indice
directory = FSDirectory.open(Paths.get(index_path))
reader = DirectoryReader.open(directory)

# 🔍 Crea i due motori di ricerca PyLucene
searcher_vsm = IndexSearcher(reader)
searcher_vsm.setSimilarity(ClassicSimilarity())  # VSM
searcher_bm25 = IndexSearcher(reader)
searcher_bm25.setSimilarity(BM25Similarity())   # BM25

# 📝 Mappatura dei campi
field_mapping_NOGS = {
    "title": "Title",
    "year": "Year",
    "genre": "Genre",
    "country": "Country",
    "plot": "Plot"
}

# Mappatura dei campi per Elasticsearch (Golden Standard)
field_mapping_GS = {
    "title": "title",
    "year": "year",
    "genre": "genre",
    "country": "country",
    "plot": "plot"
}

# 🔍 Selezione multipla di campi
valid_fields = list(field_mapping_NOGS.keys())
fields = input(f"In quali campi vuoi cercare? {valid_fields} (separati da virgola): ").strip().lower().split(",")
fields = [f.strip() for f in fields if f.strip() in valid_fields]

if not fields:
    print("❌ Nessun campo valido selezionato!")
    reader.close()
    exit()

# 🔍 L'utente inserisce le query UNA SOLA VOLTA per tutti i campi
queries = {}
for field in fields:
    queries[field] = input(f"Inserisci la query per '{field}': ").strip()

# 🔍 Costruisci la query in PyLucene
analyzer = StandardAnalyzer()
bq = BooleanQuery.Builder()

for field, query_string in queries.items():
    query_parser = QueryParser(field_mapping_NOGS[field], analyzer)
    query = query_parser.parse(query_string)
    bq.add(query, BooleanClause.Occur.MUST)

final_query = bq.build()

# 🎯 Esegui la ricerca con PyLucene
result_by_model = {}

for model_name, searcher in [("VSM", searcher_vsm), ("BM25", searcher_bm25)]:
    top_docs = searcher.search(final_query, 10)
    result_by_model[model_name] = []

    print(f"**Risultati {model_name}**")
    for hit in top_docs.scoreDocs:
        doc = reader.storedFields().document(hit.doc)
        title = doc.get("Title")
        result_by_model[model_name].append(title)
        print(f"🎬 Titolo: {doc.get('Title')} | ⭐ Score: {hit.score}")
    print("-" * 50)
top_docs = searcher.search(final_query, 10)

# Mi connetto ad ElasticSearch per GoldenStandard
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "NaePd5lzxh-rYkg1Aop3"))
if not es.ping():
    print("❌ Errore di connessione ad Elasticsearch!")
    reader.close()
    exit()

# 🔍 Costruzione query Elasticsearch per più campi
def getGoldenStandard(fields, queries):
    try:
        match_queries = [{"match": {field_mapping_GS[field]: queries[field]}} for field in fields]

        search_body = {
            "query": {
                "bool": {
                    "should": match_queries,
                    "minimum_should_match": 1  # Almeno una delle query deve matchare
                }
            },
            "_source": [field_mapping_GS["title"]],  # La risposta contiene solo il titolo
            "size": 10  # Limitato ai primi 10 risultati
        }
        response = es.search(index="films", body=search_body)
        GS = [hit['_source']['title'].lower().strip() for hit in response['hits']['hits']]
        return GS
    except Exception as e:
        print(f"Errore durante la ricerca del Golden Standard: {e}")
        return []

# 🟢 Usiamo le stesse query di BM25/VSM per Elasticsearch
golden_standard = getGoldenStandard(fields, queries)

# 🔍 Mostra i risultati di Elasticsearch
print(f"\n🔍 **Golden Standard da Elasticsearch**")
for title in golden_standard:
    print(f"🏅 Titolo: {title}")
print("-" * 50)

# 📊 **Calcolo Precision, Recall e F1-score**
def compute_metrics(retrieved, golden):
    """Calcola Precision, Recall e F1-score con fuzzy matching"""
    true_positives = 0
    for retrieved_title in retrieved:
        for golden_title in golden:
            if fuzz.ratio(retrieved_title.lower(), golden_title.lower()) >= 80:  # Se la similarità è alta
                true_positives += 1
                break  # Evita di contare lo stesso retrieved più volte

    retrieved_count = len(retrieved)
    golden_count = len(golden)

    precision = true_positives / retrieved_count if retrieved_count > 0 else 0
    recall = true_positives / golden_count if golden_count > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

from math import log2

def compute_ndcg(retrieved, golden, k=10):
    """Calcola NDCG a k per una lista di risultati rispetto al golden standard"""
    golden_lower = [g.lower() for g in golden]
    
    # Calcola gain binario (1 se match fuzzy, 0 altrimenti)
    gains = []
    for r in retrieved[:k]:
        match = any(fuzz.ratio(r.lower(), g) >= 80 for g in golden_lower)
        gains.append(1 if match else 0)
    
    # DCG
    dcg = sum(g / log2(i + 2) for i, g in enumerate(gains))
    
    # IDCG (gains ideali: tutti i rilevanti nei primi posti)
    ideal_gains = [1] * min(len(golden), k)
    idcg = sum(g / log2(i + 2) for i, g in enumerate(ideal_gains))
    
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

# 🔢 **Calcola metriche**
for model_name, retrieved_docs in result_by_model.items():
    precision, recall, f1 = compute_metrics(retrieved_docs, golden_standard)
    print(f"\n📊 **Metriche per il modello {model_name}**")
    print(f"🎯 Precision: {precision:.3f}")
    print(f"🎯 Recall: {recall:.3f}")
    print(f"🎯 F1-score: {f1:.3f}")
    print(f"🎯 NDCG: {compute_ndcg(retrieved_docs, golden_standard):.3f}")
    print("-" * 50)

import matplotlib.pyplot as plt

def precision_recall_curve(retrieved, golden):
    precisions = []
    recalls = []
    true_positives = 0
    golden_lower = [g.lower() for g in golden]
    matched_golden = set()

    for i, r in enumerate(retrieved, start=1):
        for g in golden_lower:
            if g not in matched_golden and fuzz.ratio(r.lower(), g) >= 80:
                true_positives += 1
                matched_golden.add(g)
                break
        precision = true_positives / i
        recall = true_positives / len(golden) if golden else 0
        precisions.append(precision)
        recalls.append(recall)
    
    return recalls, precisions

def plot_precision_recall(results_dict, golden_standard):
    plt.figure()
    for model, retrieved in results_dict.items():
        recalls, precisions = precision_recall_curve(retrieved, golden_standard)
        plt.plot(recalls, precisions, marker='o', label=model)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return

plot_precision_recall(result_by_model, golden_standard)
