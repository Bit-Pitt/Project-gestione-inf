from elasticsearch import Elasticsearch
from fuzzywuzzy import fuzz
import re

# Connessione a Elasticsearch con autenticazione
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "..."))


# Verifica connessione
if not es.ping():
    print("❌ Errore di connessione a Elasticsearch!")
    exit()

# Mappatura dei campi
field_mapping = {
    "title": "title",
    "year": "year",
    "genre": "genre",
    "country": "country",
    "plot": "plot"
}

# 🎯 Input utente
valid_fields = list(field_mapping.keys())
field = input(f"In che campo vuoi cercare? {valid_fields}: ").strip().lower()

if field not in field_mapping:
    print("❌ Campo non valido! Usa uno tra:", valid_fields)
    exit()

query_string = input(f"Inserisci la query per '{field}': ").strip()

# 📝 Correzione ortografica della query (usando fuzzywuzzy)
def correct_spelling(query_string):
    """Corregge la query cercando la migliore corrispondenza tramite fuzzy matching"""
    # Esegui la ricerca su Elasticsearch per trovare i primi 10 risultati
    search_body = {
        "query": {
            "match": {
                field_mapping["title"]: query_string
            }
        },
        "_source": [field_mapping["title"]],
        "size": 10  # Limitato ai primi 10 risultati
    }

    response = es.search(index="films", body=search_body)

    best_match = query_string  # Default: la query originale
    best_score = 0
    for hit in response['hits']['hits']:
        title = hit['_source']['title']
        score = fuzz.ratio(query_string.lower(), title.lower())  # Confronto fuzzy
        if score > best_score:
            best_score = score
            best_match = title

    return best_match if best_score >= 80 else query_string  # Restituisce la migliore corrispondenza se il punteggio è sufficiente

query_string = correct_spelling(query_string)  # Applica la correzione alla query

# 🎯 Esegui la ricerca su Elasticsearch
search_body = {
    "query": {
        "match": {
            field_mapping[field]: query_string
        }
    },
    "_source": [field_mapping["title"]],
    "size": 10  # Limitato ai primi 10 risultati
}

response_vsm = es.search(index="films", body=search_body)
response_bm25 = es.search(index="films", body=search_body)

# 📊 Confronto dei risultati
print(f"\n🔍 Risultati per '{query_string}' nel campo '{field}':\n")

# Inizializzo vettori per usarli dopo nel GoldenStandard
vsm_results = []
bm25_results = []

print("📌 **VSM (Vector Space Model)**")
for hit in response_vsm['hits']['hits']:
    title = hit['_source']['title']
    vsm_results.append(title)
    print(f"🎬 Titolo: {title} | ⭐ Score: {hit['_score']}")
print("-" * 50)

print("📌 **BM25 (Best Matching 25)**")
for hit in response_bm25['hits']['hits']:
    title = hit['_source']['title']
    bm25_results.append(title)
    print(f"🎬 Titolo: {title} | ⭐ Score: {hit['_score']}")
print("-" * 50)

# **Golden Standard (utilizzando i risultati di Elasticsearch)**
# Otteniamo il golden standard dalla ricerca su Elasticsearch.
def get_golden_standard(query_string):
    """Interroga Elasticsearch per ottenere i primi 10 risultati come golden standard"""
    search_body = {
        "query": {
            "match": {
                field_mapping["title"]: query_string
            }
        },
        "_source": [field_mapping["title"]],
        "size": 10  # Limitato ai primi 10 risultati
    }

    response = es.search(index="films", body=search_body)

    golden_standard = [hit['_source']['title'].lower().strip() for hit in response['hits']['hits']]
    return golden_standard

golden_standard = get_golden_standard(query_string)

print(f"\n🏅 **Golden Standard (Elasticsearch Top Results) per '{query_string}':**")
for title in golden_standard:
    print(f"✅ {title}")
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

# 🔢 **Calcola metriche**
precision_vsm, recall_vsm, f1_vsm = compute_metrics(vsm_results, golden_standard)
precision_bm25, recall_bm25, f1_bm25 = compute_metrics(bm25_results, golden_standard)

# 📊 **Stampa metriche**
print("\n📈 **Performance dei Modelli**")
print("📌 **VSM (Vector Space Model)**")
print(f"🎯 Precision: {precision_vsm:.3f}")
print(f"🎯 Recall: {recall_vsm:.3f}")
print(f"🎯 F1-score: {f1_vsm:.3f}")
print("-" * 50)

print("📌 **BM25 (Best Matching 25)**")
print(f"🎯 Precision: {precision_bm25:.3f}")
print(f"🎯 Recall: {recall_bm25:.3f}")
print(f"🎯 F1-score: {f1_bm25:.3f}")
print("-" * 50)
