# Progetto PyLucene / Elasticsearch - Documentazione

## 📦 Requisiti

Installare le dipendenze:
pip install -r requirements.txt

Per l'uso di elasticsearch, scaricare da https://www.elastic.co/downloads/elasticsearch, unzippare la cartella e,
    da terminale, digitare da dentro la cartella: ./bin/elasticsearch
L'uso di elasticsearch è essere usato come goldenstandard. 

🏗️ Script e funzionalità

make_II.py:
    Crea gli indici:
        Locali (PyLucene) con StandardAnalyzer, Lemmatizzatore.
        Elasticsearch, se attivo.

manualQuery.py

    Permette di eseguire query in modo interattivo:
        L’utente inserisce i campi desiderati.
        Le metriche vengono stampate a terminale.
        Viene generato il grafico precision_curve.png che mostra la precisione ai vari livelli di recall per:
            TF-IDF (standard analyzer)
            BM25 (standard analyzer)
            TF-IDF con stemming
            TF-IDF con lemmatizzazione

benchmark.py

    Esegue test sulle 15 query UIN:
        Genera grafici in pylucene/grafici/
        Mostra:
            Precision, Recall, F1, NDCG, R-Precision per ogni query
            Medie e deviazioni standard aggregate
            Grafici comparativi tra TF-IDF e BM25 con standard analyzer

std_vs_lemma.py
    Confronta:
        TF-IDF con StandardAnalyzer
        TF-IDF con Lemmatizzazione
        Valuta metriche individuali e precision@recall.

bench_plotting.py
    Contiene le funzioni per generare i grafici usati negli script di benchmark.

utils.py
    Funzioni ausiliarie:
        Calcolo metriche (precision, recall, f1, ndcg, ecc.)
        Normalizzazione query e supporto ai test

dataset-constr.py
    Costruisce il dataset di partenza da cui vengono creati gli indici e i golden standard.

📂 Output
    I grafici vengono salvati in pylucene/grafici/
    I risultati intermedi possono essere stampati a terminale o salvati su file se specificato nei singoli script.

🔧 Note
    Gli script sono modulari e riutilizzabili.
    Per testare tutto da zero: eseguire prima make_II.py, poi benchmark.py o manualQuery.py a seconda del tipo di analisi.