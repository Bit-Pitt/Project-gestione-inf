Indicazioni per l'installazione:
- Leggere il file requirmets.txt per le librerie necessarie

Indicazioni sull'uso:

1) indicazioni sull'uso di Whoosh:
- make_II è lo scrip che genera tutti gli indici sia locali che di elastic search. Per far funzionare elastic search deve essere scaricato lo zip da internet e fatto partire il servizio accedendo una volta estratta la cartella a /bin/elasticsearch.bat.
Tuttavia non sempre è sufficiente runnare  /bin/elasticsearch.bat a noi è stato necessario modificare il file di config ".yml" per ad esempio a me togliere la sicurezza (andando quindi di http) e cambiare l'ip (non local_host ma un altro visto che sono su wsl)

- script_interattivo permette di creare interattivamente la query selezionando i field disponibili,le metriche saranno mostrate sul terminale e in aggiunta viene creato un grafico "precision_curve.png" che mostrerà la precisione ai livelli di recall per tutti e 4 i SE sviluppati ovvero:
Un tf_idf, e bm25 standard su indice creato con std_analyzer
Un tf_idf creato su indice che usa stemming
Un tf_idf creato su indice che usa lemmatizazzione
Ognuno sarà valutato con appropriati golden standard di elastic search.

- benchmark.py  testa le 15 UIN create e mostra una serie di grafici in Whoosh/grafici/ 
 I grafici mostreranno la precision, la recall, la precision ai livelli di recall per ogni query ma anche una finale facendo la media e la deviazione standard, un grafico per la ndcg, rprecision e un grafico che mostri le medie aggregate di precision,recall,f1,ndcg.
 Questo per confrontare tf_idf e bm25 con std_analyzer

-  std_vs_lemma è analogo a benchmark.py ma mosta il confronto per td_idf standard e td_idf con lemmatization

- std_vs_stem analogo a il precedente

- query_espansa analogo a benchamark ma le query vengono espanse tramite dei sinonimi utilizzando wordnet e nltk

2) # Progetto PyLucene / Elasticsearch - Documentazione

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

3) # Istruzioni per l'Uso del Progetto - Postgresql

## ✅ Requisiti

Installare tutte le dipendenze con:
    pip install -r requirements.txt

Assicurarsi di aver installato e correttamente configurato:
    PyLucene (in locale, vedi riga -e /.../jcc nel requirements.txt)
    PostgreSQL con tabella films già indicizzata via to_tsvector

🔍 File Principali

benchmark_SQL.py
    Esegue benchmarking tra i due sistemi di ricerca su PostgreSQL:
        VSM-like: basato su ts_rank
        BM25-like: basato su ts_rank_cd
    Per ciascuna query, vengono calcolate:
        precision, recall, f1-score, ndcg
        grafici salvati automaticamente (usando bench_plotting.py)

utils.py
    Contiene funzioni di supporto:
        calcolo metriche (compute_precision, recall, f1, ndcg, etc.)
        gestione dei golden standard
        stampa dei risultati

bench_plotting.py
    Genera grafici:
        precision ai vari livelli di recall
        istogrammi comparativi (VSM vs BM25)
        r-precision, ndcg, medie aggregate, ecc.

📌 Note Finali
    Le query usate sono 15 e salvate in un dizionario all’interno del main script.
    Per funzionare correttamente, serve un database PostgreSQL popolato e configurato.
    Le metriche vengono salvate sia su console che in forma grafica nella cartella specificata.