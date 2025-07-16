# Istruzioni per l'Uso del Progetto

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