
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