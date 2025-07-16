import os
import numpy as np
import matplotlib.pyplot as plt

#Computa media e varianza delle curve (una per ogni query)
def compute_avg_and_std(curves):
    curves_np = np.array(curves)  # shape: (num_query, 11 livelli di recall)
    avg = np.mean(curves_np, axis=0)
    std = np.std(curves_np, axis=0)
    return avg, std

# Funzione per plottare la precisione ai livelli di recall con varianza (std)
def plot_precision_recall_with_variance(avg_curves, output_path):
    recall_levels = [i / 10 for i in range(11)]

    bm25_avg, bm25_std = compute_avg_and_std(avg_curves["BM25"])
    tfidf_avg, tfidf_std = compute_avg_and_std(avg_curves["TF-IDF"])

    plt.figure(figsize=(8, 6))

    # BM25
    plt.plot(recall_levels, bm25_avg, marker='o', label="to_tsquery_cd", color="blue")
    plt.fill_between(recall_levels, bm25_avg - bm25_std, bm25_avg + bm25_std,
                     alpha=0.2, color="blue", label="to_tsquery_cd ±1 std")

    # TF-IDF
    plt.plot(recall_levels, tfidf_avg, marker='x', label="to_tsquery", color="green")
    plt.fill_between(recall_levels, tfidf_avg - tfidf_std, tfidf_avg + tfidf_std,
                     alpha=0.2, color="green", label="to_tsquery ±1 std")

    plt.xlabel("Recall medio")
    plt.ylabel("Precision media")
    plt.title("Curve Precision-Recall medie con varianza (to_tsquery_cd vs to_tsquery)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Grafico con varianza salvato in: {output_path}")



def plot_per_query_precision_recall(avg_curves, output_path):
    recall_levels = [i / 10 for i in range(11)]

    num_queries = len(avg_curves["BM25"])
    rows, cols = 5, 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle("Precision-Recall Interpolated per Query (to_tsquery_cd vs to_tsquery)", fontsize=16)

    for idx in range(num_queries):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        ax.plot(recall_levels, avg_curves["BM25"][idx], label="to_tsquery_cd", color="blue", marker="o", linewidth=1)
        ax.plot(recall_levels, avg_curves["TF-IDF"][idx], label="to_tsquery", color="green", marker="x", linewidth=1)

        ax.set_title(f"Query {idx+1}", fontsize=8)
        ax.grid(True)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # Rimuove gli assi vuoti se ci fossero
    for idx in range(num_queries, rows * cols):
        fig.delaxes(axes[idx // cols, idx % cols])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Subplot per-query salvato in: {output_path}")



# PLotting della media delle precisione per query
def plot_querywise_precision_bar_chart(precision_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_queries = len(precision_dict["BM25"])
    query_indices = np.arange(1, num_queries + 1)

    width = 0.35  # larghezza delle barre
    fig, ax = plt.subplots(figsize=(12, 6))

    bm25_precisions = precision_dict["BM25"]
    tfidf_precisions = precision_dict["TF-IDF"]

    ax.bar(query_indices - width/2, bm25_precisions, width, label='to_tsquery_cd', color='skyblue')
    ax.bar(query_indices + width/2, tfidf_precisions, width, label='to_tsquery', color='lightgreen')

    ax.set_xlabel('Query ID')
    ax.set_ylabel('Precisione')
    ax.set_title('Precisione media per query (to_tsquery_cd vs to_tsquery)')
    ax.set_xticks(query_indices)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Grafico precision per query salvato in: {output_path}")



# Analogo ma per le recall
def plot_querywise_recall_bar_chart(recall_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_queries = len(recall_dict["BM25"])
    query_indices = np.arange(1, num_queries + 1)

    width = 0.35  # larghezza delle barre
    fig, ax = plt.subplots(figsize=(12, 6))

    bm25_recalls = recall_dict["BM25"]
    tfidf_recalls = recall_dict["TF-IDF"]

    ax.bar(query_indices - width/2, bm25_recalls, width, label='to_tsquery_cd', color='coral')
    ax.bar(query_indices + width/2, tfidf_recalls, width, label='to_tsquery', color='gold')

    ax.set_xlabel('Query ID')
    ax.set_ylabel('Recall')
    ax.set_title('Recall media per query (to_tsquery_cd vs to_tsquery)')
    ax.set_xticks(query_indices)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Grafico recall per query salvato in: {output_path}")




# Funzione per plottare la NDCG (in modo analogo alla recall e precision)
def plot_querywise_ndcg_bar_chart(ndcg_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_queries = len(ndcg_dict["BM25"])
    query_indices = np.arange(1, num_queries + 1)

    width = 0.35  # larghezza delle barre
    fig, ax = plt.subplots(figsize=(12, 6))

    bm25_ndcg = ndcg_dict["BM25"]
    tfidf_ndcg = ndcg_dict["TF-IDF"]

    ax.bar(query_indices - width/2, bm25_ndcg, width, label='to_tsquery_cd', color='mediumslateblue')
    ax.bar(query_indices + width/2, tfidf_ndcg, width, label='to_tsquery', color='mediumseagreen')

    ax.set_xlabel('Query ID')
    ax.set_ylabel('NDCG')
    ax.set_title('NDCG media per query (to_tsquery_cd vs to_tsquery)')
    ax.set_xticks(query_indices)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Grafico NDCG per query salvato in: {output_path}")


# Plot della R@ precision con grafico a barra singola!
def plot_querywise_rprecision_bar_chart(r_precision_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = len(r_precision_dict["BM25"])
    x = np.arange(1, n + 1)

    bm25_values = np.array(r_precision_dict["BM25"])
    tfidf_values = np.array(r_precision_dict["TF-IDF"])
    
    # Differenza (positiva = BM25 meglio, negativa = TF-IDF meglio)
    diff = bm25_values - tfidf_values

    colors = ['green' if d >= 0 else 'red' for d in diff]
    bar_heights = diff

    plt.figure(figsize=(12, 6))
    plt.bar(x, bar_heights, color=colors)
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(x)
    plt.xlabel("Query ID")
    plt.ylabel("Δ R@3 (to_tsquery_cd - to_tsquery)")
    plt.title("Differenza di R@3 per Query (verde=to_tsquery_cd vince, rosso=to_tsquery vince)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅Grafico r@prec salvato in: {output_path}")




# Plotting delle Medie e deviazioni Aggregate, passiamo tutti i dizionari completi
def plot_final_metric_summary_barplot(precision_dict, recall_dict, f1_dict, ndcg_dict, save_path):
    metrics = ['Precision', 'Recall', 'F1', 'NDCG']
    models = ['BM25', 'TF-IDF']
    
    # Calcolo medie e deviazioni standard per ciascun modello e metrica
    averages = {
        "BM25": [
            np.mean(precision_dict["BM25"]),
            np.mean(recall_dict["BM25"]),
            np.mean(f1_dict["BM25"]),
            np.mean(ndcg_dict["BM25"]),
        ],
        "TF-IDF": [
            np.mean(precision_dict["TF-IDF"]),
            np.mean(recall_dict["TF-IDF"]),
            np.mean(f1_dict["TF-IDF"]),
            np.mean(ndcg_dict["TF-IDF"]),
        ]
    }

    std_devs = {
        "BM25": [
            np.std(precision_dict["BM25"]),
            np.std(recall_dict["BM25"]),
            np.std(f1_dict["BM25"]),
            np.std(ndcg_dict["BM25"]),
        ],
        "TF-IDF": [
            np.std(precision_dict["TF-IDF"]),
            np.std(recall_dict["TF-IDF"]),
            np.std(f1_dict["TF-IDF"]),
            np.std(ndcg_dict["TF-IDF"]),
        ]
    }

    # Plotting
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, averages["BM25"], width, label='to_tsquery_cd', yerr=std_devs["BM25"], capsize=5)
    rects2 = ax.bar(x + width/2, averages["TF-IDF"], width, label='to_tsquery', yerr=std_devs["TF-IDF"], capsize=5)

    # Etichette e stile
    ax.set_ylabel('Score')
    ax.set_title('Average Metric Summary with Std Deviation')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅Grafico Medie Aggregate + errore std salvato in: {save_path}")


