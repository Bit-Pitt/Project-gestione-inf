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
def plot_precision_recall_with_variance(avg_curves, output_path, model1="BM25", model2="TF-IDF"):
    recall_levels = [i / 10 for i in range(11)]

    avg1, std1 = compute_avg_and_std(avg_curves[model1])
    avg2, std2 = compute_avg_and_std(avg_curves[model2])

    plt.figure(figsize=(8, 6))

    plt.plot(recall_levels, avg1, marker='o', label=model1, color="blue")
    plt.fill_between(recall_levels, avg1 - std1, avg1 + std1, alpha=0.2, color="blue", label=f"{model1} ±1 std")

    plt.plot(recall_levels, avg2, marker='x', label=model2, color="green")
    plt.fill_between(recall_levels, avg2 - std2, avg2 + std2, alpha=0.2, color="green", label=f"{model2} ±1 std")

    plt.xlabel("Recall medio")
    plt.ylabel("Precision media")
    plt.title(f"Curve Precision-Recall medie con varianza ({model1} vs {model2})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Grafico con varianza salvato in: {output_path}")


def plot_per_query_precision_recall(avg_curves, output_path, model1="BM25", model2="TF-IDF"):
    recall_levels = [i / 10 for i in range(11)]
    num_queries = len(avg_curves[model1])
    rows, cols = 5, 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(f"Precision-Recall Interpolated per Query ({model1} vs {model2})", fontsize=16)

    for idx in range(num_queries):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        ax.plot(recall_levels, avg_curves[model1][idx], label=model1, color="blue", marker="o", linewidth=1)
        ax.plot(recall_levels, avg_curves[model2][idx], label=model2, color="green", marker="x", linewidth=1)

        ax.set_title(f"Query {idx+1}", fontsize=8)
        ax.grid(True)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

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
def plot_querywise_precision_bar_chart(precision_dict, output_path, model1="BM25", model2="TF-IDF"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_queries = len(precision_dict[model1])
    query_indices = np.arange(1, num_queries + 1)

    width = 0.35  # larghezza delle barre
    fig, ax = plt.subplots(figsize=(12, 6))

    bm25_precisions = precision_dict[model1]
    tfidf_precisions = precision_dict[model2]

    ax.bar(query_indices - width/2, bm25_precisions, width, label=model1, color='skyblue')
    ax.bar(query_indices + width/2, tfidf_precisions, width, label=model2, color='lightgreen')

    ax.set_xlabel('Query ID')
    ax.set_ylabel('Precisione')
    ax.set_title('Precisione media per query (BM25 vs TF-IDF)')
    ax.set_xticks(query_indices)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Grafico precision per query salvato in: {output_path}")



# Analogo ma per le recall
def plot_querywise_recall_bar_chart(recall_dict, output_path, model1="BM25", model2="TF-IDF"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_queries = len(recall_dict[model1])
    query_indices = np.arange(1, num_queries + 1)

    width = 0.35  # larghezza delle barre
    fig, ax = plt.subplots(figsize=(12, 6))

    bm25_recalls = recall_dict[model1]
    tfidf_recalls = recall_dict[model2]

    ax.bar(query_indices - width/2, bm25_recalls, width, label=model1, color='coral')
    ax.bar(query_indices + width/2, tfidf_recalls, width, label=model2, color='gold')

    ax.set_xlabel('Query ID')
    ax.set_ylabel('Recall')
    ax.set_title('Recall media per query (BM25 vs TF-IDF)')
    ax.set_xticks(query_indices)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Grafico recall per query salvato in: {output_path}")




# Funzione per plottare la NDCG (in modo analogo alla recall e precision)
def plot_querywise_ndcg_bar_chart(ndcg_dict, output_path, model1="BM25", model2="TF-IDF"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_queries = len(ndcg_dict[model1])
    query_indices = np.arange(1, num_queries + 1)

    width = 0.35  # larghezza delle barre
    fig, ax = plt.subplots(figsize=(12, 6))

    bm25_ndcg = ndcg_dict[model1]
    tfidf_ndcg = ndcg_dict[model2]

    ax.bar(query_indices - width/2, bm25_ndcg, width, label=model1, color='mediumslateblue')
    ax.bar(query_indices + width/2, tfidf_ndcg, width, label=model2, color='mediumseagreen')

    ax.set_xlabel('Query ID')
    ax.set_ylabel('NDCG')
    ax.set_title('NDCG media per query (BM25 vs TF-IDF)')
    ax.set_xticks(query_indices)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Grafico NDCG per query salvato in: {output_path}")


# Plot della R@ precision con grafico a barra singola!
def plot_querywise_rprecision_bar_chart(r_precision_dict, output_path, model1="BM25", model2="TF-IDF"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = len(r_precision_dict[model1])
    x = np.arange(1, n + 1)

    bm25_values = np.array(r_precision_dict[model1])
    tfidf_values = np.array(r_precision_dict[model2])
    
    # Differenza (positiva = BM25 meglio, negativa = TF-IDF meglio)
    diff = bm25_values - tfidf_values

    colors = ['green' if d >= 0 else 'red' for d in diff]
    bar_heights = diff

    plt.figure(figsize=(12, 6))
    plt.bar(x, bar_heights, color=colors)
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(x)
    plt.xlabel("Query ID")
    plt.ylabel("Δ R@3 (BM25 - TF-IDF)")
    plt.title(f"Differenza di R@3 per Query (verde={model1} vince, rosso={model2} vince)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅Grafico r@prec salvato in: {output_path}")




# Plotting delle Medie e deviazioni Aggregate, passiamo tutti i dizionari completi
def plot_final_metric_summary_barplot(precision_dict, recall_dict, f1_dict, ndcg_dict, save_path, model1="BM25", model2="TF-IDF"):
    metrics = ['Precision', 'Recall', 'F1', 'NDCG']
    models = [model1, model2]
    
    # Calcolo medie e deviazioni standard per ciascun modello e metrica
    averages = {
        model1: [
            np.mean(precision_dict[model1]),
            np.mean(recall_dict[model1]),
            np.mean(f1_dict[model1]),
            np.mean(ndcg_dict[model1]),
        ],
        model2: [
            np.mean(precision_dict[model2]),
            np.mean(recall_dict[model2]),
            np.mean(f1_dict[model2]),
            np.mean(ndcg_dict[model2]),
        ]
    }

    std_devs = {
        model1: [
            np.std(precision_dict[model1]),
            np.std(recall_dict[model1]),
            np.std(f1_dict[model1]),
            np.std(ndcg_dict[model1]),
        ],
        model2: [
            np.std(precision_dict[model2]),
            np.std(recall_dict[model2]),
            np.std(f1_dict[model2]),
            np.std(ndcg_dict[model2]),
        ]
    }

    # Plotting
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, averages[model1], width, label=model1, yerr=std_devs[model1], capsize=5)
    ax.bar(x + width/2, averages[model2], width, label=model2, yerr=std_devs[model2], capsize=5)

    # Etichette e stile
    ax.set_ylabel('Score')
    ax.set_title(f'Average Metric Summary with Std Deviation ({model1} vs {model2})')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Grafico Medie Aggregate + errore std salvato in: {save_path}")

