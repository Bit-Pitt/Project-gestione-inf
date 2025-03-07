import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

def search_movies():
    """Funzione chiamata quando si preme il pulsante Cerca"""
    title = title_var.get().strip()
    year = year_var.get().strip()
    genre = genre_var.get().strip()
    country = country_var.get().strip()
    plot = plot_var.get().strip()
    model = model_var.get()
    
    # Creazione della query
    query_parts = []
    if title:
        query_parts.append(f"title:{title}")
    if year:
        query_parts.append(f"year:{year}")
    if genre:
        query_parts.append(f"genre:{genre}")
    if country:
        query_parts.append(f"country:{country}")
    if plot:
        query_parts.append(f"plot:{plot}")
    
    query_string = ", ".join(query_parts)
    
    if not query_string:
        messagebox.showerror("Errore", "Devi inserire almeno un campo di ricerca!")
        return
    
    # Esegui il programma Python con i parametri
    try:
        result = subprocess.run(
            ["python", "search_UI.py", model, query_string],
            capture_output=True, text=True
        )
        
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, result.stdout)
    except Exception as e:
        messagebox.showerror("Errore", f"Errore durante l'esecuzione: {e}")

# Creazione della finestra principale
root = tk.Tk()
root.title("Ricerca Film con Lucene")
root.geometry("500x600")

# Variabili Tkinter per i campi di input
title_var = tk.StringVar()
year_var = tk.StringVar()
genre_var = tk.StringVar()
country_var = tk.StringVar()
plot_var = tk.StringVar()
model_var = tk.StringVar(value="BM25")  # Default BM25

# Layout
fields = [
    ("Titolo", title_var),
    ("Anno", year_var),
    ("Genere", genre_var),
    ("Paese", country_var),
    ("Trama", plot_var),
]

for i, (label, var) in enumerate(fields):
    tk.Label(root, text=label).grid(row=i, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(root, textvariable=var, width=40).grid(row=i, column=1, padx=10, pady=5)

# Selettore modello di ricerca
tk.Label(root, text="Modello di Ricerca").grid(row=len(fields), column=0, sticky="w", padx=10, pady=5)
model_menu = ttk.Combobox(root, textvariable=model_var, values=["BM25", "VSM"], state="readonly")
model_menu.grid(row=len(fields), column=1, padx=10, pady=5)

# Pulsante Cerca
tk.Button(root, text="Cerca", command=search_movies).grid(row=len(fields) + 1, columnspan=2, pady=10)

# Area Risultati
results_text = tk.Text(root, height=15, width=55)
results_text.grid(row=len(fields) + 2, columnspan=2, padx=10, pady=5)

# Avvia Tkinter
root.mainloop()
