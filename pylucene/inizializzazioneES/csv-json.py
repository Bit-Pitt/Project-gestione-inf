import csv
import json

# Nome del file CSV di input e del file JSON di output
csv_file = "films.csv"  # Sostituiscilo con il nome corretto del tuo file
json_file = "films.json"

# Leggi il CSV e converti in JSON
movies = []
with open(csv_file, encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        movie = {
            "title": row.get("Title", "").strip(),
            "year": int(row.get("Year", 0)) if row.get("Year", "").isdigit() else None,
            "genre": row.get("Genre", "").strip(),
            "country": row.get("Country", "").strip(),
            "plot": row.get("Plot", "").strip()
        }
        movies.append(movie)

# Salva i dati in JSON
with open(json_file, "w", encoding="utf-8") as file:
    json.dump(movies, file, indent=4, ensure_ascii=False)

print(f"✅ File JSON salvato come {json_file} con {len(movies)} film.")
