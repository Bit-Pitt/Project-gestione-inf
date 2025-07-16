import requests
import pandas as pd
import wikipediaapi

def query_wikidata():
    url = "https://query.wikidata.org/sparql"
    query = """
    SELECT ?film ?filmLabel ?year ?genreLabel ?countryLabel WHERE {
      ?film wdt:P31 wd:Q11424;  # Istanza di "film"
            wdt:P577 ?date.     # Data di pubblicazione
      OPTIONAL { ?film wdt:P136 ?genre. }
      OPTIONAL { ?film wdt:P495 ?country. }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
      BIND(YEAR(?date) AS ?year)
    }
    LIMIT 10000
    """
    headers = {"User-Agent": "WikidataFilmScraper/1.0 (your_email@example.com)"}
    try:
        response = requests.get(url, params={"query": query, "format": "json"}, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Errore nella richiesta SPARQL: {e}")
        return None

def get_wikipedia_summary(title, lang='en'):
    user_agent = "MyWikiScraper/1.0 (als.chiarabini@gmail.com)"  # Cambia con la tua email o un identificativo valido
    wiki_wiki = wikipediaapi.Wikipedia(lang, headers={'User-Agent': user_agent})
    
    page = wiki_wiki.page(title)
    if page.exists():
        return page.summary
    else:
        print(f"Trama non trovata per {title}.")
        return "No plot available"

def extract_film_data():
    data = query_wikidata()
    films = []
    titles_seen = set()  # Set per tracciare i titoli già visti
    if data is None:
        return films
    
    for item in data.get("results", {}).get("bindings", []):
        title = item.get("filmLabel", {}).get("value", "Unknown")
        
        # Controlla se il titolo è già stato visto
        if title in titles_seen:
            continue  # Salta il film se è già stato processato
        titles_seen.add(title)
        
        plot = get_wikipedia_summary(title)
        films.append({
            "Title": title,
            "Year": item.get("year", {}).get("value", "Unknown"),
            "Genre": item.get("genreLabel", {}).get("value", "Unknown"),
            "Country": item.get("countryLabel", {}).get("value", "Unknown"),
            "Plot": plot
        })
    return films

def save_to_csv(films, filename="films.csv"):
    if not films:
        print("Nessun dato da salvare.")
        return
    df = pd.DataFrame(films)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"File salvato: {filename}")

# Esegui il processo
film_data = extract_film_data()
save_to_csv(film_data)
