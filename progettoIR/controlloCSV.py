import pandas as pd
df = pd.read_csv("films.csv")
print("Il numero di righe è: ")
print(df.shape)  # Numero di righe e colonne
print("I duplicati sono: ")
print(df["Title"].duplicated().sum())
