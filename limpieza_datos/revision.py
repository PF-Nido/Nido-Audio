import pandas as pd

df = pd.read_parquet("embeddings_limpios.parquet")
df["sp"] = df["genero"] + " " + df["especie"]
conteo = df["sp"].value_counts()

print(f"Especies con < 10 muestras:  {(conteo < 10).sum()}")
print(f"Especies con < 30 muestras:  {(conteo < 30).sum()}")
print(f"Especies con < 50 muestras:  {(conteo < 50).sum()}")
print(f"Especies con >= 50 muestras: {(conteo >= 50).sum()}")
print()
print(conteo.tail(20))  # las 20 más pequeñas