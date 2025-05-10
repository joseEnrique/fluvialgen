from river import datasets
import pandas as pd
from fluvialgen.movingwindow_generator import MovingWindowBatcher
from collections import Counter

dataset = datasets.Bikes()

batcher = MovingWindowBatcher(
    dataset=dataset,
    instance_size=3,
    batch_size=3,
    n_instances=10
)

x_batches = []
y_batches = []

for x, y in batcher:
    x_batches.append(x)
    y_batches.append(y)

print((x_batches))
#print(len(y_batches))

def dataframe_to_hash(df: pd.DataFrame) -> int:
    # Convertimos el DataFrame a CSV (sin incluir el índice) y lo transformamos en un hash
    # Asegúrate de que la conversión refleje completamente la estructura y contenido del DataFrame
    return hash(df.to_csv(index=False))

# Convertimos cada DataFrame en un hash
hashes = [dataframe_to_hash(df) for df in x_batches]

# Contamos la cantidad de ocurrencias de cada hash (cada hash representa un DataFrame "idéntico")
contador = Counter(hashes)


print(contador)