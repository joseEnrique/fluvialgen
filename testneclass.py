from river import datasets
import pandas as pd
from fluvialgen.past_forecast_batcher import PastForecastBatcher
from collections import Counter

dataset = datasets.Bikes()

batcher = PastForecastBatcher(
    dataset=dataset,
    past_size=2,
    forecast_size=0,
    n_instances=5
)

x_batches = []
y_batches = []

for x, y in dataset.take(5):
    print (x,y)

print ("This is what I really care about")

for x,y,current in batcher:
    print (current)

