# FluvialGen

A Python package for generating synthetic river networks and datasets.

## Installation

You can install FluvialGen using pip:

```bash
pip install fluvialgen
```

Or install from source:

```bash
git clone https://github.com/joseenriqueruiznavarro/FluvialGen.git
cd FluvialGen
pip install -e .
```

## Requirements

- Python >= 3.8
- NumPy
- Pandas
- SciPy
- Matplotlib
- GeoPandas
- Shapely
- Rasterio
- tqdm

## Integration with River Models

Here's an example of how to use MovingWindowBatcher with a River model:

```python
from river import compose, linear_model, preprocessing, optim, metrics
from generator.movingwindow_generator import MovingWindowBatcher
from river import datasets

# Create a River pipeline
model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))

# Initialize metrics
metric = metrics.MAE()

# Create the dataset and batcher
dataset = datasets.Bikes()
batcher = MovingWindowBatcher(
    dataset=dataset,
    instance_size=2,
    batch_size=2,
    n_instances=1000
)

# Train the model
try:
    # Process batches and train the model
    for X, y in batcher:
        # Train on each instance in the batch
        for i in range(len(X)):
            x = X.iloc[i]
            target = y.iloc[i]
            model.learn_one(x, target)
            
        # Make predictions and update metrics
        for i in range(len(X)):
            x = X.iloc[i]
            target = y.iloc[i]
            y_pred = model.predict_one(x)
            metric.update(target, y_pred)
            
    print(f"Final MAE: {metric}")

finally:
    # Clean up
    batcher.stop()
```

This example shows how to:
1. Create a River model pipeline
2. Use MovingWindowBatcher to process data in batches
3. Train the model on each instance in the batch
4. Make predictions and update metrics
5. Handle cleanup properly

The batcher provides a convenient way to process data in overlapping windows while maintaining compatibility with River's streaming learning paradigm.
