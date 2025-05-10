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

### MovingWindowBatcher

This class provides a way to process data in overlapping windows with batching support:

```python
from river import compose, linear_model, preprocessing, optim, metrics
from fluvialgen.movingwindow_generator import MovingWindowBatcher
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
    instance_size=2,  # Size of each window
    batch_size=2,     # Number of instances per batch
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

### PastForecastBatcher

This class provides a way to process data with past and forecast windows:

```python
from river import compose, linear_model, preprocessing, optim, metrics
from fluvialgen.past_forecast_batcher import PastForecastBatcher
from river import datasets

# Create a River pipeline
model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))

# Initialize metrics
metric = metrics.MAE()

# Create the dataset and batcher
dataset = datasets.Bikes()
batcher = PastForecastBatcher(
    dataset=dataset,
    past_size=3,      # Number of past instances to include
    forecast_size=1,  # Number of future instances to include
    n_instances=1000
)

# Train the model
try:
    # Process instances and train the model
    for X_past, y_past, current_x in batcher:
        # Train on past data
        for i in range(len(X_past)):
            x = X_past.iloc[i]
            target = y_past.iloc[i]
            model.learn_one(x, target)
            
        # Make prediction for current moment
        y_pred = model.predict_one(current_x)
        metric.update(current_x['value'], y_pred)
            
    print(f"Final MAE: {metric}")

finally:
    # Clean up
    batcher.stop()
```

## Data Structure

### MovingWindowBatcher
For each batch, MovingWindowBatcher returns:
- `X`: DataFrame with all instances in the batch
- `y`: Series with all targets in the batch

For example, with `instance_size=2` and `batch_size=2`:
- First batch:
  - `X` = DataFrame with [x1,x2,x2,x3]
  - `y` = Series with [y1,y2,y2,y3]
- Second batch:
  - `X` = DataFrame with [x2,x3,x3,x4]
  - `y` = Series with [y2,y3,y3,y4]

### PastForecastBatcher
For each instance, PastForecastBatcher returns:
- `X_past`: DataFrame with past data
- `y_past`: Series with past targets
- `current_x`: Dictionary with current moment data

For example, with `past_size=3` and `forecast_size=1`:
- At time t=4:
  - `X_past` = DataFrame with [x1, x2, x3]
  - `y_past` = Series with [y1, y2, y3]
  - `current_x` = x4 (current moment)
