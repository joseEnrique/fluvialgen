import pytest
import pandas as pd
from datetime import datetime
from fluvialgen.past_forecast_batcher import PastForecastBatcher

class MockDataset:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        item = self.data[self.index]
        self.index += 1
        return item

def test_past_forecast_batcher_initialization():
    """Test initialization of PastForecastBatcher"""
    dataset = MockDataset([])
    batcher = PastForecastBatcher(dataset, past_size=3)
    
    assert batcher.past_size == 3
    assert batcher.forecast_size == 0
    assert batcher.buffer == []
    assert batcher._count == 0
    assert batcher._last_element is None

def test_past_forecast_batcher_with_sufficient_data():
    """Test PastForecastBatcher with sufficient data"""
    # Create mock data
    data = [
        ({'moment': datetime(2024, 1, 1, 0, 0), 'value': 1}, 1),
        ({'moment': datetime(2024, 1, 1, 0, 1), 'value': 2}, 2),
        ({'moment': datetime(2024, 1, 1, 0, 2), 'value': 3}, 3),
        ({'moment': datetime(2024, 1, 1, 0, 3), 'value': 4}, 4),
        ({'moment': datetime(2024, 1, 1, 0, 4), 'value': 5}, 5)
    ]
    
    dataset = MockDataset(data)
    batcher = PastForecastBatcher(dataset, past_size=3)
    
    # Get first instance
    X_past, y_past, current_x = batcher.get_message()
    
    # Check past data
    assert isinstance(X_past, pd.DataFrame)
    assert isinstance(y_past, pd.Series)
    assert len(X_past) == 3
    assert len(y_past) == 3
    assert X_past.iloc[0]['value'] == 1
    assert X_past.iloc[1]['value'] == 2
    assert X_past.iloc[2]['value'] == 3
    assert y_past.iloc[0] == 1
    assert y_past.iloc[1] == 2
    assert y_past.iloc[2] == 3
    
    # Check current x
    assert current_x['value'] == 4
    assert current_x['moment'] == datetime(2024, 1, 1, 0, 3)
    
    # Get second instance
    X_past, y_past, current_x = batcher.get_message()
    
    # Check past data
    assert len(X_past) == 3
    assert len(y_past) == 3
    assert X_past.iloc[0]['value'] == 2
    assert X_past.iloc[1]['value'] == 3
    assert X_past.iloc[2]['value'] == 4
    assert y_past.iloc[0] == 2
    assert y_past.iloc[1] == 3
    assert y_past.iloc[2] == 4
    
    # Check current x
    assert current_x['value'] == 5
    assert current_x['moment'] == datetime(2024, 1, 1, 0, 4)

def test_past_forecast_batcher_with_insufficient_data():
    """Test PastForecastBatcher with insufficient data"""
    # Create mock data with less than past_size + 1 elements
    data = [
        ({'moment': datetime(2024, 1, 1, 0, 0), 'value': 1}, 1),
        ({'moment': datetime(2024, 1, 1, 0, 1), 'value': 2}, 2)
    ]
    
    dataset = MockDataset(data)
    batcher = PastForecastBatcher(dataset, past_size=3)
    
    # Should raise StopIteration
    with pytest.raises(StopIteration):
        batcher.get_message()

def test_past_forecast_batcher_count():
    """Test that count is incremented correctly"""
    data = [
        ({'moment': datetime(2024, 1, 1, 0, 0), 'value': 1}, 1),
        ({'moment': datetime(2024, 1, 1, 0, 1), 'value': 2}, 2),
        ({'moment': datetime(2024, 1, 1, 0, 2), 'value': 3}, 3),
        ({'moment': datetime(2024, 1, 1, 0, 3), 'value': 4}, 4)
    ]
    
    dataset = MockDataset(data)
    batcher = PastForecastBatcher(dataset, past_size=2)
    
    # Get first instance
    batcher.get_message()
    assert batcher.get_count() == 1
    
    # Get second instance
    batcher.get_message()
    assert batcher.get_count() == 2 