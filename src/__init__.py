
__version__ = "3.0.0"
__author__ = "Yeabsira Samuel"
__course__ = "Supervised Learning - Linear Regression"
__currency__ = "Ethiopian Birr (ETB)"

# This makes imports easier
# Instead of: from src.data_loader import load_data
# You can do: from src import load_data

from .data_loader import load_data, explore_data
from .model import LinearRegressionModel
from .train import train_model
from .predict import predict_bill

__all__ = [
    'load_data',
    'explore_data',
    'LinearRegressionModel',
    'train_model',
    'predict_bill'
]