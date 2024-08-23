from .src.config import Config
from pathlib import Path

# Model version
__version__ = '0.01'

# Initialise global variables from config module
config = Config()
data_path = Path(config.data_path)
churn_app_path = Path(config.churn_app_models)
seed = config.seed