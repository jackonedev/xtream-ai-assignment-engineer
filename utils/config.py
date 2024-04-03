import os

PROJECT_ROOT = os.path.dirname('.')

DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'diamonds', 'diamonds.csv')
SEED = 33

os.makedirs(os.path.join(PROJECT_ROOT, '.models'), exist_ok=True)
MODELS_PATH = os.path.join(PROJECT_ROOT, '.models')

os.makedirs(os.path.join(MODELS_PATH, 'data'), exist_ok=True)
MODEL_DATA_PATH = os.path.join(MODELS_PATH, 'data')