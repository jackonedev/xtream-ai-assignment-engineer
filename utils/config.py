import os

SEED = 33
PROJECT_ROOT = os.path.dirname('.')
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'datasets')

DATASET_PATH = os.path.join(DATASET_ROOT, 'diamonds', 'diamonds.csv')
DATASET_INFO_PATH = os.path.join(DATASET_ROOT, 'diamonds', 'README.md')

os.makedirs(os.path.join(PROJECT_ROOT, '.models'), exist_ok=True)
MODELS_PATH = os.path.join(PROJECT_ROOT, '.models')

os.makedirs(os.path.join(MODELS_PATH, 'data'), exist_ok=True)
MODEL_DATA_PATH = os.path.join(MODELS_PATH, 'data')
