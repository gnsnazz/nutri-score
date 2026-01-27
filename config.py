import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cartelle Dataset
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset", "raw", "en.openfoodfacts.org.products.tsv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "dataset", "processed", "dataset_nutrizione_pulito.csv")
ENRICHED_DATA_PATH = os.path.join(BASE_DIR, "dataset", "processed", "nutriscore_products.csv")

# Cartelle Output
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Cartelle Knowledge Base (Prolog)
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
FACTS_PATH = os.path.join(KB_DIR, "facts.pl")
RULES_PATH = os.path.join(KB_DIR, "rules.pl")


# Nome della colonna target
TARGET_COL = "target"

# Features Numeriche
FEATURES = [
    'sugars_100g', 'fat_100g', 'salt_100g', 'fiber_100g',
    'fruit_veg_100g', 'additives_n', 'proteins_100g',
    'is_empty_calories', 'is_hidden_sodium', 'is_hyper_processed',
    'is_high_satiety', 'is_low_fat_sugar_trap', 'is_misleading_label',
    'symbolic_score'
]
