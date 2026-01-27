import os
import sys
import pandas as pd

from utils.preprocessor import process_and_save
from knowledge_base.kb import KnowledgeBase
from models.classifier import train_all_models
from config import PROCESSED_DATA_PATH, RAW_DATA_PATH, ENRICHED_DATA_PATH


def main():
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Preprocessing dati...")
        process_and_save(RAW_DATA_PATH, PROCESSED_DATA_PATH, n_samples = 20000)

    try:
        if not os.path.exists(ENRICHED_DATA_PATH):
            print("Applicazione regole Prolog (Reasoning)...")
            df = pd.read_csv(PROCESSED_DATA_PATH)
            kb = KnowledgeBase()
            df_enriched = kb.apply_reasoning(df)
            df_enriched.to_csv(ENRICHED_DATA_PATH, index = False)
            print(f"Dataset salvato: {ENRICHED_DATA_PATH}")
        else:
            print(f"Dataset già presente: {ENRICHED_DATA_PATH}")

    except Exception as e:
        print(f"[ERRORE] Problema con Prolog: {e}")
        sys.exit(1)

    print("Avvio Pipeline Machine Learning...")

    train_all_models(ENRICHED_DATA_PATH)

if __name__ == "__main__":
    main()