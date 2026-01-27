import os
import pandas as pd

from config import FEATURES

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataset.dataset import Dataset


def build_preprocessor():
    transformers = []

    if FEATURES:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy = 'constant', fill_value = 0)),
            ('scaler', StandardScaler())
        ])
        transformers.append(("num", num_pipeline, FEATURES))

    preprocessor = ColumnTransformer(
        transformers = transformers,
        remainder = 'drop'
    )

    return preprocessor


def process_and_save(raw_path, output_path, n_samples):
    """
    Pulisce il CSV Raw e lo salva in formato leggibile.
    """
    print(f"[Preprocessor] Caricamento raw da: {raw_path}")

    try:
        ds = Dataset(raw_path, sep = '\t')
        df = ds.get_dataset()
    except Exception as e:
        print(f"[Errore] {e}")
        return False

    if df is None: return False

    # Rinomina colonne
    rename_map = {'fruits-vegetables-nuts_100g': 'fruit_veg_100g', 'nutrition_grade_fr': 'grade'}
    df = df.rename(columns = rename_map)

    # Calcolo Target
    if 'grade' not in df.columns:
        print("[Errore] Colonna 'nutrition_grade_fr' mancante.")
        return False

    df = df.dropna(subset = ['grade'])
    df = df.copy()
    df['grade'] = df['grade'].astype(str).str.lower()
    df['target'] = df['grade'].map({'a': 0, 'b': 0, 'c': 1, 'd': 1, 'e': 1})
    df = df.dropna(subset = ['target'])

    # Solo colonne utili + target
    cols_to_keep = [c for c in FEATURES if c in df.columns] + ['target']
    df = df[cols_to_keep]

    # Riempimento base per salvare il CSV senza buchi
    df = df.fillna(0)

    # Bilanciamento
    df = _balance_dataset(df, n_samples)

    # Salvataggio
    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    df.to_csv(output_path, index = False)
    print(f"[Preprocessor] Dataset salvato: {output_path}")
    return True


def _balance_dataset(df, n_samples):
    good = df[df['target'] == 0]
    bad = df[df['target'] == 1]
    n = min(len(good), len(bad), n_samples // 2)
    return pd.concat([good.sample(n, random_state = 42), bad.sample(n, random_state = 42)]).sample(frac = 1,random_state = 42).reset_index(drop = True)