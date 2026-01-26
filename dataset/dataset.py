import pandas as pd

class Dataset:
    def __init__(self, path, sep = ','):
        self.path = path
        try:
            self.dataset = pd.read_csv(path, sep = sep, low_memory = False)
        except FileNotFoundError:
            self.dataset = None
            print(f"[Dataset] Errore: File non trovato in {path}")

    def get_dataset(self):
        return self.dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def drop_dataset_columns(self, columns_to_remove):
        """Rimuove colonne se esistono."""
        cols = [c for c in columns_to_remove if c in self.dataset.columns]
        self.dataset = self.dataset.drop(columns = cols)

    def save_dataset(self, path):
        if self.dataset is not None:
            self.dataset.to_csv(path, index = False)
            print(f"[Dataset] Salvato in: {path}")
        else:
            print("[Dataset] Nessun dato da salvare.")