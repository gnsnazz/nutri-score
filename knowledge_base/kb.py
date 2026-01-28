import re
import pandas as pd

from pyswip import Prolog
from config import FACTS_PATH, RULES_PATH

class KnowledgeBase:
    def __init__(self):
        self.prolog = Prolog()

        print(f"[KB] Caricamento Facts da: {FACTS_PATH}")
        self.prolog.consult(FACTS_PATH.replace("\\", "/"))
        print(f"[KB] Caricamento Rules da: {RULES_PATH}")
        self.prolog.consult(RULES_PATH.replace("\\", "/"))

    def _clean_text(self, text: str) -> str:
        """Pulisce il testo per renderlo un atomo Prolog valido."""
        if pd.isna(text): return "unknown"
        # Rimuove tutto ciò che non è alfanumerico o spazio
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
        return clean.lower()

    def _query_safe(self, query_str: str):
        """Esegue query Prolog con gestione errori."""
        try:
            result = list(self.prolog.query(query_str))
            return 1 if result else 0
        except:
            return 0

    def apply_reasoning(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"[KB] Avvio ragionamento su {len(df)} prodotti...")

        new_cols = {
            'is_empty_calories': [],
            'is_hidden_sodium': [],
            'is_hyper_processed': [],
            'is_low_fat_sugar_trap': [],
            'is_hyperpalatable': [],
            'violates_expectations': [],
            'is_functionally_inconsistent': [],
            'is_ultra_processed_combo': [],
            'symbolic_score': []
        }

        query = self.prolog.query

        for _, row in df.iterrows():
            # Estrazione valori
            s = row.get('sugars_100g', 0)
            f = row.get('fiber_100g', 0)
            salt = row.get('salt_100g', 0)
            fv = row.get('fruit_veg_100g', 0)
            a = row.get('additives_n', 0)
            p = row.get('proteins_100g', 0)
            fat = row.get('fat_100g', 0)
            c = row.get('carbohydrates_100g', 0)

            clean_name = self._clean_text(row.get('product_name', ''))
            p_name = f"'{clean_name}'"

            new_cols['is_empty_calories'].append(
                self._query_safe(f"is_empty_calories({s}, {f})"))
            new_cols['is_hidden_sodium'].append(
                self._query_safe(f"is_hidden_sodium({salt}, {fv})"))
            new_cols['is_hyper_processed'].append(
                self._query_safe(f"is_hyper_processed({a}, {s}, {salt})"))
            new_cols['is_low_fat_sugar_trap'].append(
                self._query_safe(f"is_low_fat_sugar_trap({fat}, {s})"))
            new_cols['is_hyperpalatable'].append(
                self._query_safe(f"is_hyperpalatable({s}, {fat})"))
            new_cols['violates_expectations'].append(
                self._query_safe(f"violates_category_expectation({p_name}, _, _)"))
            new_cols['is_functionally_inconsistent'].append(
                self._query_safe(f"is_functionally_inconsistent({p_name}, {p}, {s})"))
            new_cols['is_ultra_processed_combo'].append(
                self._query_safe(f"is_ultra_processed_combo({a}, {s}, {salt})"))

            try:
                query_str = f"compute_risk_score({s}, {fat}, {salt}, {f}, {fv}, {a}, {p}, {c}, {p_name}, TotalScore)"
                res = list(query(query_str))
                score = res[0]['TotalScore'] if res else 0
            except:
                score = 0
            new_cols['symbolic_score'].append(score)

        # Concatenazione ottimizzata
        df_kb = pd.DataFrame(new_cols)
        df_final = pd.concat([df.reset_index(drop = True), df_kb], axis = 1)

        print("[KB] Ragionamento completato.")
        return df_final