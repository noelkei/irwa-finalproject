import os
import pandas as pd
from myapp.search.objects import Document
from typing import Dict

def load_corpus(path: str) -> Dict[str, Document]:
    """
    Load the corpus file (JSON or CSV) and return a dictionary of Documents.
    Falls back to CSV if JSON file not found or fails to load.
    """
    df = None
    # Try loading the primary dataset (JSON or CSV as given)
    try:
        if path.lower().endswith(".json"):
            df = pd.read_json(path)
        elif path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            # If no extension or unknown, attempt JSON first
            df = pd.read_json(path)
    except Exception as e:
        print(f"Primary data file load failed: {e}")
        # Determine fallback path: if original was JSON, use corresponding CSV name
        fallback_path = path
        if path.lower().endswith(".json"):
            # e.g., "fashion_products_dataset.json" -> "fashion_products_clean.csv"
            fallback_path = path.rsplit(".", 1)[0] + "_clean.csv"
        else:
            # If primary was CSV and failed, try JSON as fallback
            fallback_path = path.rsplit(".", 1)[0] + ".json"
        print(f"Attempting to load fallback data file: {fallback_path}")
        try:
            if fallback_path.lower().endswith(".csv"):
                df = pd.read_csv(fallback_path)
            elif fallback_path.lower().endswith(".json"):
                df = pd.read_json(fallback_path)
        except Exception as e2:
            print(f"Fallback data file load failed: {e2}")
            raise e2  # Re-raise exception if both attempts fail
    # Build Document objects corpus from DataFrame
    corpus = _build_corpus(df)
    return corpus

def _build_corpus(df: pd.DataFrame) -> Dict[str, Document]:
    """
    Build corpus dictionary from a pandas DataFrame.
    Keys are document IDs, values are Document objects.
    """
    corpus = {}
    for _, row in df.iterrows():
        doc = Document(**row.to_dict())
        corpus[doc.pid] = doc
    return corpus
