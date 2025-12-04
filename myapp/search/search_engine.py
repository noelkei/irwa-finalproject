import numpy as np
from myapp.search.objects import Document
from myapp.search import algorithms  # Import algorithms module

class SearchEngine:
    """Implements the search engine logic using various search algorithms."""
    def search(self, search_query: str, search_id: int, corpus: dict):
        print("Search query:", search_query)
        # Perform the search using the implemented algorithm
        results = algorithms.search_in_corpus(search_query, corpus, search_id)
        # Convert results to list of Document or ResultItem objects if needed
        # Here, results is a list of dict or Pydantic models; ensure consistent output
        final_results = []
        for item in results:
            # If the algorithm returned a dict (fallback or default), convert to Document for consistency
            if isinstance(item, dict):
                # We use Document model to leverage existing fields (unknown fields like ranking will be ignored by BaseModel)
                try:
                    res_doc = Document(**item)
                except Exception:
                    # In case Document cannot take ranking, remove it and retry
                    item_copy = item.copy()
                    item_copy.pop("ranking", None)
                    res_doc = Document(**item_copy)
                final_results.append(res_doc)
            else:
                # If already a Document/ResultItem (Pydantic model), just append
                final_results.append(item)
        return final_results
