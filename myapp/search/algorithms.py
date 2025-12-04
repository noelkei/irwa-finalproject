import re
import numpy as np
# Attempt to import scikit-learn's TfidfVectorizer for advanced search; fallback to simple search if unavailable
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

# Global variables for the index (to avoid rebuilding on each query)
_vectorizer = None
_doc_vectors = None
_doc_ids = None

# Simple synonym mapping for query expansion (for improved recall)
SYNONYMS = {
    "cheap": ["affordable", "inexpensive", "low cost", "low-cost", "budget"],
    "expensive": ["costly", "pricey", "high-end"],
    "shoes": ["shoe", "sneakers", "footwear"],
    "shoe": ["shoes", "sneaker", "footwear"],
    "sneakers": ["shoes", "running shoes", "sneakers"],
    "pants": ["trousers"],
    "trousers": ["pants"],
}

def search_in_corpus(query: str, corpus: dict, search_id: int, num_results: int = 20):
    """
    Search for the query in the corpus and return a ranked list of result items.
    Uses TF-IDF vector space model for ranking (falls back to simple keyword matching if needed).
    """
    global _vectorizer, _doc_vectors, _doc_ids

    # Expand query with synonyms (improved query expansion)
    expanded_query = query
    for word in query.split():
        wl = word.lower()
        if wl in SYNONYMS:
            # Append synonyms to the query string (space-separated)
            expanded_query += " " + " ".join(SYNONYMS[wl])

    # If TfidfVectorizer is available, use vector space model for search
    if TfidfVectorizer:
        # Build index on first run
        if _vectorizer is None or _doc_vectors is None or _doc_ids is None:
            _doc_ids = []
            docs_texts = []
            for doc_id, doc in corpus.items():
                # Combine relevant text fields: title, description, brand, category, sub_category
                text_fields = []
                text_fields.append(doc.title if doc.title else "")
                text_fields.append(doc.description if doc.description else "")
                if getattr(doc, "brand", None):
                    text_fields.append(doc.brand)
                if getattr(doc, "category", None):
                    text_fields.append(doc.category)
                if getattr(doc, "sub_category", None):
                    text_fields.append(doc.sub_category)
                # Join all fields into one document text
                doc_text = " ".join(str(t) for t in text_fields)
                docs_texts.append(doc_text)
                _doc_ids.append(doc_id)
            # Initialize and fit TF-IDF vectorizer on the corpus texts
            _vectorizer = TfidfVectorizer(stop_words='english')
            _doc_vectors = _vectorizer.fit_transform(docs_texts)
        # Transform the (expanded) query into vector space
        query_vec = _vectorizer.transform([expanded_query])
        # Compute similarity scores (cosine similarity since TF-IDF vectors are L2-normalized by default)
        # _doc_vectors is shape (N_docs, N_terms), query_vec is (1, N_terms)
        scores = (_doc_vectors * query_vec.T).toarray().ravel()
        # Sort document indices by score (highest first)
        ranked_indices = np.argsort(scores)[::-1]
        # Collect top results with a positive score
        results = []
        for idx in ranked_indices[:num_results]:
            if scores[idx] <= 0:
                break  # stop at first non-positive score (no more relevant docs)
            doc_id = _doc_ids[idx]
            doc = corpus[doc_id]
            # Construct result item with necessary fields
            results.append({
                "pid": doc.pid,
                "title": doc.title,
                "description": doc.description,
                "url": f"doc_details?pid={doc.pid}&search_id={search_id}",
                "ranking": float(scores[idx])
            })
        return results

    # Fallback: If TfidfVectorizer is not available, use a simple keyword matching approach
    results = []
    query_terms = [t.lower() for t in expanded_query.split() if t != ""]
    for doc_id, doc in corpus.items():
        text = f"{doc.title} {doc.description}".lower()
        # Count occurrences of each query term in the combined text
        score = 0
        for term in query_terms:
            if term in text:
                # Add simple weight = count of term occurrences (or just presence as 1)
                score += text.count(term)
        if score > 0:
            results.append({
                "pid": doc.pid,
                "title": doc.title,
                "description": doc.description,
                "url": f"doc_details?pid={doc.pid}&search_id={search_id}",
                "ranking": float(score)
            })
    # Sort results by the naive score and return top N
    results.sort(key=lambda x: x["ranking"], reverse=True)
    return results[:num_results]
