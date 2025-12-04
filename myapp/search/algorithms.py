import math
import numpy as np

# Try TF-IDF as fallback if BM25 indexing fails
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None


# ------------------------------------------------------------
# LIGHT STOPWORD LIST
# ------------------------------------------------------------

STOPWORDS = {
    "the", "and", "for", "of", "to", "a", "an", "in", "on",
    "is", "it", "this", "that", "with", "at", "from", "by"
}


# ------------------------------------------------------------
# VERY LIGHTWEIGHT STEMMER (Porter-like)
# ------------------------------------------------------------

def simple_stem(word):
    """
    A minimal stemming function.
    Not a full Porter stemmer but works well for plurals and common forms.
    """
    if len(word) <= 3:
        return word

    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ing"):
        return word[:-3]
    if word.endswith("ed"):
        return word[:-2]
    if word.endswith("s") and len(word) > 4:
        return word[:-1]

    return word


# ------------------------------------------------------------
# SYNONYMS — improved recall for fashion search
# ------------------------------------------------------------

SYNONYMS = {
    # Prices / Adjectives
    "cheap": ["affordable", "inexpensive", "low cost", "budget", "low-price"],
    "affordable": ["cheap", "budget"],
    "expensive": ["premium", "high-end", "luxury", "pricey"],
    "premium": ["luxury", "high-end", "expensive"],
    "comfortable": ["comfy", "soft", "relaxed"],
    "elegant": ["stylish", "classy", "refined"],
    "sport": ["sportswear", "athletic", "fitness"],
    "casual": ["everyday", "relaxed", "streetwear"],
    "formal": ["dressy", "smart", "office", "elegant"],

    # Colors
    "red": ["maroon", "crimson", "burgundy"],
    "blue": ["navy", "sky blue", "royal blue"],
    "green": ["olive", "mint", "emerald"],
    "black": ["dark", "charcoal"],
    "white": ["ivory", "cream"],
    "yellow": ["gold", "mustard"],
    "pink": ["rose", "blush"],
    "brown": ["tan", "beige", "camel"],

    # Shoes
    "shoes": ["shoe", "sneakers", "footwear", "trainers", "running shoes"],
    "shoe": ["shoes", "sneaker", "footwear"],
    "sneakers": ["running shoes", "trainers", "sports shoes"],
    "boots": ["ankle boots", "combat boots", "boot"],

    # Bottomwear
    "pants": ["trousers", "bottoms", "slacks"],
    "trousers": ["pants", "bottomwear"],
    "jeans": ["denim", "denims", "skinny jeans", "slim jeans"],

    # Tops
    "tshirt": ["t-shirt", "tee", "tees", "tshirts"],
    "t-shirt": ["tee", "tshirt", "shirt"],
    "shirt": ["top", "blouse"],

    # Dresses
    "dress": ["gown", "one-piece", "maxi dress", "midi dress"],

    # Outerwear
    "jacket": ["coat", "outerwear", "blazer"],

    # Accessories
    "bag": ["handbag", "purse", "tote"],

    # Gender
    "women": ["woman", "ladies", "female"],
    "men": ["man", "male", "mens"],
    "kids": ["children", "child", "boy", "girl"],
}


# ------------------------------------------------------------
# TOKENIZER + PREPROCESSING
# ------------------------------------------------------------

def preprocess_text(text: str):
    """
    Tokenizes, removes stopwords, applies stemming.
    """
    tokens = []
    for w in text.lower().split():
        w = w.strip()
        if not w or w in STOPWORDS:
            continue
        tokens.append(simple_stem(w))
    return tokens


# ------------------------------------------------------------
# BM25 Index (precomputed once)
# ------------------------------------------------------------

_bm25_ready = False
_doc_ids = []
_doc_texts = []
_avgdl = 0
_k1 = 1.5
_b = 0.75
_freqs = []
_doc_lengths = []


def _build_bm25_index(corpus):
    global _bm25_ready, _doc_ids, _doc_texts, _freqs, _doc_lengths, _avgdl

    if _bm25_ready:
        return

    _doc_ids = []
    _doc_texts = []
    _freqs = []
    _doc_lengths = []

    for doc_id, doc in corpus.items():
        text_parts = [
            doc.title or "",
            doc.description or "",
            doc.brand or "",
            doc.category or "",
            doc.sub_category or "",
        ]
        joined = " ".join(map(str, text_parts))
        tokens = preprocess_text(joined)

        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        _doc_ids.append(doc_id)
        _doc_texts.append(tokens)
        _freqs.append(freq)
        _doc_lengths.append(len(tokens))

    _avgdl = sum(_doc_lengths) / max(len(_doc_lengths), 1)
    _bm25_ready = True


def _bm25_score(query_terms, index):
    freq = _freqs[index]
    dl = _doc_lengths[index]
    score = 0

    N = len(_doc_ids)

    for term in query_terms:
        if term not in freq:
            continue

        f = freq[term]
        n_qi = sum(1 for d in _freqs if term in d)

        idf = math.log((N - n_qi + 0.5) / (n_qi + 0.5) + 1)
        denom = f + _k1 * (1 - _b + _b * dl / _avgdl)
        score += idf * ((f * (_k1 + 1)) / denom)

    return score


# ------------------------------------------------------------
# MAIN SEARCH FUNCTION
# ------------------------------------------------------------

def search_in_corpus(query: str, corpus: dict, search_id: int, num_results: int = 20):
    """
    BM25 Search Engine with:
    - synonym expansion
    - stopword removal
    - stemming
    - exact title match boosting
    """

    # Build expanded query
    expanded_query = query.lower()
    for w in query.split():
        wl = w.lower()
        if wl in SYNONYMS:
            expanded_query += " " + " ".join(SYNONYMS[wl])

    query_terms = preprocess_text(expanded_query)

    # Build BM25 index if not ready
    _build_bm25_index(corpus)

    scores = []
    for idx, doc_id in enumerate(_doc_ids):
        base_score = _bm25_score(query_terms, idx)
        if base_score <= 0:
            continue

        doc = corpus[doc_id]

        # -------------------------------
        # BOOST: exact title match (+5%)
        # -------------------------------
        title_words = preprocess_text(doc.title or "")
        if any(t in title_words for t in query_terms):
            base_score *= 1.05

        scores.append((base_score, doc_id))

    # Sort results
    scores.sort(reverse=True, key=lambda x: x[0])
    scores = scores[:num_results]

    results = []
    for rank_pos, (score, doc_id) in enumerate(scores, start=1):
        doc = corpus[doc_id]
        results.append({
            "pid": doc.pid,
            "title": doc.title,
            "description": doc.description,
            "ranking": float(score),
            "rank_position": rank_pos,
            "url": f"doc_details?pid={doc.pid}&search_id={search_id}&rank={rank_pos}"
        })

    return results
