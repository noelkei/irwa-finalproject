# IRWA Final Project – Search Engine, RAG & Web Analytics

This repository contains the full implementation of the **Information Retrieval and Web Analytics (IRWA)** final project.  
It includes:

- A working **Flask-based search engine**
- **Retrieval algorithms** (TF-IDF with stemming, stopword removal, query expansion & title boosting)
- A **fully improved RAG system** with an optional fallback to the **baseline RAG template**
- **Analytics tracking** (sessions, requests, ranking clicks, dwell time)
- A **dashboard** with KPIs and visual charts

---

## 📦 Project Structure



/myapp
/search
/analytics
/generation
/templates
/static
/data
web_app.py
requirements.txt
.env (ignored)
README.md


---

## 📥 Dataset

Place the instructor-provided file:



data/fashion_products_dataset.json


The system will:

1. Attempt to load the JSON  
2. Normalize price fields  
3. If JSON fails, fall back to:



data/fashion_products_dataset_clean.csv

Here is the **clean, final README section in pure Markdown** — **NOT inside a code block**, ready to paste directly into your README.md.

---

## 🧪 Virtual Environment Setup

### 1. Create venv

```bash
virtualenv irwa_venv
```

### 2. Activate it

**Mac/Linux**

```bash
source irwa_venv/bin/activate
```

**Windows**

```bash
irwa_venv\Scripts\activate.bat
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```
SECRET_KEY=your-secret-key
DEBUG=True
DATA_FILE_PATH=data/fashion_products_dataset.json

GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama-3.1-8b-instant
```

`.env` is ignored by git and must **never** be uploaded.

---

## 🚀 Running the Web App

From the project root:

### ▶️ Default (uses improved RAG)

```bash
python web_app.py
```

### ▶️ Force improved RAG

```bash
python web_app.py --rag-mode=improved
```

### ▶️ Use professor template RAG

```bash
python web_app.py --rag-mode=template
```

If no flag is used → **improved RAG is the default**.

---

## 🔍 Search Engine Features

* TF-IDF retrieval with:

  * synonym-based query expansion
  * stopword removal
  * stemming
  * exact-title boosting
* Rank position tracking passed to analytics
* Optimized preprocessing for ~28k products

---

## 🤖 RAG (Retrieval-Augmented Generation)

Two interchangeable systems are available:

### 1. Improved RAG (default)

Uses:

* extended metadata (brand, category, price, rating)
* description snippets
* refined prompts
* clearer structure & reasoning
* graceful handling of *“no good products”*

### 2. Template RAG (professor’s version)

Matches the original boilerplate functionality exactly.

Select via command line:

```bash
python web_app.py --rag-mode=template
python web_app.py --rag-mode=improved
```

---

## 📊 Web Analytics

The system tracks:

### ✔ Sessions

* session id
* IP
* user agent
* timestamps

### ✔ HTTP Request Logging

* path
* method
* query string
* number of terms
* browser & device detection

### ✔ Click Logging

* document clicked
* rank of the document in results
* query used
* dwell time (time before returning)

### ✔ Query Statistics

* query frequency
* term frequency

### ✔ Dashboard Visualizations

Accessible at:

```
/dashboard
/stats
```

Includes:

* Total sessions
* Total clicks
* Total requests
* Avg dwell time
* Top clicked products
* Most frequent queries
* Top terms
* Browser distribution (Chart.js)
* Device distribution
* Rank distribution
* Dwell time histogram
* Clicks per hour
* Document view bar chart (Altair)

---

## 📝 Notes for Evaluators

All required features from **Part 4** are implemented:

* Complete user interface
* Retrieval algorithms
* Improved RAG + baseline RAG
* Full analytics (session, request, clicks, dwell time)
* Dashboard with charts

To run the project, the evaluator only needs to:

1. Place the dataset in the `data/` folder
2. Add their personal `.env` with the GROQ API key

Everything works out of the box.
