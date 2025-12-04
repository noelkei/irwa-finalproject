import os
from json import JSONEncoder

import httpagentparser  # for parsing the user agent string
from flask import Flask, render_template, session, request
from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

# Enable custom JSON serialization for our Pydantic models (Document, etc.)
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default

# Instantiate Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "IRWA_SECRET_KEY")
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME", "IRWA_SEARCH_ENGINE")

# Instantiate core components
search_engine = SearchEngine()
analytics_data = AnalyticsData()
rag_generator = RAGGenerator()

# Load documents corpus into memory
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
data_file_path = os.path.join(path, os.getenv("DATA_FILE_PATH", "data/fashion_products_dataset.json"))
corpus = load_corpus(data_file_path)
print("\nCorpus is loaded. Number of documents:", len(corpus))
# Print first document for verification
first_doc = list(corpus.values())[0] if corpus else None
print("First document sample:", first_doc)

# Routes

@app.route('/')
def index():
    # Home page with search form
    user_agent = request.headers.get('User-Agent')
    user_ip = request.remote_addr
    agent_info = httpagentparser.detect(user_agent)
    print("Home page accessed. User Agent:", user_agent)
    print("Parsed agent info:", agent_info, "| IP:", user_ip)
    # (We do not log analytics here for page load; only on search and clicks)
    return render_template('index.html', page_title="Welcome")

@app.route('/search', methods=['POST'])
def search_form_post():
    # Handle the search form submission
    search_query = request.form.get('search-query', "")
    session['last_search_query'] = search_query  # store last query in session
    # Save query in analytics and get a search ID
    search_id = analytics_data.save_query_terms(search_query)
    # Log additional analytics for this search (user agent, etc.)
    agent_info = httpagentparser.detect(request.headers.get('User-Agent'))
    analytics_data.log_search_event(search_id, search_query, agent_info, request.remote_addr)
    # Perform search
    results = search_engine.search(search_query, search_id, corpus)
    found_count = len(results)
    session['last_found_count'] = found_count
    # Generate RAG-based summary of results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)
    print("Search results found:", found_count)
    return render_template('results.html', results_list=results, page_title="Results",
                           found_counter=found_count, rag_response=rag_response)

@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Document details page – shows the details of a clicked document and logs the click.
    """
    clicked_doc_id = request.args.get('pid')
    search_id = request.args.get('search_id')
    if not clicked_doc_id or clicked_doc_id not in corpus:
        # Invalid ID, return to home or 404
        return render_template('doc_details.html', page_title="Document Details")
    # Log the click in analytics (increment view count for this doc)
    if clicked_doc_id in analytics_data.fact_clicks:
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1
    print(f"Document {clicked_doc_id} clicked. Total views = {analytics_data.fact_clicks[clicked_doc_id]}")
    # Retrieve the document object
    doc = corpus[clicked_doc_id]
    # Try to retrieve the original query for context (if available)
    user_query = None
    try:
        if search_id is not None:
            user_query = analytics_data.search_queries.get(int(search_id))
    except Exception as e:
        print("Error retrieving query for search_id:", e)
    # Generate an AI summary for the detail page (if query and RAG available)
    rag_detail = None
    if user_query:
        rag_detail = rag_generator.generate_detail_response(user_query, doc)
        print("RAG detail response:", rag_detail)
    return render_template('doc_details.html', doc=doc, rag_detail=rag_detail, page_title="Document Details")

@app.route('/stats', methods=['GET'])
def stats():
    """
    Simple statistics page (lists documents by number of clicks).
    """
    docs_stats = []
    for doc_id, count in analytics_data.fact_clicks.items():
        if doc_id in corpus:
            d = corpus[doc_id]
            stats_doc = StatsDocument(pid=d.pid, title=d.title, description=d.description, url=d.url, count=count)
            docs_stats.append(stats_doc)
    # Sort by click count descending
    docs_stats.sort(key=lambda doc: doc.count or 0, reverse=True)
    return render_template('stats.html', clicks_data=docs_stats, page_title="Statistics")

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """
    Dashboard page showing analytics insights.
    """
    visited_docs = []
    for doc_id, count in analytics_data.fact_clicks.items():
        if doc_id in corpus:
            d = corpus[doc_id]
            # Use document title for a succinct description in the dashboard
            visited_docs.append(ClickedDoc(doc_id, d.title, count))
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)
    # Prepare top search queries data
    top_queries = sorted(analytics_data.query_freq.items(), key=lambda x: x[1], reverse=True)
    if len(top_queries) > 5:
        top_queries = top_queries[:5]
    # Prepare top query terms data
    top_terms = sorted(analytics_data.term_freq.items(), key=lambda x: x[1], reverse=True)
    if len(top_terms) > 5:
        top_terms = top_terms[:5]
    # Prepare browser usage data for chart
    browser_labels = list(analytics_data.browser_counts.keys())
    browser_counts = list(analytics_data.browser_counts.values())
    return render_template('dashboard.html', visited_docs=visited_docs,
                           top_queries=top_queries, top_terms=top_terms,
                           browser_labels=browser_labels, browser_counts=browser_counts,
                           page_title="Dashboard")

@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    """
    Returns an HTML snippet with an embedded chart of document view counts.
    This is used in an <iframe> on the dashboard page.
    """
    return analytics_data.plot_number_of_views()

if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG", "False"))
