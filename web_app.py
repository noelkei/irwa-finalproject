import os
import sys
import time
from flask import Flask, request, render_template, redirect, url_for, session
from dotenv import load_dotenv

from myapp.search.load_corpus import load_corpus
from myapp.search.search_engine import SearchEngine
from myapp.analytics.analytics_data import AnalyticsData
from myapp.generation.rag import RAGGenerator
from myapp.search.objects import StatsDocument

load_dotenv()

# --------------------------
# RAG MODE SELECTOR
# --------------------------
mode = "improved"  # default

# Accept BOTH styles:
# python web_app.py template_rag
# python web_app.py --rag-mode=template_rag
for arg in sys.argv[1:]:
    if arg.startswith("--rag-mode="):
        mode = arg.split("=", 1)[1].strip().lower()
    elif arg.lower() in ("template_rag", "template", "baseline", "professor", "improved"):
        mode = arg.lower()

# Normalize modes
if mode in ("template_rag", "baseline", "professor", "template"):
    RAGGenerator.MODE = "template_rag"
    print("➡️  Using BASELINE RAG (professor version)")
else:
    RAGGenerator.MODE = "improved"
    print("➡️  Using IMPROVED RAG")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-key")

DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "data/fashion_products_dataset.json")

# Load corpus
corpus = load_corpus(DATA_FILE_PATH)
print("Corpus loaded:", len(corpus))

search_engine = SearchEngine()
analytics = AnalyticsData()
rag = RAGGenerator()


def detect_browser(ua_string):
    ua = ua_string.lower()
    if "safari" in ua and "chrome" not in ua:
        return "Safari"
    if "chrome" in ua and "chromium" not in ua:
        return "Chrome"
    if "firefox" in ua:
        return "Firefox"
    if "edge" in ua:
        return "Edge"
    if "opera" in ua or "opr" in ua:
        return "Opera"
    return "Unknown"


def get_or_create_session_id():
    sid = session.get("session_id")
    if not sid:
        sid = analytics.create_session(
            ip=request.remote_addr or "",
            user_agent=request.user_agent.string or ""
        )
        session["session_id"] = sid
    else:
        analytics.touch_session(sid)
    return sid


@app.route("/")
def index():
    return render_template("index.html", page_title="Home")


@app.route("/search", methods=["GET", "POST"])
def search():
    session_id = get_or_create_session_id()

    # Query retrieval
    query = request.form.get("search-query", "").strip() if request.method == "POST" else request.args.get("q", "").strip()
    if not query:
        return redirect(url_for("index"))

    # --- Dwell time ---
    last_click_time = session.pop("last_click_time", None)
    last_click_pid = session.pop("last_click_pid", None)
    if last_click_time and last_click_pid:
        analytics.update_last_click_dwell(session_id, last_click_pid, time.time() - last_click_time)

    # Save query
    search_id = analytics.save_query_terms(query)

    # Log search
    agent_info = {
        "browser": {"name": request.user_agent.browser},
        "platform": {"name": request.user_agent.platform},
    }
    analytics.log_search_event(search_id, query, agent_info, request.remote_addr)

    # Log request
    ua_str = request.user_agent.string or ""
    browser = detect_browser(ua_str)
    analytics.log_http_request(
        session_id=session_id,
        path=request.path,
        method=request.method,
        query_string=request.query_string.decode() if hasattr(request.query_string, "decode") else "",
        num_terms=len(query.split()),
        user_agent_str=ua_str,
        browser_name=browser,
        platform_name=request.user_agent.platform or "",
        is_mobile="mobile" in ua_str.lower()
    )

    # Execute search
    results = search_engine.search(query, search_id, corpus)

    # AI summary (RAG, depending on selected mode)
    rag_response = rag.generate_response(query, results)

    return render_template("results.html",
                           results_list=results,
                           results_count=len(results),
                           query=query,
                           search_id=search_id,
                           rag_response=rag_response,
                           page_title="Results")


@app.route("/doc_details")
def doc_details():
    session_id = get_or_create_session_id()

    pid = request.args.get("pid")
    search_id = request.args.get("search_id", type=int)
    rank = request.args.get("rank", type=int)

    if pid not in corpus:
        return render_template("doc_details.html", doc=None, rag_detail=None)

    doc = corpus[pid]
    query = analytics.search_queries.get(search_id, "")

    # Log request
    ua_str = request.user_agent.string or ""
    browser = detect_browser(ua_str)
    analytics.log_http_request(
        session_id=session_id,
        path=request.path,
        method=request.method,
        query_string=request.query_string.decode() if hasattr(request.query_string, "decode") else "",
        num_terms=len(query.split()) if query else 0,
        user_agent_str=ua_str,
        browser_name=browser,
        platform_name=request.user_agent.platform or "",
        is_mobile="mobile" in ua_str.lower()
    )

    # Register click
    analytics.register_click(session_id, pid, query, rank)

    # Start dwell timer
    session["last_click_time"] = time.time()
    session["last_click_pid"] = pid

    # Detail page RAG
    rag_detail = rag.generate_detail_response(query, doc)

    return render_template("doc_details.html",
                           doc=doc,
                           rag_detail=rag_detail,
                           page_title="Product")


@app.route("/dashboard")
def dashboard():
    visited_docs = []
    for pid, count in analytics.get_top_clicked_docs(10):
        d = corpus.get(pid)
        title = d.title if d else pid
        desc = (d.description[:150] + "...") if d and d.description else ""
        visited_docs.append({"pid": pid, "count": count, "description": title + " — " + desc})

    click_rank_labels, click_rank_counts = analytics.get_click_rank_distribution()
    dwell_labels, dwell_counts = analytics.get_dwell_time_distribution()
    click_hour_labels, click_hour_counts = analytics.get_clicks_per_hour()

    return render_template("dashboard.html",
                           visited_docs=visited_docs,
                           top_queries=analytics.get_top_queries(10),
                           top_terms=analytics.get_top_terms(10),
                           browser_labels=analytics.get_browser_distribution()[0],
                           browser_counts=analytics.get_browser_distribution()[1],
                           device_labels=analytics.get_device_distribution()[0],
                           device_counts=analytics.get_device_distribution()[1],
                           kpis=analytics.get_basic_kpis(),

                           click_rank_labels=click_rank_labels,
                           click_rank_counts=click_rank_counts,
                           dwell_labels=dwell_labels,
                           dwell_counts=dwell_counts,
                           click_hour_labels=click_hour_labels,
                           click_hour_counts=click_hour_counts,

                           page_title="Dashboard")


@app.route("/plot_number_of_views")
def plot_number_of_views():
    return analytics.plot_number_of_views()


@app.route("/stats")
def stats():
    entries = []
    for pid, count in analytics.get_top_clicked_docs(50):
        d = corpus.get(pid)
        if d:
            entries.append(
                StatsDocument(
                    pid=pid,
                    title=d.title,
                    description=d.description[:300] if d.description else "",
                    url=d.url,
                    count=count,
                )
            )
    return render_template("stats.html", clicks_data=entries, page_title="Stats")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8088, debug=True)
