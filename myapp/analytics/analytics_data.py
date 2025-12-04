import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

import altair as alt
import pandas as pd

# A simple set of stopwords for query term analysis
STOPWORDS = {"the", "and", "for", "of", "to", "a", "an", "in", "on", "is", "it"}


@dataclass
class SessionInfo:
    """Represents a user session (star schema: Session table)."""
    session_id: str
    ip: str
    user_agent: str
    start_time: datetime
    last_activity: datetime


@dataclass
class RequestEvent:
    """Represents an HTTP request (star schema: Request table)."""
    timestamp: datetime
    session_id: str
    path: str
    method: str
    query_string: str
    num_terms: int
    user_agent: str
    browser: str
    platform: str
    is_mobile: bool


@dataclass
class ClickEvent:
    """Represents a click on a search result (star schema: Click table)."""
    timestamp: datetime
    session_id: str
    pid: str
    query: str
    rank: Optional[int]
    dwell_time: Optional[float]  # seconds


class AnalyticsData:
    """
    In-memory persistence for analytics data.
    Holds sessions, requests, clicks and also the legacy counters (fact_clicks, query_freq, etc.).
    """

    def __init__(self):
        # Star-schema-like structures
        self.sessions: Dict[str, SessionInfo] = {}        # Session table
        self.requests: List[RequestEvent] = []            # Request table
        self.clicks: List[ClickEvent] = []                # Click table

        # Legacy / aggregated counters (for dashboard and compatibility)
        self.fact_clicks: Dict[str, int] = {}             # doc_id -> clicks/views
        self.search_queries: Dict[int, str] = {}          # search_id -> query text
        self.query_freq: Dict[str, int] = {}              # lowercased query -> count
        self.term_freq: Dict[str, int] = {}               # term -> count
        self.browser_counts: Dict[str, int] = {}          # browser name -> count
        self.device_counts: Dict[str, int] = {}           # "Mobile"/"Desktop" -> count

    # ---------------------------
    # Session management
    # ---------------------------

    def create_session(self, ip: str, user_agent: str) -> str:
        """
        Create a new SessionInfo and return its ID.
        """
        session_id = str(random.randint(0, 10**9))
        now = datetime.utcnow()
        self.sessions[session_id] = SessionInfo(
            session_id=session_id,
            ip=ip,
            user_agent=user_agent,
            start_time=now,
            last_activity=now,
        )
        return session_id

    def touch_session(self, session_id: str):
        """
        Update last_activity for an existing session.
        """
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.utcnow()

    # ---------------------------
    # Search queries
    # ---------------------------

    def save_query_terms(self, terms: str) -> int:
        """
        Generate and store a search_id -> terms mapping.
        (Backward compatible with previous implementation, but now also fills search_queries.)
        """
        search_id = random.randint(0, 10**9)
        self.search_queries[search_id] = terms
        return search_id

    def log_search_event(self, search_id: int, query: str, agent_info: dict, user_ip: str):
        """
        Log details of a search event: query text, user agent info, etc.
        Backward compatible with old signature used in web_app.py.
        Also updates query_freq, term_freq, browser_counts, device_counts.
        """
        # Store query by id (if not already stored by save_query_terms)
        if search_id not in self.search_queries:
            self.search_queries[search_id] = query

        # Update query frequency (case-insensitive)
        q_lower = query.lower()
        self.query_freq[q_lower] = self.query_freq.get(q_lower, 0) + 1

        # Update term frequency for terms in the query (exclude common stopwords)
        for term in query.split():
            t = term.lower().strip()
            if t in STOPWORDS or t == "":
                continue
            self.term_freq[t] = self.term_freq.get(t, 0) + 1

        # Browser and platform from agent_info (keeps compatibility with old code)
        browser_name = "Unknown"
        platform_name = ""
        try:
            browser_name = agent_info.get("browser", {}).get("name", "Unknown")
        except Exception:
            browser_name = "Unknown"
        try:
            platform_name = agent_info.get("platform", {}).get("name", "") or ""
        except Exception:
            platform_name = ""

        # Update browser usage count
        self.browser_counts[browser_name] = self.browser_counts.get(browser_name, 0) + 1

        # Update device type count (Mobile vs Desktop) using platform hint
        # Update device type count (Mobile vs Desktop) using platform hint
        bn = browser_name if isinstance(browser_name, str) else ""
        pn = platform_name if isinstance(platform_name, str) else ""
        ua_string = f"{bn} {pn}".lower()

        device_type = "Mobile" if any(
            kw in ua_string for kw in ["android", "iphone", "ipad", "mobile"]
        ) else "Desktop"

        self.device_counts[device_type] = self.device_counts.get(device_type, 0) + 1

    # ---------------------------
    # HTTP Requests
    # ---------------------------

    def log_http_request(self, session_id: str, path: str, method: str,
                         query_string: str, num_terms: int,
                         user_agent_str: str, browser_name: str, platform_name: str,
                         is_mobile: bool):
        """
        Store an HTTP request event (for /search, /doc_details, etc.).
        This is the Request table in the star schema.
        """
        event = RequestEvent(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            path=path,
            method=method,
            query_string=query_string,
            num_terms=num_terms,
            user_agent=user_agent_str,
            browser=browser_name or "Unknown",
            platform=platform_name or "",
            is_mobile=is_mobile,
        )
        self.requests.append(event)

        # Update browser and device counters (redundant but handy for dashboard)
        self.browser_counts[event.browser] = self.browser_counts.get(event.browser, 0) + 1
        device_type = "Mobile" if is_mobile else "Desktop"
        self.device_counts[device_type] = self.device_counts.get(device_type, 0) + 1

    # ---------------------------
    # Clicks and dwell time
    # ---------------------------

    def register_click(self, session_id: str, pid: str, query: str,
                       rank: Optional[int], dwell_time: Optional[float] = None):
        """
        Register a click on a document (star schema Click table, plus fact_clicks).
        """
        event = ClickEvent(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            pid=pid,
            query=query or "",
            rank=rank,
            dwell_time=dwell_time,
        )
        self.clicks.append(event)

        # Update document view counts (for top clicked docs)
        self.fact_clicks[pid] = self.fact_clicks.get(pid, 0) + 1

    def update_last_click_dwell(self, session_id: str, pid: str, dwell_time: float):
        """
        Update the dwell_time for the last click event for this session and pid.
        """
        for ev in reversed(self.clicks):
            if ev.session_id == session_id and ev.pid == pid and ev.dwell_time is None:
                ev.dwell_time = dwell_time
                break

    # ---------------------------
    # Aggregations for dashboard
    # ---------------------------

    def get_top_clicked_docs(self, top_n: int = 10):
        """
        Returns list of (pid, count) for top clicked documents.
        """
        return sorted(self.fact_clicks.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_top_queries(self, top_n: int = 10):
        """
        Returns list of (query, count).
        """
        return sorted(self.query_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_top_terms(self, top_n: int = 10):
        """
        Returns list of (term, count).
        """
        return sorted(self.term_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_basic_kpis(self) -> Dict[str, float]:
        """
        Returns basic KPIs: total_sessions, total_requests, total_clicks, avg_dwell_time.
        """
        total_sessions = len(self.sessions)
        total_requests = len(self.requests)
        total_clicks = len(self.clicks)
        dwell_values = [c.dwell_time for c in self.clicks if c.dwell_time is not None]
        avg_dwell_time = sum(dwell_values) / len(dwell_values) if dwell_values else 0.0
        return {
            "total_sessions": total_sessions,
            "total_requests": total_requests,
            "total_clicks": total_clicks,
            "avg_dwell_time": avg_dwell_time,
        }

    def get_browser_distribution(self):
        """
        Returns (labels, counts) for browsers.
        """
        labels = list(self.browser_counts.keys())
        counts = [self.browser_counts[b] for b in labels]
        return labels, counts

    def get_device_distribution(self):
        """
        Returns (labels, counts) for device types (Mobile/Desktop).
        """
        labels = list(self.device_counts.keys())
        counts = [self.device_counts[d] for d in labels]
        return labels, counts

    def plot_number_of_views(self):
        """
        Generate an Altair bar chart (as HTML) for number of views per document.
        """
        if len(self.fact_clicks) == 0:
            # No data to plot yet
            return "<p>No views yet.</p>"

        data = [{'Document ID': doc_id, 'Number of Views': count}
                for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        chart = alt.Chart(df).mark_bar().encode(
            x='Document ID:O',  # treat Document ID as ordinal on x-axis
            y='Number of Views:Q'
        ).properties(title='Number of Views per Document')
        return chart.to_html()

    def get_click_rank_distribution(self):
        ranks = [c.rank for c in self.clicks if c.rank is not None]
        if not ranks:
            return [], []
        labels = sorted(list(set(ranks)))
        counts = [ranks.count(r) for r in labels]
        return labels, counts

    def get_dwell_time_distribution(self):
        dwell = [c.dwell_time for c in self.clicks if c.dwell_time is not None]
        if not dwell:
            return [], []
        # bucket dwell times into whole seconds
        buckets = {}
        for d in dwell:
            sec = int(d)
            buckets[sec] = buckets.get(sec, 0) + 1
        labels = sorted(buckets.keys())
        counts = [buckets[k] for k in labels]
        return labels, counts

    def get_clicks_per_hour(self):
        hours = [c.timestamp.hour for c in self.clicks]
        if not hours:
            return [], []
        labels = list(range(24))
        counts = [hours.count(h) for h in labels]
        return labels, counts


class ClickedDoc:
    """
    Simple helper for the stats page (compatible with your existing stats.html),
    not strictly part of the star schema but convenient.
    """
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        return json.dumps(self.__dict__)
