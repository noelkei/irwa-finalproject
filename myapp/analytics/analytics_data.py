import json
import random
import altair as alt
import pandas as pd

# A simple set of stopwords for query term analysis
STOPWORDS = {"the", "and", "for", "of", "to", "a", "an", "in", "on", "is", "it"}

class AnalyticsData:
    """
    In-memory persistence for analytics data.
    Holds search queries, click counts, and other analytics tables.
    """
    def __init__(self):
        # Track document view counts: key = doc_id, value = number of clicks/views
        self.fact_clicks = {}
        # Track search queries: key = search_id, value = query text
        self.search_queries = {}
        # Frequency of search queries (lowercased) for analytics
        self.query_freq = {}
        # Frequency of individual terms across all queries (excluding common stopwords)
        self.term_freq = {}
        # Count of browsers used (by name) for searches
        self.browser_counts = {}
        # (Optional) Count of device types if needed (e.g., Mobile vs Desktop)
        self.device_counts = {}

    def save_query_terms(self, terms: str) -> int:
        """
        Save the search query terms (placeholder implementation).
        Returns a generated search_id for the query.
        """
        # Generate a random search ID for tracking (ensures unique for each query)
        search_id = random.randint(0, 100000)
        return search_id

    def log_search_event(self, search_id: int, query: str, agent_info: dict, user_ip: str):
        """
        Log details of a search event: query text, user agent info, etc.
        """
        # Store the query text by its ID
        self.search_queries[search_id] = query
        # Update query frequency (case-insensitive)
        q_lower = query.lower()
        self.query_freq[q_lower] = self.query_freq.get(q_lower, 0) + 1
        # Update term frequency for terms in the query (exclude common stopwords)
        for term in query.split():
            t = term.lower()
            if t in STOPWORDS or t == "":
                continue
            self.term_freq[t] = self.term_freq.get(t, 0) + 1
        # Update browser usage count
        browser_name = "Unknown"
        try:
            browser_name = agent_info.get("browser", {}).get("name", "Unknown")
        except Exception:
            browser_name = "Unknown"
        self.browser_counts[browser_name] = self.browser_counts.get(browser_name, 0) + 1
        # Update device type count (mobile vs desktop) based on user agent info
        device_type = "Desktop"
        try:
            platform_name = agent_info.get("platform", {}).get("name", "") or ""
        except Exception:
            platform_name = ""
        ua_string = str(platform_name)
        if "Android" in ua_string or "iPhone" in ua_string or "iPad" in ua_string or "Mobile" in ua_string:
            device_type = "Mobile"
        self.device_counts[device_type] = self.device_counts.get(device_type, 0) + 1
        # (We could also log user_ip or other metadata if needed for further analysis)

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


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        # For simplicity, using 'description' field to store some identifying text (e.g., title or description)
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """Print the object content as a JSON string"""
        return json.dumps(self.__dict__)
