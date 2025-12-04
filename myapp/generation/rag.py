import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class RAGGenerator:

    MODE = "improved"   # overwritten by web_app.py at runtime

    # ------------------------------------------------------------------
    # Baseline template (professor)
    # ------------------------------------------------------------------
    BASE_TEMPLATE = """
You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

## Instructions:
1. Identify the single best product that matches the user's request.
2. Present the recommendation clearly in this format:
- Best Product: [Product PID] [Product Name]
- Why: [Explain in plain language why this product is the best fit.]
3. Mention an alternative if relevant.
4. If no product fits, return EXACTLY:
"There are no good products that fit the request based on the retrieved results."

## Retrieved Products:
{retrieved_results}

## User Request:
{user_query}

## Output Format:
- Best Product: ...
- Why: ...
- Alternative (optional): ...
"""

    # ------------------------------------------------------------------
    # Improved template
    # ------------------------------------------------------------------
    IMPROVED_TEMPLATE = """
You are an expert product advisor. Your goal is to recommend the best product from the retrieved results.

Evaluate products based on:
- Title and description relevance to the user query
- Category or sub-category match
- Price attractiveness
- Brand and rating when available

## Instructions:
1. Select the single best product for the user's needs.
2. Explain the reasoning concisely.
3. Optionally mention one close alternative.
4. If none are suitable, return EXACTLY:
"There are no good products that fit the request based on the retrieved results."

## Retrieved Products:
{retrieved_results}

## User Request:
{user_query}

## Output Format:
- Best Product: ...
- Why: ...
- Alternative (optional): ...
"""

    # ------------------------------------------------------------------
    # Detail page template
    # ------------------------------------------------------------------
    DETAIL_TEMPLATE = """
You are an expert product analyst. Provide a short analysis of the product in relation to the user's search query.

## User Query:
{user_query}

## Product:
Title: {title}
Brand: {brand}
Category: {category}
Price: {price}
Rating: {rating}
Description: {description}

## Output:
One concise paragraph evaluating how well this product satisfies the search query.
"""

    # ------------------------------------------------------------------
    # Baseline generation (professor)
    # ------------------------------------------------------------------
    def _generate_baseline(self, user_query, retrieved_results, top_N=20):
        if not retrieved_results:
            return "There are no good products that fit the request based on the retrieved results."

        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            formatted = "\n".join([
                f"- PID: {getattr(r, 'pid', '')}, Title: {getattr(r, 'title', '')}"
                for r in retrieved_results[:top_N]
            ])

            prompt = self.BASE_TEMPLATE.format(
                retrieved_results=formatted,
                user_query=user_query
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

            return resp.choices[0].message.content

        except Exception as e:
            print("Baseline RAG error:", e)
            return "RAG unavailable."

    # ------------------------------------------------------------------
    # Improved generation
    # ------------------------------------------------------------------
    def _generate_improved(self, user_query, retrieved_results, top_N=20):
        if not retrieved_results:
            return "There are no good products that fit the request based on the retrieved results."

        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            formatted = ""
            for r in retrieved_results[:top_N]:
                pid = getattr(r, "pid", "")
                title = getattr(r, "title", "")
                category = getattr(r, "category", "")
                price = getattr(r, "selling_price", None) or getattr(r, "actual_price", None)
                rating = getattr(r, "average_rating", None)
                desc = getattr(r, "description", "")

                snippet = desc[:120].strip() + ("..." if desc and len(desc) > 120 else "")

                formatted += (
                    f"- PID: {pid}, Title: {title}, "
                    f"Category: {category}, Price: {price}, Rating: {rating}, "
                    f"Description: {snippet}\n"
                )

            prompt = self.IMPROVED_TEMPLATE.format(
                retrieved_results=formatted.strip(),
                user_query=user_query
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

            return resp.choices[0].message.content

        except Exception as e:
            print("Improved RAG error:", e)
            return "RAG unavailable."

    # ------------------------------------------------------------------
    # Main unified entrypoint (mode-controlled)
    # ------------------------------------------------------------------
    def generate_response(self, user_query, retrieved_results, top_N=20):
        if RAGGenerator.MODE == "template_rag":
            return self._generate_baseline(user_query, retrieved_results, top_N)
        return self._generate_improved(user_query, retrieved_results, top_N)

    # ------------------------------------------------------------------
    # Detail page generation
    # ------------------------------------------------------------------
    def generate_detail_response(self, user_query, product_doc):
        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            price = (
                f"{product_doc.selling_price:.2f}"
                if getattr(product_doc, "selling_price", None)
                else (
                    f"{product_doc.actual_price:.2f}"
                    if getattr(product_doc, "actual_price", None)
                    else "Unknown"
                )
            )

            desc = product_doc.description or "No description available."
            desc = desc[:300].strip() + ("..." if len(desc) > 300 else "")

            prompt = self.DETAIL_TEMPLATE.format(
                user_query=user_query,
                title=product_doc.title,
                brand=product_doc.brand or "Unknown",
                category=product_doc.category or product_doc.sub_category or "Unknown",
                price=price,
                rating=f"{product_doc.average_rating:.1f}" if product_doc.average_rating else "N/A",
                description=desc,
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

            return resp.choices[0].message.content.strip()

        except Exception as e:
            print("Detail RAG error:", e)
            return "No AI-generated summary available."
