import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env (for GROQ_API_KEY, GROQ_MODEL, etc.)

class RAGGenerator:
    # Base prompt template for recommending the best product
    PROMPT_TEMPLATE = """
You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

## Instructions:
1. Identify the single best product that matches the user's request.
2. Present the recommendation clearly in this format:
- Best Product: [Product PID] [Product Name]
- Why: [Explain why this product is the best fit, referring to specific attributes like price, features, quality, or fit to user’s needs.]
3. If there is another product that could also work, mention it briefly as an alternative.
4. If no product is a good fit, return ONLY this exact phrase:
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

    # An improved prompt template for detailed analysis of a single product in context of the user query
    PROMPT_TEMPLATE_DETAIL = """
You are an expert product advisor. A user searched for something and is now viewing a specific product.

## Task:
Given the user's search query and the details of the product they are viewing, provide a concise analysis of this product. Focus on how well it meets the user's needs, mentioning key features, price, brand, or quality as relevant. If the product might not fully satisfy the query, politely mention any shortcomings.

## User Query:
{user_query}

## Product Details:
Title: {title}
Brand: {brand}
Category: {category}
Price: {price}
Rating: {rating}
Description: {description}

## Output:
Provide a short paragraph summarizing the product and its suitability for the user's query.
"""

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20):
        """
        Generate an AI response (recommendation) using the retrieved search results.
        Returns the generated response text (or a default message if unavailable).
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
        if retrieved_results is None or len(retrieved_results) == 0:
            # If no results were retrieved, return the "no good products" message directly
            return "There are no good products that fit the request based on the retrieved results."
        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            # Use preferred model from env or default to a fast smaller model
            primary_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
            model_name = primary_model
            # Prepare the retrieved results list for the prompt.
            # Improved: include a short snippet of description for more context (truncated to 100 chars).
            formatted_results = ""
            for res in retrieved_results[:top_N]:
                # Each result can be a Document/ResultItem object or dict
                pid = getattr(res, "pid", res.get("pid") if isinstance(res, dict) else "")
                title = getattr(res, "title", res.get("title") if isinstance(res, dict) else "")
                # If description available, include first 100 characters
                desc = getattr(res, "description", res.get("description") if isinstance(res, dict) else "")
                if desc:
                    desc_snippet = str(desc)[:100].strip()
                    if len(str(desc)) > 100:
                        desc_snippet += "..."
                else:
                    desc_snippet = ""
                formatted_results += f"- PID: {pid}, Title: {title}"
                if desc_snippet:
                    formatted_results += f", Description: {desc_snippet}"
                formatted_results += "\n"
            prompt = self.PROMPT_TEMPLATE.format(retrieved_results=formatted_results.strip(), user_query=user_query)
            # Attempt generation with the primary model, fallback to a smaller model if needed
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                )
            except Exception as e:
                print(f"Primary model {model_name} failed or not available, error: {e}")
                if model_name != "llama-3.1-8b-instant":
                    # Fallback to default smaller model
                    model_name = "llama-3.1-8b-instant"
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                    )
                else:
                    raise  # If primary was already the smallest model, re-raise to go to outer except
            # Extract generated content
            generation = chat_completion.choices[0].message.content
            return generation
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER

    def generate_detail_response(self, user_query: str, product_doc):
        """
        Generate an AI response focusing on a single product detail page and the original user query.
        Returns a summary/analysis of the product for the user.
        """
        DEFAULT_ANSWER = "No AI-generated summary available for this product."
        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
            # Prepare product details for the prompt, truncating description to keep prompt concise
            title = getattr(product_doc, "title", "")
            brand = getattr(product_doc, "brand", "") or "Unknown"
            category = getattr(product_doc, "category", "") or getattr(product_doc, "sub_category", "") or "Unknown"
            price = None
            if hasattr(product_doc, "selling_price") and product_doc.selling_price:
                price = product_doc.selling_price
            elif hasattr(product_doc, "actual_price") and product_doc.actual_price:
                price = product_doc.actual_price
            price = f"{price:.2f}" if price is not None else "Unknown"
            rating = getattr(product_doc, "average_rating", None)
            rating = f"{rating:.1f}/5" if rating else "N/A"
            desc = getattr(product_doc, "description", "")
            if desc:
                desc_text = str(desc)
                # Truncate description to at most 300 characters for the prompt
                if len(desc_text) > 300:
                    desc_text = desc_text[:300].strip() + "..."
            else:
                desc_text = "No description available."
            # Format the detail prompt
            prompt = self.PROMPT_TEMPLATE_DETAIL.format(
                user_query=user_query,
                title=title,
                brand=brand,
                category=category,
                price=price,
                rating=rating,
                description=desc_text
            )
            # Attempt generation (use same model fallback logic as above)
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                )
            except Exception as e:
                print(f"Detail generation with model {model_name} failed: {e}")
                if model_name != "llama-3.1-8b-instant":
                    model_name = "llama-3.1-8b-instant"
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                    )
                else:
                    raise
            generation = chat_completion.choices[0].message.content
            return generation.strip()
        except Exception as e:
            print(f"Error during detail RAG generation: {e}")
            return DEFAULT_ANSWER
