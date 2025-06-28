import streamlit as st
import requests
import json
import os

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Hybrid Product Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FastAPI Backend URL ---
FASTAPI_URL = "http://localhost:8000/recommend"
PRODUCTS_FILE_PATH = 'data/products.json'

@st.cache_data
def load_products(file_path):
    if not os.path.exists(file_path):
        st.error(f"Products file not found at {file_path}.")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in {file_path}")
        return []
    except Exception as e:
        st.error(f"An error occurred while loading products: {e}")
        return []

# --- Load Products ---
all_products = load_products(PRODUCTS_FILE_PATH)

# --- UI: Sidebar Inputs ---
st.title("🧠 Hybrid Product Recommendation System")

st.markdown("""
This system combines:
- TF-IDF
- BERT
- Collaborative Filtering (Item-Similarity) # UPDATED: Added Collaborative
- Popularity
- Sentiment
To generate intelligent recommendations.
""")

st.sidebar.header("User Input")

# User ID
user_id_input = st.sidebar.text_input("User ID (e.g., U001)")

# 🔍 Category Filtering
categories = sorted(list(set(p.get('category', 'Uncategorized') for p in all_products)))
selected_category = st.sidebar.selectbox("Filter by Category", ["All"] + categories)

if selected_category != "All":
    filtered_products = [p for p in all_products if p.get('category') == selected_category]
else:
    filtered_products = all_products

# Product selector after filtering
# Ensure product_id is correctly mapped
product_map = {p["title"]: p["product_id"] for p in filtered_products if "title" in p and "product_id" in p}
product_titles = sorted(product_map.keys())
selected_title = st.sidebar.selectbox("Select Product (optional)", ["-- None --"] + product_titles)
product_id_input = product_map.get(selected_title) if selected_title != "-- None --" else None

# Number of recommendations
num_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

# Advanced Weights
st.sidebar.subheader("Advanced: Model Weights")
use_custom_weights = st.sidebar.checkbox("Use Custom Weights", value=False)
weights = {}

if use_custom_weights:
    st.sidebar.info("Weights should total ~1.0. Normalization will be applied if needed.")
    weights['tfidf'] = st.sidebar.number_input("TF-IDF", 0.0, 1.0, 0.2, step=0.05)
    weights['bert'] = st.sidebar.number_input("BERT", 0.0, 1.0, 0.2, step=0.05)
    weights['collaborative'] = st.sidebar.number_input("Collaborative", 0.0, 1.0, 0.2, step=0.05) # UPDATED: Added Collaborative
    # Removed 'autoencoder'
    weights['popularity'] = st.sidebar.number_input("Popularity", 0.0, 1.0, 0.2, step=0.05)
    weights['sentiment'] = st.sidebar.number_input("Sentiment", 0.0, 1.0, 0.2, step=0.05)
    total = sum(weights.values())
    st.sidebar.write(f"Total: {total:.2f}")
    if not 0.99 <= total <= 1.01:
        st.sidebar.warning("Weights will be normalized.")
else:
    # Default weights should match the hybrid model's defaults if no custom weights are used
    weights = None # Let backend handle defaults

# --- Get Recommendations ---
if st.sidebar.button("Get Recommendations"):
    if not user_id_input and not product_id_input:
        st.warning("Please enter a User ID or select a product.")
    else:
        payload = {
            "user_id": user_id_input if user_id_input else None, # Ensure empty string becomes None
            "product_id": product_id_input if product_id_input else None, # Ensure empty string becomes None
            "num_recommendations": num_recommendations,
            "weights": weights
        }

        st.subheader("🔍 Recommendations")
        with st.spinner("Fetching results..."):
            try:
                res = requests.post(FASTAPI_URL, json=payload)
                res.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                data = res.json()

                if data.get("recommendations"):
                    st.success(f"Top {len(data['recommendations'])} Recommendations:")
                    # Display recommendations as before
                    # The get_product_details in hybrid_model.py returns full product dicts
                    for i, p in enumerate(data['recommendations']):
                        st.markdown("---")
                        col1, col2, col3 = st.columns([1, 2, 3])
                        with col1:
                            if p.get("image_url"):
                                st.image(p["image_url"], width=150, caption=f"ID: {p['product_id']}")
                            else:
                                st.image("https://placehold.co/150x150?text=No+Image", width=150)

                        with col2:
                            st.markdown(f"### {i+1}. {p.get('title', 'N/A')}")
                            st.markdown(f"**Category:** {p.get('category', 'N/A')}")
                            # Add product_id display for clarity
                            st.markdown(f"**Product ID:** {p.get('product_id', 'N/A')}")

                        with col3:
                            st.markdown("**Description:**")
                            st.write(p.get("description", "N/A"))
                else:
                    st.info("No recommendations found. Try different input.")

            except requests.exceptions.ConnectionError:
                st.error("❌ Backend not responding. Is FastAPI running?")
                st.code("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error occurred: {http_err} - Response: {res.text}")
            except json.JSONDecodeError:
                st.error("❌ Could not decode JSON response from backend. Check backend logs.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("👨‍💻 Project by: GOPIKA.A")
st.markdown("[GitHub Repo](https://github.com/your-link)")