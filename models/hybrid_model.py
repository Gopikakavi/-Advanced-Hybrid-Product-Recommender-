import json
import os
import pandas as pd
from collections import defaultdict
import numpy as np

# Import all individual models
# Ensure these files (e.g., tfidf_model.py, bert_model.py) are in the 'models' directory
from .tfidf_model import TFIDFModel
from .bert_model import BERTModel
from .popularity_model import PopularityModel
from .sentiment_analysis_model import SentimentAnalysisModel
# NEW: Import the collaborative filtering model (assuming it exists in models/)
from .item_similarity_model import ItemSimilarityModel # Or whatever your collaborative model is named

class HybridModel:
    """
    A class to combine recommendations from TF-IDF, BERT, Popularity, Sentiment Analysis, and Collaborative Filtering models.
    It provides a unified interface for generating recommendations using a weighted approach.
    """
    def __init__(self, products_file_path='data/products.json', ratings_file_path='data/ratings.json', reviews_file_path='data/reviews.json'):
        """
        Initializes the HybridModel by loading and setting up the individual models.
        Args:
            products_file_path (str): Path to the products JSON file.
            ratings_file_path (str): Path to the ratings JSON file.
            reviews_file_path (str): Path to the reviews JSON file (for sentiment analysis).
        """
        self.products_file_path = products_file_path
        self.ratings_file_path = ratings_file_path
        self.reviews_file_path = reviews_file_path

        # Load all products once for general use and product detail retrieval
        self.products_df = self._load_json_to_df(self.products_file_path)
        if self.products_df is None:
            print("Failed to load products data. Hybrid model might not function correctly.")
            self.products_df = pd.DataFrame(columns=['product_id', 'title', 'description'])

        # Initialize individual models
        print("Initializing TFIDFModel...")
        self.tfidf_model = TFIDFModel(products_file_path=self.products_file_path)
        print("Initializing BERTModel...")
        self.bert_model = BERTModel(products_file_path=self.products_file_path)
        print("Initializing PopularityModel...")
        self.popularity_model = PopularityModel(ratings_file_path=self.ratings_file_path, products_file_path=self.products_file_path)
        print("Initializing SentimentAnalysisModel...")
        self.sentiment_model = SentimentAnalysisModel(reviews_file_path=self.reviews_file_path, products_file_path=self.products_file_path)
        # NEW: Initialize Collaborative Filtering Model
        print("Initializing Collaborative Filtering Model...")
        self.collaborative_model = ItemSimilarityModel(ratings_file_path=self.ratings_file_path, products_file_path=self.products_file_path)


    def _load_json_to_df(self, file_path):
        """Helper to load a JSON file into a pandas DataFrame."""
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Is it valid JSON?")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while loading {file_path}: {e}")
            return None

    def get_hybrid_recommendations(self, user_id=None, product_id=None, num_recommendations=5, weights=None):
        """
        Generates hybrid recommendations based on weighted scores from multiple models.
        Args:
            user_id (str, optional): The ID of the user for collaborative filtering.
            product_id (str, optional): The ID of the product for content-based filtering.
            num_recommendations (int): The number of recommendations to return.
            weights (dict, optional): A dictionary of weights for each model (e.g., {'tfidf': 0.2, 'bert': 0.2, 'collaborative': 0.3, 'popularity': 0.1, 'sentiment': 0.2}).
                                      If None, default weights or adaptive weighting can be applied.
        Returns:
            list: A list of recommended product_ids.
        """
        # Default weights if not provided, adjusted for new sentiment model
        if weights is None:
            # UPDATED: Added 'collaborative' weight to defaults
            weights = {'tfidf': 0.2, 'bert': 0.2, 'collaborative': 0.2, 'popularity': 0.1, 'sentiment': 0.2} # Example weights, can be tuned

        # Ensure weights sum to 1 (or handle normalization)
        total_weight = sum(weights.values())
        if total_weight == 0: # Avoid division by zero
            print("Warning: Total weight is zero. Using equal weights as fallback.")
            num_models = len(weights)
            weights = {model: 1/num_models for model in weights}
            total_weight = 1
        else:
            weights = {k: v / total_weight for k, v in weights.items()} # Normalize weights

        all_product_ids = self.products_df['product_id'].tolist()
        combined_scores = defaultdict(float)

        # Products to exclude from recommendations (e.g., the input product itself, or already rated products)
        excluded_pids = set()
        if product_id:
            excluded_pids.add(product_id)
        
        # Add products already rated by the user to the exclusion list
        if user_id:
            user_rated_products = self.popularity_model.get_user_rated_products(user_id) # Assuming popularity model can provide this, or make a separate function
            excluded_pids.update(user_rated_products)
        

        # 1. TF-IDF Recommendations (Content-based, requires product_id)
        tfidf_recs = []
        if 'tfidf' in weights and weights['tfidf'] > 0 and product_id:
            tfidf_recs = self.tfidf_model.get_recommendations(product_id, len(all_product_ids))
            self._add_scores(combined_scores, tfidf_recs, weights['tfidf'], all_product_ids)

        # 2. BERT Recommendations (Content-based, requires product_id)
        bert_recs = []
        if 'bert' in weights and weights['bert'] > 0 and product_id:
            bert_recs = self.bert_model.get_recommendations(product_id, len(all_product_ids))
            self._add_scores(combined_scores, bert_recs, weights['bert'], all_product_ids)

        # 3. Collaborative Filtering Recommendations (Requires user_id)
        collaborative_recs = []
        if 'collaborative' in weights and weights['collaborative'] > 0 and user_id:
            # Assuming get_recommendations for ItemSimilarityModel takes user_id and num_recommendations
            collaborative_recs = self.collaborative_model.get_recommendations(user_id, num_recommendations=len(all_product_ids))
            self._add_scores(combined_scores, collaborative_recs, weights['collaborative'], all_product_ids)
            
            # Note: exclusion of user-rated products is handled above in the general excluded_pids logic.
            # Ensure your ItemSimilarityModel's get_recommendations *doesn't* return items already rated by the user,
            # or that the user_rated_products update correctly reflects all such items.
            

        # 4. Popularity Recommendations (General, fallback)
        popularity_recs = []
        if 'popularity' in weights and weights['popularity'] > 0:
            popularity_recs = self.popularity_model.get_recommendations(num_recommendations=len(all_product_ids), user_id=user_id) # Pass user_id to popularity model for exclusion
            self._add_scores(combined_scores, popularity_recs, weights['popularity'], all_product_ids)

        # 5. Sentiment Analysis Influence (Adjust existing scores based on product sentiment)
        # This is integrated differently as it's a "quality" score rather than a recommendation generator
        if 'sentiment' in weights and weights['sentiment'] > 0 and self.sentiment_model.product_sentiment_scores:
            print("Applying sentiment analysis influence to recommendations.")
            temp_combined_scores = defaultdict(float)
            for pid, current_combined_score in combined_scores.items():
                sentiment_score = self.sentiment_model.get_sentiment_score(pid)
                
                # Simple sentiment boosting/penalizing:
                # If sentiment is > 0.5, it boosts the base score. If < 0.5, it lowers it.
                # Max boost/penalty controlled by the sentiment weight.
                
                # Scale sentiment_score from [0,1] to influence factor like [-1,1] or [0.5, 1.5]
                # A common approach: adjust score based on (sentiment - 0.5) * factor
                # Here, we'll try a simple multiplicative influence:
                # If score is 0.5 (neutral), factor is 1.
                # If score is 1.0 (positive), factor is > 1.
                # If score is 0.0 (negative), factor is < 1.
                sentiment_factor = 1.0 + (sentiment_score - 0.5) * weights['sentiment'] * 2 # Multiply by 2 to give full range for sentiment weight

                adjusted_score = current_combined_score * sentiment_factor
                temp_combined_scores[pid] = adjusted_score
            combined_scores = temp_combined_scores
            print("Sentiment analysis influence applied.")
        elif 'sentiment' in weights and weights['sentiment'] > 0:
            print("Sentiment model not ready or no sentiment scores available; cannot apply sentiment influence.")


        # Combine and rank
        # Sort products by their combined scores in descending order
        # Ensure we only consider products that actually received some score
        final_recommendations_with_scores = sorted(
            [item for item in combined_scores.items() if item[1] > 0], # Only consider products with score > 0
            key=lambda item: item[1],
            reverse=True
        )

        # Filter out excluded products (self-product, already rated) and take the top N
        filtered_recommendation_ids = []
        for product_id, score in final_recommendations_with_scores:
            if product_id not in excluded_pids:
                filtered_recommendation_ids.append(product_id)
            if len(filtered_recommendation_ids) >= num_recommendations:
                break # Stop once we have enough recommendations

        # If after all filtering, we still don't have enough, return what we have.
        # The slice [ : num_recommendations] is safe even if the list is shorter.
        return filtered_recommendation_ids[:num_recommendations]


    def _add_scores(self, combined_scores, recommendations_list, weight, all_product_ids):
        """
        Adds scores from a single model's recommendations to the combined scores.
        Assumes recommendations_list contains product IDs, ordered from most to least relevant.
        Scores are inversely proportional to rank.
        """
        num_products = len(all_product_ids)
        if num_products == 0:
            return # Avoid division by zero if no products

        score_map = {}
        for rank, product_id in enumerate(recommendations_list):
            # Assign a score based on rank (higher rank means higher initial score)
            # Normalize it by num_products to keep it between 0 and 1
            score_map[product_id] = (num_products - rank) / num_products

        for product_id in all_product_ids:
            # Get the score from this model; 0 if the product wasn't recommended by this model
            score_from_this_model = score_map.get(product_id, 0)
            combined_scores[product_id] += score_from_this_model * weight

    def get_product_details(self, product_ids):
        """
        Retrieves details for a list of product IDs.
        Args:
            product_ids (list): A list of product IDs.
        Returns:
            list: A list of dictionaries, each containing details for a product.
        """
        if self.products_df is None or self.products_df.empty:
            return []
        
        details = []
        for pid in product_ids:
            # Use .get() with a default empty dict to avoid KeyError if product_id is not found
            # Convert to dictionary if you're pulling from a DataFrame row directly
            product_info = self.products_df[self.products_df['product_id'] == pid]
            if not product_info.empty:
                details.append(product_info.iloc[0].to_dict())
            else:
                # Provide a fallback for product details if not found (e.g., if IDs mismatch)
                details.append({
                    "product_id": pid,
                    "title": "Unknown Product",
                    "description": "Details not available.",
                    "category": "Unknown",
                    "image_url": "https://placehold.co/150x150/AAAAAA/FFFFFF?text=Product+Not+Found"
                })
        return details

# Example Usage (for testing purposes, typically used by main.py)
if __name__ == "__main__":
    # Ensure 'data' directory and necessary JSON files exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Dummy products.json
    products_file = os.path.join(data_dir, 'products.json')
    if not os.path.exists(products_file):
        dummy_products = [
            {"product_id": "P001", "title": "Wireless Bluetooth Headphones", "category": "Audio", "description": "Noise-cancelling over-ear headphones."},
            {"product_id": "P002", "title": "USB-C Charging Cable", "category": "Accessories", "description": "Fast charging cable."},
            {"product_id": "P003", "title": "Smartphone Stand", "category": "Accessories", "description": "Adjustable aluminum stand."},
            {"product_id": "P004", "title": "Portable Power Bank", "category": "Accessories", "description": "Compact 10000mAh power bank."},
            {"product_id": "P005", "title": "Gaming Mouse", "category": "Gaming", "description": "Ergonomic gaming mouse RGB."},
            {"product_id": "P006", "title": "Smartwatch Fitness Tracker", "category": "Wearables", "description": "Tracks heart rate and steps."},
            {"product_id": "P007", "title": "External SSD 1TB", "category": "Storage", "description": "High-speed portable SSD."},
            {"product_id": "P008", "title": "Wireless Charging Pad", "category": "Accessories", "description": "Qi-compatible charging pad."},
            {"product_id": "P009", "title": "Laptop Backpack", "category": "Bags", "description": "Durable water-resistant backpack."},
            {"product_id": "P010", "title": "Action Camera 4K", "category": "Cameras", "description": "Ultra HD action camera."}
        ]
        with open(products_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_products, f, indent=2)
        print(f"Created dummy {products_file}.")

    # Dummy ratings.json
    ratings_file = os.path.join(data_dir, 'ratings.json')
    if not os.path.exists(ratings_file):
        dummy_ratings = [
            {"user_id": "U001", "product_id": "P001", "rating": 5},
            {"user_id": "U001", "product_id": "P002", "rating": 4},
            {"user_id": "U002", "product_id": "P001", "rating": 4},
            {"user_id": "U002", "product_id": "P003", "rating": 5},
            {"user_id": "U003", "product_id": "P002", "rating": 5},
            {"user_id": "U003", "product_id": "P004", "rating": 4},
            {"user_id": "U001", "product_id": "P008", "rating": 2}, # Add for testing negative sentiment
            {"user_id": "U002", "product_id": "P005", "rating": 3}, # Add for testing mixed sentiment
        ]
        with open(ratings_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_ratings, f, indent=2)
        print(f"Created dummy {ratings_file}.")

    # Dummy reviews.json for sentiment analysis
    reviews_file = os.path.join(data_dir, 'reviews.json')
    if not os.path.exists(reviews_file):
        dummy_reviews = [
            {"product_id": "P001", "review_text": "These headphones are amazing, great sound quality!", "rating": 5},
            {"product_id": "P001", "review_text": "Very comfortable, but battery life could be better.", "rating": 4},
            {"product_id": "P002", "review_text": "Good cable, charges fast.", "rating": 4},
            {"product_id": "P003", "review_text": "Excellent stand, very sturdy.", "rating": 5},
            {"product_id": "P004", "review_text": "Works as expected, compact power bank.", "rating": 4},
            {"product_id": "P005", "review_text": "Mouse is okay, not as ergonomic as I hoped.", "rating": 3},
            {"product_id": "P006", "review_text": "Fitness tracker is accurate and stylish.", "rating": 5},
            {"product_id": "P007", "review_text": "SSD is super fast, highly recommend.", "rating": 5},
            {"product_id": "P008", "review_text": "Wireless charger is slow.", "rating": 2}, # Negative review
            {"product_id": "P009", "review_text": "Backpack is durable, love it.", "rating": 5},
            {"product_id": "P010", "review_text": "Camera quality is great for the price.", "rating": 4},
            {"product_id": "P005", "review_text": "RGB lighting is cool, but buttons feel cheap.", "rating": 2}, # Another review for P005
            {"product_id": "P004", "review_text": "Charges my phone twice, very handy!", "rating": 5}, # Positive review for P004
        ]
        with open(reviews_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_reviews, f, indent=2)
        print(f"Created dummy {reviews_file}.")
    
    # Initialize the HybridModel
    print("\n--- Initializing HybridModel (this will initialize all sub-models) ---")
    hybrid_model = HybridModel(
        products_file_path=products_file,
        ratings_file_path=ratings_file,
        reviews_file_path=reviews_file # Pass reviews file path
    )

    print("\n--- Testing Hybrid Recommendations ---")

    # Scenario 1: User-based and Content-based (e.g., for User U001, interested in P001-like items)
    print(f"Scenario 1: Recs for U001 interested in P001-like items (TF-IDF, BERT, Collaborative, Popularity, Sentiment):")
    # Note: User U001 has rated P001, P002, P008. These should be excluded if the models handle it.
    recommendations_1 = hybrid_model.get_hybrid_recommendations(user_id="U001", product_id="P001", num_recommendations=5)
    rec_details_1 = hybrid_model.get_product_details(recommendations_1)
    for rec in rec_details_1:
        print(f"- {rec['title']} (ID: {rec['product_id']})")

    # Scenario 2: Only content-based (e.g., for a new user, or if only product context is available)
    print(f"\nScenario 2: Recs for P003-like items (TF-IDF, BERT, Sentiment - modified weights):")
    recommendations_2 = hybrid_model.get_hybrid_recommendations(product_id="P003", num_recommendations=3,
                                                                 weights={'tfidf':0.4, 'bert':0.4, 'popularity':0.1, 'sentiment':0.1, 'collaborative':0.0})
    rec_details_2 = hybrid_model.get_product_details(recommendations_2)
    for rec in rec_details_2:
        print(f"- {rec['title']} (ID: {rec['product_id']})")

    # Scenario 3: Only Popularity (as a fallback or general trending) - explicit static weights
    print(f"\nScenario 3: Recs for general popularity (no user/product context, explicit static weights):")
    recommendations_3 = hybrid_model.get_hybrid_recommendations(num_recommendations=5,
                                                                 weights={'tfidf':0.0, 'bert':0.0, 'popularity':0.8, 'sentiment':0.2, 'collaborative':0.0})
    rec_details_3 = hybrid_model.get_product_details(recommendations_3)
    for rec in rec_details_3:
        print(f"- {rec['title']} (ID: {rec['product_id']})")

    # Scenario 4: Focus on Collaborative Filtering for a user
    print(f"\nScenario 4: Recs for U002 focusing on Collaborative Filtering:")
    # U002 has rated P001 and P003. Collaborative model should find similar users/items based on these.
    recommendations_4 = hybrid_model.get_hybrid_recommendations(user_id="U002", num_recommendations=5,
                                                                 weights={'tfidf':0.0, 'bert':0.0, 'popularity':0.1, 'sentiment':0.1, 'collaborative':0.8})
    rec_details_4 = hybrid_model.get_product_details(recommendations_4)
    for rec in rec_details_4:
        print(f"- {rec['title']} (ID: {rec['product_id']})")