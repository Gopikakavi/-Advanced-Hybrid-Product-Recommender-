# models/item_similarity_model.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os
import json
import numpy as np
from collections import defaultdict

class ItemSimilarityModel:
    """
    A collaborative filtering model based on item-item similarity.
    It recommends items similar to those a user has already liked/rated.
    """
    def __init__(self, ratings_file_path='data/ratings.json', products_file_path='data/products.json'):
        self.ratings_file_path = ratings_file_path
        self.products_file_path = products_file_path
        self.ratings_df = None
        self.products_df = None
        self.item_similarity_matrix = None
        self.user_product_matrix = None # User-Item matrix for lookups
        self.product_to_idx = {}
        self.idx_to_product = {}
        self._load_data()
        self._build_model()

    def _load_data(self):
        """
        Loads ratings and product data from JSON files.
        """
        try:
            if os.path.exists(self.ratings_file_path):
                with open(self.ratings_file_path, 'r', encoding='utf-8') as f:
                    self.ratings_df = pd.DataFrame(json.load(f))
                print(f"Successfully loaded {len(self.ratings_df)} ratings for ItemSimilarityModel.")
            else:
                print(f"Error: Ratings file not found at {self.ratings_file_path}")
                self.ratings_df = pd.DataFrame(columns=['user_id', 'product_id', 'rating'])

            if os.path.exists(self.products_file_path):
                with open(self.products_file_path, 'r', encoding='utf-8') as f:
                    self.products_df = pd.DataFrame(json.load(f))
                print(f"Successfully loaded {len(self.products_df)} products for ItemSimilarityModel.")
            else:
                print(f"Error: Products file not found at {self.products_file_path}")
                self.products_df = pd.DataFrame(columns=['product_id', 'title'])

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.ratings_file_path} or {self.products_file_path}. Are they valid JSON?")
            self.ratings_df = pd.DataFrame(columns=['user_id', 'product_id', 'rating'])
            self.products_df = pd.DataFrame(columns=['product_id', 'title'])
        except Exception as e:
            print(f"Unexpected error while loading data for ItemSimilarityModel: {e}")
            self.ratings_df = pd.DataFrame(columns=['user_id', 'product_id', 'rating'])
            self.products_df = pd.DataFrame(columns=['product_id', 'title'])

        # Ensure product_id column exists in products_df for mapping
        if 'product_id' not in self.products_df.columns:
            self.products_df['product_id'] = self.products_df.index.astype(str) # Fallback if no product_id column

    def _build_model(self):
        """
        Builds the user-item matrix and computes item-item similarity matrix.
        """
        if self.ratings_df.empty or self.products_df.empty:
            print("No ratings or product data, skipping ItemSimilarityModel build.")
            self.item_similarity_matrix = None
            self.user_product_matrix = None
            return

        # Create a combined list of all unique product IDs from both ratings and products_df
        all_product_ids = pd.concat([self.ratings_df['product_id'], self.products_df['product_id']]).unique()
        self.product_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
        self.idx_to_product = {i: pid for i, pid in enumerate(all_product_ids)}

        # Create the user-item matrix
        # Pivot the ratings_df to get users as rows, products as columns, and ratings as values
        self.user_product_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating'
        ).fillna(0) # Fill NaN with 0 for items not rated by a user

        # Filter out products that are in ratings but not in products_df (if any, though ideally they should match)
        # And ensure all products from products_df are included as columns, even if no ratings yet
        missing_product_cols = [pid for pid in all_product_ids if pid not in self.user_product_matrix.columns]
        for col in missing_product_cols:
            self.user_product_matrix[col] = 0

        # Ensure columns are in a consistent order if needed (e.g., using product_to_idx order)
        self.user_product_matrix = self.user_product_matrix[all_product_ids]

        # Convert to a sparse matrix for efficiency, though for small datasets DataFrame is fine
        item_user_matrix = self.user_product_matrix.T # Transpose to get items as rows, users as columns

        # Compute item-item similarity using cosine similarity
        # We use a try-except block in case all items have zero variance (e.g., all 0 ratings)
        try:
            self.item_similarity_matrix = cosine_similarity(item_user_matrix)
            print("Item-item similarity matrix computed.")
        except ValueError as e:
            print(f"Error computing cosine similarity (likely sparse matrix issue or no variance): {e}")
            self.item_similarity_matrix = None # Indicate failure to build
            return

        # Store product IDs in the same order as the matrix
        self.item_idx_to_product_id = item_user_matrix.index.tolist()
        self.product_id_to_item_idx = {pid: i for i, pid in enumerate(self.item_idx_to_product_id)}


    def get_recommendations(self, user_id, num_recommendations=5):
        """
        Generates collaborative filtering recommendations for a given user.
        Args:
            user_id (str): The ID of the user.
            num_recommendations (int): The number of recommendations to return.
        Returns:
            list: A list of recommended product_ids.
        """
        if self.item_similarity_matrix is None or self.user_product_matrix is None:
            print("Item similarity model not built or data not available.")
            return []

        if user_id not in self.user_product_matrix.index:
            print(f"User '{user_id}' not found in ratings data. Cannot generate collaborative recommendations.")
            # Fallback for new users: return popular items or empty list
            return []

        user_ratings = self.user_product_matrix.loc[user_id]
        
        # Products the user has already rated
        rated_product_ids = user_ratings[user_ratings > 0].index.tolist()
        
        # Calculate predicted ratings for unrated items
        predicted_ratings = defaultdict(float)

        # Iterate through each unrated product
        unrated_product_ids = [
            pid for pid in self.item_idx_to_product_id
            if pid not in rated_product_ids
        ]

        if not unrated_product_ids:
            print(f"User '{user_id}' has rated all available products or no unrated products found.")
            return []

        for unrated_pid in unrated_product_ids:
            unrated_item_idx = self.product_id_to_item_idx.get(unrated_pid)
            if unrated_item_idx is None: # Should not happen if unrated_product_ids are from item_idx_to_product_id
                continue

            similarity_scores_for_unrated = self.item_similarity_matrix[unrated_item_idx]
            
            # Weighted sum of ratings from similar items that the user *has* rated
            numerator = 0.0
            denominator = 0.0
            
            for rated_pid in rated_product_ids:
                rated_item_idx = self.product_id_to_item_idx.get(rated_pid)
                if rated_item_idx is None:
                    continue
                
                # Similarity between the unrated item and the currently rated item
                similarity = similarity_scores_for_unrated[rated_item_idx]
                
                # User's actual rating for the rated item
                rating = user_ratings[rated_pid]
                
                numerator += similarity * rating
                denominator += abs(similarity) # Use absolute value to avoid negative similarities canceling out

            if denominator > 0:
                predicted_ratings[unrated_pid] = numerator / denominator
            # If denominator is 0, means no similar rated items, predicted_rating remains 0 (defaultdict)

        # Sort recommendations by predicted rating
        recommended_product_ids = sorted(predicted_ratings.items(), key=lambda item: item[1], reverse=True)
        
        # Filter out products with 0 or very low predicted ratings (optional, but good for quality)
        # and take the top N
        final_recs = [pid for pid, score in recommended_product_ids if score > 0][:num_recommendations]

        print(f"Generated {len(final_recs)} collaborative recommendations for user '{user_id}'.")
        return final_recs

    def get_user_rated_products(self, user_id):
        """
        Helper method to get product IDs a user has already rated.
        Used by HybridModel to exclude already rated items from recommendations.
        """
        if self.user_product_matrix is None or user_id not in self.user_product_matrix.index:
            return set()
        
        user_ratings = self.user_product_matrix.loc[user_id]
        rated_products = user_ratings[user_ratings > 0].index.tolist()
        return set(rated_products)


# Example Usage
if __name__ == "__main__":
    # Ensure 'data' directory and 'ratings.json', 'products.json' exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Dummy products.json (ensure it has products that also appear in ratings)
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
            {"user_id": "U001", "product_id": "P005", "rating": 3}, # U001 also rated P005
            {"user_id": "U002", "product_id": "P004", "rating": 3}, # U002 also rated P004
            {"user_id": "U004", "product_id": "P001", "rating": 5}, # New user with some ratings
            {"user_id": "U004", "product_id": "P002", "rating": 5},
            {"user_id": "U004", "product_id": "P003", "rating": 4},
        ]
        with open(ratings_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_ratings, f, indent=2)
        print(f"Created dummy {ratings_file}.")

    print("\n--- Initializing ItemSimilarityModel ---")
    item_model = ItemSimilarityModel(ratings_file_path=ratings_file, products_file_path=products_file)

    print("\n--- Testing Item-Similarity Recommendations ---")
    
    # User U001 has rated P001, P002, P005. Let's see what else they might like.
    print(f"Recommendations for U001 (rated P001, P002, P005): {item_model.get_recommendations('U001', num_recommendations=3)}")

    # User U002 has rated P001, P003, P004.
    print(f"Recommendations for U002 (rated P001, P003, P004): {item_model.get_recommendations('U002', num_recommendations=3)}")

    # Test for a user not in the dataset
    print(f"Recommendations for U999 (non-existent user): {item_model.get_recommendations('U999', num_recommendations=3)}")

    # Test get_user_rated_products
    print(f"Products rated by U001: {item_model.get_user_rated_products('U001')}")
    print(f"Products rated by U005 (non-existent user): {item_model.get_user_rated_products('U005')}")