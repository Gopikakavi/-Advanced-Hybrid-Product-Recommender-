import json
import os
import pandas as pd
import numpy as np

class PopularityModel:
    """
    A class to build and manage a popularity-based recommendation model.
    It recommends products based on their overall average rating or number of interactions.
    """
    def __init__(self, products_file_path='data/products.json', ratings_file_path='data/ratings.json'):
        """
        Initializes the PopularityModel.
        Args:
            products_file_path (str): The path to the products JSON file.
            ratings_file_path (str): The path to the ratings JSON file.
        """
        self.products_file_path = products_file_path
        self.ratings_file_path = ratings_file_path
        self.products_df = None
        self.ratings_df = None
        self.popular_products = [] # Stores sorted list of popular product IDs
        self._load_data()
        self._calculate_popularity()

    def _load_data(self):
        """
        Loads product and ratings data from the specified JSON files.
        """
        # Load products data
        if not os.path.exists(self.products_file_path):
            print(f"Error: Products file not found at {self.products_file_path}")
            self.products_df = pd.DataFrame(columns=['product_id', 'title'])
        else:
            try:
                with open(self.products_file_path, 'r', encoding='utf-8') as f:
                    products_data = json.load(f)
                self.products_df = pd.DataFrame(products_data)
                print(f"Successfully loaded {len(self.products_df)} products for Popularity model.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {self.products_file_path}. Is it valid JSON?")
                self.products_df = pd.DataFrame(columns=['product_id', 'title'])
            except Exception as e:
                print(f"An unexpected error occurred while loading products for Popularity: {e}")
                self.products_df = pd.DataFrame(columns=['product_id', 'title'])

        # Load ratings data
        if not os.path.exists(self.ratings_file_path):
            print(f"Error: Ratings file not found at {self.ratings_file_path}")
            self.ratings_df = pd.DataFrame(columns=['user_id', 'product_id', 'rating'])
        else:
            try:
                with open(self.ratings_file_path, 'r', encoding='utf-8') as f:
                    ratings_data = json.load(f)
                self.ratings_df = pd.DataFrame(ratings_data)
                print(f"Successfully loaded {len(self.ratings_df)} ratings for Popularity model.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {self.ratings_file_path}. Is it valid JSON?")
                self.ratings_df = pd.DataFrame(columns=['user_id', 'product_id', 'rating'])
            except Exception as e:
                print(f"An unexpected error occurred while loading ratings for Popularity: {e}")
                self.ratings_df = pd.DataFrame(columns=['user_id', 'product_id', 'rating'])

    def _calculate_popularity(self):
        """
        Calculates product popularity based on average rating and number of ratings.
        Populates self.popular_products with sorted product IDs.
        """
        if self.ratings_df is None or self.ratings_df.empty:
            print("No ratings data available to calculate popularity. Returning all products as default.")
            if self.products_df is not None and not self.products_df.empty:
                self.popular_products = self.products_df['product_id'].tolist()
            else:
                self.popular_products = []
            return

        # Calculate average rating and count for each product
        product_popularity = self.ratings_df.groupby('product_id')['rating'].agg(
            avg_rating='mean',
            rating_count='count'
        ).reset_index()

        # Merge with products_df to ensure all products are considered, even if no ratings
        if self.products_df is not None and not self.products_df.empty:
            product_popularity = pd.merge(
                self.products_df[['product_id', 'title']], # Include title for merging clarity
                product_popularity,
                on='product_id',
                how='left'
            )
        else:
            print("Warning: Products DataFrame is empty or not loaded. Popularity calculation might be limited.")

        # Fill NaN values for products with no ratings
        product_popularity['avg_rating'] = product_popularity['avg_rating'].fillna(0) # Or a neutral rating like 3.0
        product_popularity['rating_count'] = product_popularity['rating_count'].fillna(0).astype(int)

        # Sort by rating count (primary) and then by average rating (secondary)
        # Products with more ratings and higher average ratings will be considered more popular
        self.popular_products = product_popularity.sort_values(
            by=['rating_count', 'avg_rating'],
            ascending=[False, False]
        )['product_id'].tolist()

        print(f"Calculated popularity for {len(self.popular_products)} products.")

    def get_recommendations(self, num_recommendations=5, user_id=None):
        """
        Generates popularity-based recommendations.
        Optionally excludes products already seen by a specific user.
        Args:
            num_recommendations (int): The number of top popular products to return.
            user_id (str, optional): If provided, products rated by this user will be excluded.
        Returns:
            list: A list of recommended product_ids.
        """
        if not self.popular_products:
            print("No popular products calculated. Check data loading and popularity calculation.")
            return []

        recommended_products = []
        if user_id and self.ratings_df is not None and not self.ratings_df.empty:
            # Get products the user has already rated
            user_rated_products = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['product_id'].tolist())
            # Filter out products already rated by the user
            for product_id in self.popular_products:
                if product_id not in user_rated_products:
                    recommended_products.append(product_id)
                if len(recommended_products) >= num_recommendations:
                    break
        else:
            # If no user_id or no ratings data, just return top N popular products
            recommended_products = self.popular_products[:num_recommendations]

        return recommended_products
    # Inside models/popularity_model.py, within the PopularityModel class:
    def get_user_rated_products(self, user_id): # <--- CORRECTLY INDENTED NOW
        """
        Helper method to get product IDs a user has already rated.
        """
        if self.ratings_df is None or self.ratings_df.empty:
            return set()

        # Filter ratings for the given user
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]

        # Get unique product_ids from those ratings
        rated_products = user_ratings['product_id'].unique().tolist()
        return set(rated_products)


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Ensure 'data' directory and necessary JSON files exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Dummy products.json
    products_file = os.path.join(data_dir, 'products.json')
    if not os.path.exists(products_file):
        dummy_products = [
            {"product_id": "P001", "title": "Wireless Bluetooth Headphones", "category": "Audio", "description": "Good sound."},
            {"product_id": "P002", "title": "USB-C Charging Cable", "category": "Accessories", "description": "Fast charging."},
            {"product_id": "P003", "title": "Smartwatch Fitness Tracker", "category": "Wearables", "description": "Tracks heart rate."},
            {"product_id": "P004", "title": "Gaming Mouse", "category": "Gaming", "description": "RGB lighting."},
            {"product_id": "P005", "title": "Portable Power Bank", "category": "Accessories", "description": "High capacity."},
            {"product_id": "P006", "title": "Wireless Earbuds", "category": "Audio", "description": "Compact design."},
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
            {"user_id": "U001", "product_id": "P003", "rating": 5},
            {"user_id": "U002", "product_id": "P001", "rating": 4},
            {"user_id": "U002", "product_id": "P003", "rating": 5},
            {"user_id": "U002", "product_id": "P004", "rating": 3},
            {"user_id": "U003", "product_id": "P001", "rating": 5},
            {"user_id": "U003", "product_id": "P002", "rating": 5},
            {"user_id": "U003", "product_id": "P005", "rating": 4},
            {"user_id": "U004", "product_id": "P001", "rating": 4},
            {"user_id": "U004", "product_id": "P006", "rating": 3},
        ]
        with open(ratings_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_ratings, f, indent=2)
        print(f"Created dummy {ratings_file}.")

    print("\n--- Initializing PopularityModel ---")
    popularity_model = PopularityModel(products_file_path=products_file, ratings_file_path=ratings_file)

    print("\n--- Testing Popularity-Based Recommendations ---")

    # Get top 3 popular products (general)
    general_recommendations = popularity_model.get_recommendations(num_recommendations=3)
    print(f"Top 3 general popular products: {general_recommendations}")

    # Get top 3 popular products for U001, excluding what they've rated
    user_id_test = "U001"
    user_seen_products = popularity_model.ratings_df[popularity_model.ratings_df['user_id'] == user_id_test]['product_id'].tolist()
    print(f"\nProducts rated by {user_id_test}: {user_seen_products}")
    recommendations_for_user = popularity_model.get_recommendations(num_recommendations=3, user_id=user_id_test)
    print(f"Top 3 popular products for {user_id_test} (excluding rated): {recommendations_for_user}")

    # Test with a user who has rated all top products (or very few new ones)
    print("\n--- Testing Popularity-Based with User who rated many top products ---")
    user_id_extensive_rater = "U003"
    extensive_rater_seen = popularity_model.ratings_df[popularity_model.ratings_df['user_id'] == user_id_extensive_rater]['product_id'].tolist()
    print(f"Products rated by {user_id_extensive_rater}: {extensive_rater_seen}")
    recommendations_extensive = popularity_model.get_recommendations(num_recommendations=5, user_id=user_id_extensive_rater)
    print(f"Recommendations for {user_id_extensive_rater} (excluding rated): {recommendations_extensive}")

    # Test with no ratings data
    print("\n--- Testing Popularity-Based with No Ratings ---\n")
    # Temporarily remove ratings file to simulate no data
    if os.path.exists(ratings_file):
        os.remove(ratings_file)
        print("Temporarily removed ratings.json to test 'no ratings' scenario.")

    no_ratings_model = PopularityModel(products_file_path=products_file, ratings_file_path=ratings_file)
    no_ratings_recs = no_ratings_model.get_recommendations(num_recommendations=5)
    print(f"Recommendations when no ratings file: {no_ratings_recs}")

    # Re-create ratings.json for subsequent tests if it was removed
    if not os.path.exists(ratings_file):
        dummy_ratings = [
            {"user_id": "U001", "product_id": "P001", "rating": 5},
            {"user_id": "U001", "product_id": "P002", "rating": 4},
            {"user_id": "U001", "product_id": "P003", "rating": 5},
            {"user_id": "U002", "product_id": "P001", "rating": 4},
            {"user_id": "U002", "product_id": "P003", "rating": 5},
            {"user_id": "U002", "product_id": "P004", "rating": 3},
            {"user_id": "U003", "product_id": "P001", "rating": 5},
            {"user_id": "U003", "product_id": "P002", "rating": 5},
            {"user_id": "U003", "product_id": "P005", "rating": 4},
            {"user_id": "U004", "product_id": "P001", "rating": 4},
            {"user_id": "U004", "product_id": "P006", "rating": 3},
        ]
        with open(ratings_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_ratings, f, indent=2)
        print("\nRe-created dummy ratings.json.")