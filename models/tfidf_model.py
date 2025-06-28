import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

class TFIDFModel:
    """
    A class to build and manage a TF-IDF content-based recommendation model.
    It uses product descriptions to find similar products.
    """
    def __init__(self, products_file_path='data/products.json'):
        self.products_file_path = products_file_path
        self.products_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._load_products()
        self._build_model()

    def _load_products(self):
        """
        Loads product data from the specified JSON file.
        """
        if not os.path.exists(self.products_file_path):
            print(f"Error: Products file not found at {self.products_file_path}")
            self.products_df = pd.DataFrame(columns=['product_id', 'title', 'description'])
            return
        try:
            with open(self.products_file_path, 'r', encoding='utf-8') as f:
                products_data = json.load(f)
            self.products_df = pd.DataFrame(products_data)

            if 'description' not in self.products_df.columns:
                self.products_df['description'] = ''
            else:
                self.products_df['description'] = self.products_df['description'].fillna('')

            print(f"Successfully loaded {len(self.products_df)} products for TF-IDF model.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.products_file_path}. Is it valid JSON?")
            self.products_df = pd.DataFrame(columns=['product_id', 'title', 'description'])
        except Exception as e:
            print(f"Unexpected error while loading products: {e}")
            self.products_df = pd.DataFrame(columns=['product_id', 'title', 'description'])

     
    def _build_model(self):
        """
        Builds the TF-IDF model and computes the TF-IDF matrix.
        """
        if self.products_df.empty:
            print("No products loaded, skipping TF-IDF model build.")
            return

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=0.01)
        corpus = (self.products_df['title'] + ' ' + self.products_df['description']).tolist()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        print("TF-IDF model built and matrix computed.")

    def get_recommendations(self, product_id, num_recommendations=5):
        """
        Generates content-based recommendations for a given product using TF-IDF.
        """
        if self.tfidf_matrix is None or self.products_df.empty:
            print("TF-IDF model not built or product data not available.")
            return []

        try:
            idx = self.products_df.index[self.products_df['product_id'] == product_id].tolist()[0]
        except IndexError:
            print(f"Product with ID '{product_id}' not found for TF-IDF recommendation.")
            return []

        cosine_similarities = linear_kernel(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        related_product_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]

        recommended_product_ids = []
        for i in related_product_indices:
            if i != idx:
                recommended_product_ids.append(self.products_df.iloc[i]['product_id'])
            if len(recommended_product_ids) >= num_recommendations:
                break

        return recommended_product_ids


# Example Usage
if __name__ == "__main__":
    # Ensure 'data' directory and 'products.json' exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    products_file = os.path.join(data_dir, 'products.json')
    if not os.path.exists(products_file):
        dummy_products = [
            {"product_id": "P001", "title": "Wireless Bluetooth Headphones", "category": "Audio", "description": "Noise-cancelling over-ear headphones with long battery life."},
            {"product_id": "P002", "title": "USB-C Charging Cable", "category": "Accessories", "description": "Fast charging cable, 1 meter length, compatible with all Android devices."},
            {"product_id": "P003", "title": "Smartwatch Fitness Tracker", "category": "Wearables", "description": "Tracks heart rate, steps, and sleep. Water-resistant with long battery life."},
            {"product_id": "P004", "title": "Gaming Mouse", "category": "Gaming", "description": "Ergonomic gaming mouse with RGB lighting and customizable DPI settings."},
            {"product_id": "P005", "title": "Portable Power Bank 10000mAh", "category": "Accessories", "description": "Compact power bank with dual USB output for quick charging."},
            {"product_id": "P006", "title": "Gaming Keyboard Mechanical", "category": "Gaming", "description": "RGB backlit mechanical keyboard with tactile switches for gaming."},
            {"product_id": "P007", "title": "External SSD 1TB", "category": "Storage", "description": "High-speed portable solid-state drive for data storage and backup."},
            {"product_id": "P008", "title": "Wireless Earbuds", "category": "Audio", "description": "Compact true wireless earbuds with good sound quality and charging case."}
        ]
        with open(products_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_products, f, indent=2)
        print(f"Created dummy {products_file} for testing.")

    # Initialize and test
    tfidf_model = TFIDFModel(products_file_path=products_file)

    print("\nRecommendations:")
    print("P001 →", tfidf_model.get_recommendations("P001", num_recommendations=3))
    print("P004 →", tfidf_model.get_recommendations("P004", num_recommendations=2))
    print("P999 →", tfidf_model.get_recommendations("P999", num_recommendations=5))  # Invalid ID
