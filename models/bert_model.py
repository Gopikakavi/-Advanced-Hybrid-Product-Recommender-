import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

class BERTModel:
    """
    A class to build and manage a BERT-based semantic similarity model.
    It uses product descriptions to find semantically similar products.
    """
    def __init__(self, products_file_path='data/products.json', model_name='all-MiniLM-L6-v2'):
        """
        Initializes the BERTModel.
        Args:
            products_file_path (str): The path to the products JSON file.
            model_name (str): The name of the Sentence-BERT model to use.
                               'all-MiniLM-L6-v2' is a good balance of speed and performance.
        """
        self.products_file_path = products_file_path
        self.model_name = model_name
        self.products_df = None
        self.model = None
        self.product_embeddings = None
        self._load_products()
        self._load_model_and_encode()

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
            # Ensure description column exists and handle potential NaNs
            if 'description' not in self.products_df.columns:
                self.products_df['description'] = ''
            self.products_df['description'] = self.products_df['description'].fillna('')
            print(f"Successfully loaded {len(self.products_df)} products for BERT model.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.products_file_path}. Is it valid JSON?")
            self.products_df = pd.DataFrame(columns=['product_id', 'title', 'description'])
        except Exception as e:
            print(f"An unexpected error occurred while loading products for BERT: {e}")
            self.products_df = pd.DataFrame(columns=['product_id', 'title', 'description'])

    def _load_model_and_encode(self):
        """
        Loads the pre-trained BERT model and generates embeddings for all product descriptions.
        """
        if self.products_df.empty:
            print("No products loaded, skipping BERT model embedding.")
            return
        try:
            # Load pre-trained Sentence-BERT model
            self.model = SentenceTransformer(self.model_name)
            print(f"BERT model '{self.model_name}' loaded.")
            # Combine title and description for richer semantic representation
            corpus = (self.products_df['title'] + ' ' + self.products_df['description']).tolist()
            # Generate embeddings for all product descriptions
            # Using device='cuda' if GPU is available, else 'cpu'
            self.product_embeddings = self.model.encode(corpus, convert_to_tensor=True,
                                                         show_progress_bar=True,
                                                         device='cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Generated embeddings for {len(self.product_embeddings)} products.")
        except Exception as e:
            print(f"Error loading BERT model or encoding products: {e}")
            self.model = None
            self.product_embeddings = None

    def get_recommendations(self, product_id, num_recommendations=5):
        """
        Generates semantic similarity-based recommendations for a given product.
        Args:
            product_id (str): The ID of the product for which to get recommendations.
            num_recommendations (int): The number of top recommendations to return.
        Returns:
            list: A list of recommended product_ids.
        """
        if self.model is None or self.product_embeddings is None or self.products_df.empty:
            print("BERT model, embeddings, or product data not available.")
            return []

        # Find the index of the given product_id
        try:
            # This 'idx' corresponds to the DataFrame index, which is also the index in self.product_embeddings
            idx = self.products_df.index[self.products_df['product_id'] == product_id].tolist()[0]
        except IndexError:
            print(f"Product with ID '{product_id}' not found for BERT recommendation.")
            return []

        # Get the embedding of the query product
        query_embedding = self.product_embeddings[idx]

        # Calculate cosine similarities between the query product and all other products
        cosine_scores = util.cos_sim(query_embedding, self.product_embeddings)[0]

        # --- START OF THE FIX ---
        # Determine the maximum k value we can ask for
        # It should be at most the number of available products' embeddings.
        max_k_possible = len(self.product_embeddings) 

        # We ask for num_recommendations + 1 to include the query item initially, then filter it out.
        # But ensure k does not exceed the total number of products' embeddings.
        k_value = min(num_recommendations + 1, max_k_possible)
        # --- END OF THE FIX ---

        # Get the top K_VALUE similar items
        top_results = torch.topk(cosine_scores, k=k_value, largest=True, sorted=True)
        
        recommended_product_ids = []
        for score, i in zip(top_results[0], top_results[1]):
            # Exclude the query product itself
            # i.item() extracts the scalar value from the PyTorch tensor index
            if i.item() != idx:
                recommended_product_ids.append(self.products_df.iloc[i.item()]['product_id'])
            # Stop once we have enough *unique* recommendations (excluding the query product)
            if len(recommended_product_ids) >= num_recommendations:
                break
        return recommended_product_ids

# Example Usage (for testing purposes, typically used by main.py)
if __name__ == "__main__":
    # Ensure 'data' directory and 'products.json' inside it
    if not os.path.exists('data'):
        os.makedirs('data')
    products_file_path = 'data/products.json' # Define path
    if not os.path.exists(products_file_path): # Use the path variable
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
        with open(products_file_path, 'w', encoding='utf-8') as f: # Use the path variable
            json.dump(dummy_products, f, indent=2)
        print("Created dummy data/products.json for BERT testing.")

    # Initialize BERT model (this might download the model the first time)
    bert_model = BERTModel(products_file_path=products_file_path) # Pass the path

    # Get recommendations for a sample product
    recommended_ids = bert_model.get_recommendations("P001", num_recommendations=3)
    print(f"\nTop 3 BERT recommendations for P001 (Wireless Bluetooth Headphones): {recommended_ids}")

    recommended_ids = bert_model.get_recommendations("P004", num_recommendations=2)
    print(f"Top 2 BERT recommendations for P004 (Gaming Mouse): {recommended_ids}")

    recommended_ids = bert_model.get_recommendations("P999", num_recommendations=5) # Non-existent product
    print(f"Recommendations for P999: {recommended_ids}")