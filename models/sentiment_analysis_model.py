import json
import os
import pandas as pd
from transformers import pipeline

class SentimentAnalysisModel:
    """
    A class to perform sentiment analysis on product reviews and provide sentiment scores for products.
    It uses a pre-trained transformer model to classify text sentiment.
    """
    def __init__(self, products_file_path='data/products.json', reviews_file_path='data/reviews.json', model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        """
        Initializes the SentimentAnalysisModel.
        Args:
            products_file_path (str): The path to the products JSON file (to get product IDs).
            reviews_file_path (str): The path to the reviews JSON file.
            model_name (str): The name of the pre-trained sentiment analysis model.
                              'distilbert-base-uncased-finetuned-sst-2-english' is a good choice for general sentiment.
        """
        self.products_file_path = products_file_path
        self.reviews_file_path = reviews_file_path
        self.model_name = model_name
        self.products_df = None
        self.reviews_df = None
        self.sentiment_pipeline = None
        self.product_sentiment_scores = {} # Stores calculated average sentiment for each product

        self._load_data()
        self._initialize_sentiment_pipeline()
        self._calculate_product_sentiments()

    def _load_data(self):
        """
        Loads product and reviews data from the specified JSON files.
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
                print(f"Successfully loaded {len(self.products_df)} products for Sentiment Analysis.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {self.products_file_path}. Is it valid JSON?")
                self.products_df = pd.DataFrame(columns=['product_id', 'title'])
            except Exception as e:
                print(f"An unexpected error occurred while loading products for Sentiment Analysis: {e}")
                self.products_df = pd.DataFrame(columns=['product_id', 'title'])

        # Load reviews data
        if not os.path.exists(self.reviews_file_path):
            print(f"Error: Reviews file not found at {self.reviews_file_path}")
            self.reviews_df = pd.DataFrame(columns=['product_id', 'review_text'])
        else:
            try:
                with open(self.reviews_file_path, 'r', encoding='utf-8') as f:
                    reviews_data = json.load(f)
                self.reviews_df = pd.DataFrame(reviews_data)
                # Ensure review_text column exists and handle potential NaNs
                if 'review_text' not in self.reviews_df.columns:
                    self.reviews_df['review_text'] = ''
                self.reviews_df['review_text'] = self.reviews_df['review_text'].fillna('')
                print(f"Successfully loaded {len(self.reviews_df)} reviews for Sentiment Analysis.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {self.reviews_file_path}. Is it valid JSON?")
                self.reviews_df = pd.DataFrame(columns=['product_id', 'review_text'])
            except Exception as e:
                print(f"An unexpected error occurred while loading reviews for Sentiment Analysis: {e}")
                self.reviews_df = pd.DataFrame(columns=['product_id', 'review_text'])

    def _initialize_sentiment_pipeline(self):
        """
        Initializes the pre-trained sentiment analysis pipeline.
        """
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model_name)
            print(f"Sentiment analysis pipeline initialized with model '{self.model_name}'.")
        except Exception as e:
            print(f"Error initializing sentiment analysis pipeline: {e}")
            self.sentiment_pipeline = None
    def _get_single_sentiment_score(self, text):
        """
        Gets a sentiment score for a single piece of text.
        Returns a score between 0 and 1, where 1 is positive.
        """
        if self.sentiment_pipeline is None or not text.strip():
            return 0.5 # Neutral if pipeline not ready or text is empty
        
        # This line was incorrectly indented. It should execute only if the above 'if' condition is FALSE.
        result = self.sentiment_pipeline(text, truncation=True)[0]
        
        if result['label'] == 'POSITIVE':
            return result['score']
        elif result['label'] == 'NEGATIVE':
            return 1 - result['score'] # Convert negative score to be closer to 0
        return 0.5 # Neutral

    def _calculate_product_sentiments(self):
        """
        Calculates the average sentiment score for each product based on its reviews.
        Stores results in self.product_sentiment_scores.
        """
        if self.reviews_df is None or self.reviews_df.empty or self.sentiment_pipeline is None:
            print("Cannot calculate product sentiments: reviews data or pipeline not available.")
            return

        print("Calculating average sentiment for products...")
        product_reviews = self.reviews_df.groupby('product_id')['review_text'].apply(list).to_dict()

        for product_id, reviews_list in product_reviews.items():
            # Join all reviews for a product into a single string to get an overall sentiment
            combined_reviews_text = " ".join(reviews_list)
            if combined_reviews_text.strip():
                score = self._get_single_sentiment_score(combined_reviews_text)
                self.product_sentiment_scores[product_id] = score
            else:
                self.product_sentiment_scores[product_id] = 0.5 # Neutral if no reviews

        print(f"Calculated sentiment scores for {len(self.product_sentiment_scores)} products.")

    def get_sentiment_score(self, product_id):
        """
        Retrieves the pre-calculated sentiment score for a given product.
        Args:
            product_id (str): The ID of the product.
        Returns:
            float: The average sentiment score for the product (0 to 1), or 0.5 if not found.
        """
        return self.product_sentiment_scores.get(product_id, 0.5)

    def get_top_products_by_sentiment(self, num_products=5):
        """
        Returns a list of product IDs sorted by their sentiment scores (highest first).
        Args:
            num_products (int): The number of top products to return.
        Returns:
            list: A list of (product_id, sentiment_score) tuples, sorted by score.
        """
        if not self.product_sentiment_scores:
            return []

        sorted_products = sorted(
            self.product_sentiment_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )
        return sorted_products[:num_products]


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
            {"product_id": "P001", "title": "Wireless Bluetooth Headphones", "category": "Audio"},
            {"product_id": "P002", "title": "USB-C Charging Cable", "category": "Accessories"},
            {"product_id": "P003", "title": "Smartphone Stand", "category": "Accessories"},
            {"product_id": "P004", "title": "Portable Power Bank", "category": "Accessories"},
            {"product_id": "P005", "title": "Gaming Mouse", "category": "Gaming"}
        ]
        with open(products_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_products, f, indent=2)
        print(f"Created dummy {products_file}.")

    # Dummy reviews.json
    reviews_file = os.path.join(data_dir, 'reviews.json')
    if not os.path.exists(reviews_file):
        dummy_reviews = [
            {"product_id": "P001", "review_text": "These headphones are amazing, great sound quality!", "rating": 5},
            {"product_id": "P001", "review_text": "Very comfortable, but battery life could be better.", "rating": 4},
            {"product_id": "P002", "review_text": "Good cable, charges fast. No issues.", "rating": 4},
            {"product_id": "P003", "review_text": "Excellent stand, very sturdy and looks premium.", "rating": 5},
            {"product_id": "P004", "review_text": "Works as expected, but feels a bit cheap.", "rating": 3},
            {"product_id": "P004", "review_text": "Worst power bank ever. Died after a week.", "rating": 1},
            {"product_id": "P005", "review_text": "Fantastic mouse for gaming, very responsive and comfortable.", "rating": 5},
            {"product_id": "P005", "review_text": "Mediocre mouse, not worth the price.", "rating": 2},
        ]
        with open(reviews_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_reviews, f, indent=2)
        print(f"Created dummy {reviews_file}.")

    print("\n--- Initializing SentimentAnalysisModel (this might download a model) ---")
    sentiment_model = SentimentAnalysisModel(products_file_path=products_file, reviews_file_path=reviews_file)

    print("\n--- Testing Sentiment Scores ---")
    print(f"Sentiment for P001 (Headphones): {sentiment_model.get_sentiment_score('P001'):.4f}")
    print(f"Sentiment for P002 (Cable): {sentiment_model.get_sentiment_score('P002'):.4f}")
    print(f"Sentiment for P003 (Stand): {sentiment_model.get_sentiment_score('P003'):.4f}")
    print(f"Sentiment for P004 (Power Bank - mixed reviews): {sentiment_model.get_sentiment_score('P004'):.4f}")
    print(f"Sentiment for P005 (Gaming Mouse - mixed reviews): {sentiment_model.get_sentiment_score('P005'):.4f}")
    print(f"Sentiment for P999 (Non-existent product): {sentiment_model.get_sentiment_score('P999'):.4f}")

    print("\n--- Top Products by Sentiment ---")
    top_sentiment_products = sentiment_model.get_top_products_by_sentiment(num_products=3)
    for pid, score in top_sentiment_products:
        print(f"Product ID: {pid}, Sentiment Score: {score:.4f}")