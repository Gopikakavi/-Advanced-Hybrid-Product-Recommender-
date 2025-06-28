import pandas as pd
import json
import os
import numpy as np # For handling NaN values

# Input & Output Paths
INPUT_CSV = 'C:\\Users\\asus\\Desktop\\Full Stack AI Recommender System\\amazon.csv'
NORMALIZED_CSV = "normalized_amazon.csv"
PRODUCTS_JSON_PATH = 'data/products.json'
RATINGS_JSON_PATH = 'data/ratings.json'
REVIEWS_JSON_PATH = 'data/reviews.json' # New path for reviews.json

def normalize_csv_for_multiple_reviews(input_csv, output_csv):
    """
    Normalizes a CSV where a single row might contain multiple comma-separated
    user/review/rating entries into separate rows.
    Assumes multi-value columns are PIPE-separated based on the error.
    """
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV file not found at {input_csv}")
        return pd.DataFrame()

    print(f"Loading raw data for normalization from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} rows from {input_csv}.")
    except Exception as e:
        print(f"Error loading raw CSV for normalization: {e}")
        return pd.DataFrame()

    # Define columns that are PIPE-separated lists (based on common Amazon datasets and your error)
    multi_columns = ['user_id', 'review_id', 'review_title', 'review_content', 'rating']

    # List to collect individual review rows
    normalized_rows = []

    for idx, row in df.iterrows():
        try:
            # Safely get values and split by '|', falling back to empty list if NaN or empty
            user_ids = str(row.get('user_id', '')).split('|') if pd.notna(row.get('user_id')) and str(row.get('user_id')).strip() else ['']
            review_ids = str(row.get('review_id', '')).split('|') if pd.notna(row.get('review_id')) and str(row.get('review_id')).strip() else ['']
            review_titles = str(row.get('review_title', '')).split('|') if pd.notna(row.get('review_title')) and str(row.get('review_title')).strip() else ['']
            review_contents = str(row.get('review_content', '')).split('|') if pd.notna(row.get('review_content')) and str(row.get('review_content')).strip() else ['']
            ratings = str(row.get('rating', '')).split('|') if pd.notna(row.get('rating')) and str(row.get('rating')).strip() else ['']

            # Find the maximum length among the split lists to ensure all sub-reviews are captured
            max_len = max(len(user_ids), len(review_ids), len(review_titles), len(review_contents), len(ratings))
            
            # If all lists are essentially empty (e.g., original cell was NaN/empty string and split resulted in [''])
            # and max_len is 1, it means there's no actual data to normalize, so skip or handle as a single empty review
            if max_len == 1 and all(len(l) == 1 and l[0] == '' for l in [user_ids, review_ids, review_titles, review_contents, ratings]):
                # Handle cases where a row has no review data, but still needs a product entry
                # We can create a single 'dummy' review for product metadata extraction if needed later
                pass # This is implicitly handled by the loop if max_len is correctly 0, but added for clarity

            for i in range(max_len):
                # Safely get the i-th element, defaulting to empty string if index out of bounds
                current_user_id = user_ids[i].strip() if i < len(user_ids) else ''
                current_review_id = review_ids[i].strip() if i < len(review_ids) else ''
                current_review_title = review_titles[i].strip() if i < len(review_titles) else ''
                current_review_content = review_contents[i].strip() if i < len(review_contents) else ''
                current_rating_str = ratings[i].strip() if i < len(ratings) else ''

                # Skip if core review data (user, product, rating) is missing for this sub-review
                if not current_user_id and not current_review_id and not current_rating_str:
                    continue

                normalized_rows.append({
                    "product_id": row['product_id'],
                    "product_name": row['product_name'],
                    "category": row['category'],
                    "about_product": row.get('about_product', ''),
                    "img_link": row.get('img_link', ''),
                    # Normalized review details
                    "user_id": current_user_id,
                    "review_id": current_review_id,
                    "review_title": current_review_title,
                    "review_content": current_review_content,
                    "rating": current_rating_str, # Keep as string for now, convert to float later
                })
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")
            # Optionally, log the problematic row to inspect later
            # print(row)

    # Create a new DataFrame from normalized rows
    normalized_df = pd.DataFrame(normalized_rows)

    # Save to CSV
    normalized_df.to_csv(output_csv, index=False)
    print(f"✅ Normalized data saved to {output_csv}")
    return normalized_df

def prepare_amazon_data(raw_csv_path, products_json_path, ratings_json_path, reviews_json_path):
    """
    Processes a raw Amazon reviews CSV into products.json, ratings.json, and reviews.json
    formats required by the recommendation system.

    Args:
        raw_csv_path (str): Path to the raw Amazon CSV file.
        products_json_path (str): Path to save the processed products.json.
        ratings_json_path (str): Path to save the processed ratings.json.
        reviews_json_path (str): Path to save the processed reviews.json.
    """
    if not os.path.exists(raw_csv_path):
        print(f"Error: Normalized CSV file not found at {raw_csv_path}")
        print("Please ensure 'normalized_amazon.csv' exists after the first step, or specify the correct path.")
        return

    print(f"Loading data from {raw_csv_path} for JSON conversion...")
    try:
        # Load the raw CSV data using the provided column names.
        expected_columns = [
            'product_id', 'product_name', 'category', 'rating',
            'about_product', 'user_id', 'review_title', 'review_content', 'img_link', 'review_id' # Added review_id
        ]
        
        # Use only necessary columns to avoid loading too much data if the CSV is huge
        df = pd.read_csv(raw_csv_path)

        # Check for missing essential columns
        if not all(col in df.columns for col in expected_columns):
            missing_cols = [col for col in expected_columns if col not in df.columns]
            print(f"Error: Missing essential columns in CSV: {missing_cols}")
            print("Please ensure your 'normalized_amazon.csv' contains all required columns after normalization.")
            return

        print(f"Loaded {len(df)} rows from {raw_csv_path}.")

    except Exception as e:
        print(f"Error loading CSV for JSON conversion: {e}")
        return

    # --- Prepare Ratings Data (ratings.json) ---
    print("Preparing ratings data...")
    ratings_df = df[['user_id', 'product_id', 'rating']].copy()

    # Clean the 'rating' column by ensuring it's a valid number.
    ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce')
    
    # Drop rows with any NaN values in these critical columns after conversion
    ratings_df.dropna(subset=['user_id', 'product_id', 'rating'], inplace=True)
    
    # Ensure correct data types (after dropping NaNs)
    ratings_df['user_id'] = ratings_df['user_id'].astype(str)
    ratings_df['product_id'] = ratings_df['product_id'].astype(str)
    ratings_df['rating'] = ratings_df['rating'].astype(float)

    # Remove duplicate ratings for the same user-product pair (keep the latest or first)
    ratings_df.drop_duplicates(subset=['user_id', 'product_id'], inplace=True)

    # Convert to list of dictionaries and save
    ratings_data = ratings_df.to_dict(orient='records')
    with open(ratings_json_path, 'w', encoding='utf-8') as f:
        json.dump(ratings_data, f, indent=2)
    print(f"Saved {len(ratings_data)} ratings to {ratings_json_path}")

    # --- Prepare Reviews Data (reviews.json) ---
    print("Preparing reviews data...")
    # Select relevant columns for reviews
    reviews_df = df[['product_id', 'review_title', 'review_content', 'rating']].copy()

    # Combine review_title and review_content into a single 'review_text'
    # Handle NaNs by replacing them with empty strings before combining
    reviews_df['review_title'] = reviews_df['review_title'].fillna('').astype(str)
    reviews_df['review_content'] = reviews_df['review_content'].fillna('').astype(str)
    
    # Create the combined review_text column
    reviews_df['review_text'] = reviews_df.apply(
        lambda row: f"{row['review_title'].strip()}. {row['review_content'].strip()}".strip() if row['review_title'].strip() and row['review_content'].strip()
        else row['review_title'].strip() or row['review_content'].strip(),
        axis=1
    )
    
    # Clean the 'rating' column for reviews.json as well
    reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')
    
    # Drop rows where product_id or review_text is missing/empty, or rating is NaN
    reviews_df.dropna(subset=['product_id', 'review_text', 'rating'], inplace=True)
    reviews_df = reviews_df[reviews_df['review_text'].str.strip() != ''] # Ensure review_text is not just whitespace

    # Select final columns and convert to list of dictionaries
    reviews_data = reviews_df[['product_id', 'review_text', 'rating']].to_dict(orient='records')
    
    # Save to reviews.json
    with open(reviews_json_path, 'w', encoding='utf-8') as f:
        json.dump(reviews_data, f, indent=2)
    print(f"Saved {len(reviews_data)} reviews to {reviews_json_path}")


    # --- Prepare Products Data (products.json) ---
    print("Preparing products data...")
    products_list = []
    
    # Get unique products based on 'product_id'
    unique_products = df.groupby('product_id').first().reset_index()

    for index, row in unique_products.iterrows():
        product_id = str(row['product_id'])
        
        # Use 'product_name' as title. Handle potential NaNs.
        title = str(row.get('product_name', f"Product {product_id}")).strip()
        if not title or title.lower() == 'nan':
            title = f"Product {product_id}"

        # Combine 'about_product', 'review_title', and 'review_content' for a richer description
        # Note: For 'products.json', we generally want a *single* representative description per product.
        # This current approach uses the first review's content, which might not be ideal.
        # A better approach might be to combine all reviews for a product or use only 'about_product'.
        # For now, sticking to existing logic for `description` for consistency.
        about_product = str(row.get('about_product', '')).strip()
        
        # For products.json, we want a general description, not necessarily from a specific review.
        # Let's try to just use 'about_product' and fall back to a generic description.
        description = about_product if about_product and about_product.lower() != 'nan' else f"No detailed description available for Product {product_id}."
        
        # Use 'category' column directly from your CSV
        category = str(row.get('category', 'Unknown')).strip()
        if not category or category.lower() == 'nan':
            category = 'Unknown'
        
        # Use 'img_link' directly as image_url. Provide a fallback if it's missing or invalid.
        image_url = str(row.get('img_link', '')).strip()
        if not image_url or image_url.lower() == 'nan':
            image_url = f"https://placehold.co/150x150/EEEEEE/000000?text=No+Image+{product_id}" # Fallback placeholder

        products_list.append({
            "product_id": product_id,
            "title": title,
            "description": description, # Using 'about_product' for main description
            "category": category,
            "image_url": image_url
        })

    # Save to products.json
    with open(products_json_path, 'w', encoding='utf-8') as f:
        json.dump(products_list, f, indent=2)
    print(f"Saved {len(products_list)} unique products to {products_json_path}")

    print("Data preparation complete!")

if __name__ == "__main__":
    # Ensure the 'data' directory exists
    os.makedirs('data', exist_ok=True)
    
    # First, normalize the raw CSV
    normalized_df_output = normalize_csv_for_multiple_reviews(INPUT_CSV, NORMALIZED_CSV)
    
    # Then, prepare JSONs from the normalized CSV if normalization was successful
    if not normalized_df_output.empty:
        # Pass the new REVIEWS_JSON_PATH to the data preparation function
        prepare_amazon_data(NORMALIZED_CSV, PRODUCTS_JSON_PATH, RATINGS_JSON_PATH, REVIEWS_JSON_PATH)
    else:
        print("Normalization failed or resulted in empty data. Skipping JSON conversion.")

    print("\n--- Important Next Steps ---")
    print(f"1. Your '{PRODUCTS_JSON_PATH}', '{RATINGS_JSON_PATH}', and '{REVIEWS_JSON_PATH}' files should now be populated.")
    print("2. You MUST now restart your FastAPI backend (`main.py`) so it loads the new data.")
    print("   Go to the terminal running FastAPI, press CTRL+C, then rerun:")
    print("   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
    print("3. After the backend restarts, refresh your Streamlit frontend.")