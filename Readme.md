Advanced Hybrid Product Recommender (with Streamlit UI)

🚀 Overview

The Advanced Hybrid Product Recommender (with Streamlit UI) is a comprehensive full-stack application designed to deliver highly accurate and relevant product recommendations. It achieves this by employing a sophisticated hybrid AI approach in its backend, which intelligently combines the strengths of multiple recommendation models, and presents them through an intuitive Streamlit frontend. This provides a robust and intelligent recommendation experience.

This system is built with:
•	FastAPI for a high-performance, asynchronous backend API.
•	Streamlit for an intuitive and interactive frontend user interface.
•	Various AI models for diverse recommendation strategies.

✨ Features

•	Hybrid Recommendation Logic: Combines scores from multiple models for comprehensive and accurate suggestions.
o	TF-IDF (Content-Based)
o	BERT (Content-Based)
o	Collaborative Filtering (Item-Similarity)
o	Popularity-Based
o	Sentiment Analysis
•	Scalable Backend: Built with FastAPI, ready for production-grade deployment.
•	Interactive Streamlit Frontend: A user-friendly interface for exploring recommendations, applying user/product context, and even adjusting model weights.
•	Modular Design: Easily integrate or swap out recommendation models.
•	CORS Enabled: Seamless integration between frontend and backend.

🛠️ Technologies Used


•	Backend:
o	Python 3.10+
o	FastAPI
o	Uvicorn (ASGI server)
o	Pandas (for data handling)
o	scikit-learn (for TF-IDF)
o	Hugging Face Transformers / sentence-transformers (for BERT embeddings)
o	NLTK / textblob (for sentiment analysis)
o	NumPy
•	Frontend:
o	Streamlit
o	Requests (for API calls)


📂 Project Structure
.
├── api/
│   ├── main.py                 # FastAPI application entry point
│   └── __init__.py             # Makes 'api' a Python package
├── models/
│   ├── hybrid_model.py         # Orchestrates all recommendation models
│   ├── tfidf_model.py          # TF-IDF based recommender
│   ├── bert_model.py           # BERT-based content recommender
│   ├── popularity_model.py     # Popularity-based recommender
│   ├── sentiment_analysis_model.py # Sentiment analysis module
│   ├── item_similarity_model.py # Collaborative Filtering (Item-based) recommender
│   └── __init__.py             # Makes 'models' a Python package
├── data/
│   ├── products.json           # Sample product data (IDs, titles, descriptions, categories, image_urls)
│   ├── ratings.json            # Sample user-product ratings
│   └── reviews.json            # Sample product reviews for sentiment
├── app.py                      # Streamlit frontend application
├── requirements.txt            # Python dependencies
└── README.md                   # This file

⚙️ Setup and Installation

Follow these steps to get the project up and running on your local machine.
1. Clone the Repository
Bash
git clone https://github.com/Gopikakavi/-Advanced-Hybrid-Product-Recommender-.git
cd -Advanced-Hybrid-Product-Recommender-

2. Create a Virtual Environment (Recommended)

Bash
    python -m venv venv

3. Activate the Virtual Environment
•	On Windows:
Bash
.\venv\Scripts\activate
•	On macOS/Linux:
Bash
source venv/bin/activate

4. Install Dependencies
Install all the required Python packages:
Bash
pip install -r requirements.txt

5. Prepare Data
Ensure you have your data files (products.json, ratings.json, reviews.json) in the data/ directory. Dummy data files are included for initial setup. You can replace these with your own datasets.


🚀 Running the Application
This project consists of two main parts: the FastAPI backend and the Streamlit frontend. Both need to be running simultaneously.

1. Start the FastAPI Backend
Open your first terminal, activate the virtual environment, navigate to the project root, and run:
Bash
(venv) C:\path\to\your\project> uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
You should see output indicating that Uvicorn is running, typically on http://0.0.0.0:8000. You can access the API documentation (Swagger UI) at http://localhost:8000/docs.

2. Start the Streamlit Frontend
Open a second terminal, activate the same virtual environment, navigate to the project root, and run:
Bash
(venv) C:\path\to\your\project> streamlit run app.py

This will open the Streamlit application in your web browser (usually at http://localhost:8501).
________________________________________
📊 Usage
Once both the backend and frontend are running:
1.	Interact with the Streamlit App:
o	Use the sidebar to input a User ID or select a Product.
o	Adjust the Number of Recommendations.
o	(Optional) Enable Custom Weights to fine-tune the influence of each recommendation model (TF-IDF, BERT, Collaborative, Popularity, Sentiment).
o	Click "Get Recommendations" to see the results.
2.	Test the FastAPI API Directly (Optional):
o	Open your browser and go to http://localhost:8000/docs to access the interactive API documentation.
o	You can directly test the /recommend endpoint by providing a JSON request body.
________________________________________
📝 Extending and Customizing
•	Adding New Models: Create a new Python file in the models/ directory (e.g., new_model.py), implement a get_recommendations method, and then integrate it into hybrid_model.py and update the weights in api/main.py and app.py.
•	Data Updates: Replace the products.json, ratings.json, and reviews.json files with your own datasets. Ensure their structure matches the expected format.
•	Model Tuning: Experiment with different weights in hybrid_model.py or directly through the Streamlit UI to find the optimal combination for your data.
•	Deployment: For production, consider containerizing your FastAPI application with Docker and deploying it to cloud platforms like AWS, Google Cloud, or Azure.

