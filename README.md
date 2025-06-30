🧠 Advanced Hybrid Product Recommender (with Streamlit UI)

🚀 Overview
The Advanced Hybrid Product Recommender is a full-stack application delivering intelligent product recommendations. It combines cutting-edge AI techniques and traditional methods in a hybrid model, paired with an interactive Streamlit UI for an intuitive user experience.

This system is built with:

           ⚡ FastAPI for an asynchronous, high-performance backend API

           🌐 Streamlit for a rich, real-time user interface

            🧠 Multiple AI models for robust, hybrid recommendation logic
✨ Features

🔀 Hybrid Recommendation Logic – Combines multiple models for best results.

🧠 TF-IDF – Recommends based on content similarity using product descriptions.

🤖 BERT Embeddings – Uses deep learning to extract semantic meaning for similarity.

🔥 Popularity-Based – Recommends trending/high-rated products.

👥 Collaborative Filtering (Item-Similarity) – Finds items users with similar behavior liked.

❤️ Sentiment Analysis – Prioritizes products with positive review sentiment.

⚡ Scalable FastAPI Backend – High-performance asynchronous API.

🖥️ Streamlit Frontend – Intuitive and adjustable user interface.

🧩 Modular Design – Easy to extend or modify with new recommendation models.

🌐 CORS Enabled – Smooth frontend-backend communication.

🛠️ Technologies Used

Backend

Python 3.10+

FastAPI, Uvicorn

Pandas, NumPy

scikit-learn – TF-IDF & collaborative filtering

sentence-transformers – BERT embeddings

TextBlob, NLTK – Sentiment analysis

Frontend

Streamlit

Requests

📂 Project Structure: Where Everything Resides!
Navigate our well-structured project with ease:

```text
.
├── api/
│   ├── main.py                     # The API Command Center: Defines all routes and orchestrates backend logic.
│   └── __init__.py                 # Designates 'api' as a Python package.
├── models/
│   ├── hybrid_model.py             # The Maestro: Harmonizes recommendations from all individual models.
│   ├── tfidf_model.py              # TF-IDF Model: Content-based magic with keywords.
│   ├── bert_model.py               # BERT Model: Deep semantic understanding for content.
│   ├── popularity_model.py         # Popularity Model: What's trending right now!
│   ├── sentiment_analysis_model.py # Sentiment Model: Infusing emotional intelligence from reviews.
│   ├── item_similarity_model.py    # Item-Similarity Model: Connecting similar products via user behavior.
│   └── __init__.py                 # Designates 'models' as a Python package.
├── data/
│   ├── products.json               # Your Product Catalog: All product details live here.
│   ├── ratings.json                # User Ratings: The invaluable feedback loop.
│   └── reviews.json                # Product Reviews: Raw text for sentiment analysis.
├── app.py                          # The Streamlit App: Your interactive user interface, brought to life here!
├── requirements.txt                # Dependency List: All necessary Python libraries for smooth operation.
└── README.md                       # This Guide: Your comprehensive manual for the project!
```  
⚙️ Setup and Installation

Follow these steps to get the project up and running on your local machine:

1. Clone the Repository
```text
git clone https://github.com/Gopikakavi/-Advanced-Hybrid-Product-Recommender-.git
cd -Advanced-Hybrid-Product-Recommender
``` 
2. Create a Virtual Environment (Recommended)
bash
```text
python -m venv venv
``` 
3. Activate the Virtual Environment
bash
```text
.\venv\Scripts\activate
``` 
On macOS/Linux:
```text
source venv/bin/activate
```
4. Install Dependencies
```text
pip install -r requirements.txt
```
5. Prepare the Data
   
Ensure your data/ directory contains:

      products.json

      ratings.json

       reviews.json

✅ Sample data is included; feel free to replace it with your own.

🚀 Running the Application
This project has two main components: the FastAPI backend and the Streamlit frontend. Both must be running.

1. Start the FastAPI Backend
Open Terminal 1:
```text
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Visit the API at http://localhost:8000/docs to test the Swagger UI.

2. Start the Streamlit Frontend
Open Terminal 2:
```text
streamlit run app.py
```
This should open the frontend in your browser at http://localhost:8501

📊 Usage

Once the backend and frontend are live:

Enter a User ID or select a Product

Set the number of recommendations

(Optional) Enable Custom Weights for each model

Click "Get Recommendations"

🔧 Extending & Customizing
➕ Add a New Model
Create new_model.py inside models/

Implement a get_recommendations() function

Register it in hybrid_model.py and update api/main.py & app.py to use it

📈 Data Updates

Replace the files in data/:

products.json

ratings.json

reviews.json

✅ Ensure structure matches the original format.

🎛️ Model Tuning
Tweak weights inside hybrid_model.py or dynamically adjust in the Streamlit UI to optimize performance.

🚢 Deployment Tips
For production:

🐳 Dockerize your app

☁️ Deploy using AWS, GCP, or Azure

🌍 Use nginx or traefik as reverse proxy

🔐 Add HTTPS and security best practices

