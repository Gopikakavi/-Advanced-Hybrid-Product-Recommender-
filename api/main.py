from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, List
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import datetime # Import datetime to potentially show current time for logging

# Import your hybrid model
# Ensure this path is correct relative to where you run uvicorn (project root)
from models.hybrid_model import HybridModel

# Initialize FastAPI app
app = FastAPI(title="Hybrid Recommender API")

# Allow frontend access (CORS for Streamlit to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for development, restrict in production
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Request schema for recommendations
class RecommendationRequest(BaseModel):
    user_id: Optional[str] = None
    product_id: Optional[str] = None
    num_recommendations: int = 5
    # The 'weights' dictionary should now only contain active models:
    # tfidf, bert, collaborative, popularity, sentiment
    weights: Optional[Dict[str, float]] = None

# Response schema for individual product details
class ProductResponse(BaseModel):
    product_id: str
    title: str
    description: str
    category: Optional[str] = None # Make category optional as per products.json
    image_url: Optional[str] = None # Make image_url optional as per products.json

# Response schema for the recommendation list
class RecommendationResponse(BaseModel):
    recommendations: List[ProductResponse]

# Load your hybrid model at startup
# This ensures the model is loaded only once when the API starts,
# not on every request, which is crucial for performance.
print(f"{datetime.datetime.now()}: ⏳ Loading hybrid model...")
try:
    model = HybridModel()
    print(f"{datetime.datetime.now()}: ✅ Model ready.")
except Exception as e:
    print(f"{datetime.datetime.now()}: ❌ Error loading hybrid model: {e}")
    # You might want to raise the exception or handle it more gracefully
    # depending on how you want your application to behave if model loading fails.
    # For now, it will likely prevent the app from starting properly.
    model = None # Set model to None if loading fails

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Endpoint to get hybrid product recommendations.
    Accepts user_id, product_id, number of recommendations, and optional custom weights.
    """
    if model is None:
        return {"recommendations": []} # Return empty if model failed to load

    print(f"{datetime.datetime.now()}: Received recommendation request: {request.dict()}")
    try:
        # Get raw product_ids from the hybrid model
        recommended_product_ids = model.get_hybrid_recommendations(
            user_id=request.user_id,
            product_id=request.product_id,
            num_recommendations=request.num_recommendations,
            weights=request.weights
        )

        # Retrieve full product details for the recommended IDs
        # The hybrid model's get_product_details method should return
        # a list of dictionaries matching the ProductResponse schema.
        recommendations_details = model.get_product_details(recommended_product_ids)

        print(f"{datetime.datetime.now()}: Generated {len(recommendations_details)} recommendations.")
        return {"recommendations": recommendations_details}

    except Exception as e:
        print(f"{datetime.datetime.now()}: ❌ Error in recommendation logic: {e}")
        # In a production environment, you might log the full traceback
        # and return a more generic error message to the client.
        return {"recommendations": []}

# Optional: run directly for local testing/development
if __name__ == "__main__":
    # Note: When running with `uvicorn api.main:app`, the 'api' part
    # means it's looking for 'main.py' inside the 'api' directory.
    # When running directly from 'api' directory, you'd use 'main:app'.
    # If running from project root, use 'api.main:app'.
    # For simplicity and consistency with your common usage, assuming it's run from project root:
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
    # If this file was in the root directory, it would be: uvicorn.run("main:app", ...)