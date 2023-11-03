from feature_engineering import get_feature_values
from fastapi import FastAPI

app = FastAPI()

@app.post("/")
async def predict(
        model: object,
        user_id: int,
        livestream_id: int) -> float:
    """
    The prediction function for livestream recommender system

    Arguments:
        model: the trained model object
        user_id: 
        livestream_id: 
    Returns: 
        prediction (float): relevance score
    """

    # retrieve features for user-livestream pair
    features = get_feature_values(user_id, livestream_id)

    # make prediction 
    prediction = model.predict(features)

    return {"score": 1}