import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define possible values for each feature
sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
body_types = ['hourglass', 'pear', 'rectangular', 'apple', 'inverted triangle']
skin_tones = ['fair', 'medium', 'dark']
styles = ['casual', 'formal', 'sporty', 'elegant', 'streetwear']

# Define the model loading function
def load_model():
    try:
        model = joblib.load('outfit_recommendation_model.pkl')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Test model prediction or use it in your Streamlit app
if model:
    # Example usage
    new_data = pd.DataFrame({
        'size': ['M'],
        'body_type': ['hourglass'],
        'skin_tone': ['fair'],
        'style': ['casual']
    })

    prediction = model.predict(new_data)
    print("Model prediction:", prediction)

