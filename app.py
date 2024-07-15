import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define possible values for each feature
sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
body_types = ['hourglass', 'pear', 'rectangular', 'apple', 'inverted triangle']
skin_tones = ['fair', 'medium', 'dark']
styles = ['casual', 'formal', 'sporty', 'elegant', 'streetwear']

# Define deterministic outfit recommendations based on body type and style
outfits = {
    'hourglass': {
        'casual': [('fitted top', 'high-waisted pants'), ('crop top', 'wide-leg pants'), ('wrap top', 'jeans')],
        'formal': [('bodycon dress', ''), ('maxi dress', ''), ('pencil dress', '')],
        'sporty': [('fitted top', 'leggings'), ('tank top', 'shorts'), ('hoodie', 'track pants')],
        'elegant': [('fitted top', 'skirt'), ('silk blouse', 'flared pants'), ('wrap top', 'trousers')],
        'streetwear': [('crop top', 'jeans'), ('graphic tee', 'cargo pants'), ('hoodie', 'leggings')]
    },
    'pear': {
        'casual': [('off-shoulder top', 'a-line skirt'), ('ruffled blouse', 'straight pants'), ('boatneck top', 'capri pants')],
        'formal': [('wrap dress', ''), ('floral gown', ''), ('mermaid dress', '')],
        'sporty': [('tank top', 'shorts'), ('crop top', 'joggers'), ('tee', 'athletic leggings')],
        'elegant': [('off-shoulder top', 'pencil skirt'), ('lace blouse', 'flare pants'), ('silk top', 'skirt')],
        'streetwear': [('graphic tee', 'wide-leg pants'), ('sweater', 'skinny jeans'), ('denim jacket', 'leggings')]
    },
    'rectangular': {
        'casual': [('ruffled top', 'straight pants'), ('striped blouse', 'jeans'), ('t-shirt', 'cargo pants')],
        'formal': [('shift dress', ''), ('pleated dress', ''), ('midi dress', '')],
        'sporty': [('t-shirt', 'joggers'), ('tank top', 'track pants'), ('sweatshirt', 'leggings')],
        'elegant': [('ruffled top', 'flare skirt'), ('blouse', 'wide-leg pants'), ('silk blouse', 'trousers')],
        'streetwear': [('sweatshirt', 'skinny jeans'), ('graphic tee', 'boyfriend jeans'), ('plaid shirt', 'cargo pants')]
    },
    'apple': {
        'casual': [('v-neck top', 'wide-leg pants'), ('tank top', 'capri pants'), ('blouse', 'jeans')],
        'formal': [('empire waist dress', ''), ('sheath dress', ''), ('shift dress', '')],
        'sporty': [('v-neck top', 'capri pants'), ('tank top', 'leggings'), ('hoodie', 'track pants')],
        'elegant': [('v-neck top', 'wrap skirt'), ('blouse', 'trousers'), ('silk top', 'pencil skirt')],
        'streetwear': [('tank top', 'cargo pants'), ('crop top', 'jeans'), ('hoodie', 'wide-leg pants')]
    },
    'inverted triangle': {
        'casual': [('scoop neck top', 'flared pants'), ('ruffle top', 'straight pants'), ('striped top', 'wide-leg pants')],
        'formal': [('fit and flare dress', ''), ('sheath dress', ''), ('midi dress', '')],
        'sporty': [('scoop neck top', 'track pants'), ('tank top', 'shorts'), ('hoodie', 'leggings')],
        'elegant': [('scoop neck top', 'a-line skirt'), ('blouse', 'wide-leg pants'), ('silk top', 'flare pants')],
        'streetwear': [('hoodie', 'boyfriend jeans'), ('graphic tee', 'joggers'), ('crop top', 'cargo pants')]
    }
}

# Define color recommendations based on skin tone
color_recommendations = {
    'fair': ['pastel colors', 'jewel tones', 'light pink', 'lavender', 'ivory'],
    'medium': ['earth tones', 'olive green', 'rich yellows', 'coral', 'peach'],
    'dark': ['bold colors', 'bright yellow', 'emerald green', 'royal blue', 'white']
}

# Streamlit app
st.title("Personalized Outfit Recommendation System")

# Collect user inputs
size = st.selectbox("Select your size", sizes)
body_type = st.selectbox("Select your body type", body_types)
skin_tone = st.selectbox("Select your skin tone", skin_tones)
style = st.selectbox("Select your preferred style", styles)

# Load the model
# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = joblib.load('./outfit_recommendation_model.pkl')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        raise e
model = joblib.load(./outfit_recommendation_model.pkl')


model = load_model()

# Function to get top N recommendations
def get_top_n_recommendations(model, new_data, n=3):
    probabilities = model.predict_proba(new_data)
    classes = model.classes_
    top_n_indices = probabilities.argsort()[0, -n:][::-1]
    top_n_classes = classes[top_n_indices]
    return top_n_classes

# Make predictions when the button is clicked
if model:
    if st.button("Get Recommendations"):
        input_data = {
            'size': [size],
            'body_type': [body_type],
            'skin_tone': [skin_tone],
            'style': [style]
        }
        input_data_df = pd.DataFrame(input_data)
        top_recommendations = get_top_n_recommendations(model, input_data_df)
        st.write("Top Recommendations:")
        for i, recommendation in enumerate(top_recommendations, 1):
            st.write(f"{i}. {recommendation}")
