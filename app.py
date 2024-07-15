import streamlit as st
import joblib
import pandas as pd

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('Fashion-Fit/outfit_recommendation_model.pkl')
    return model

# Define the possible values for each feature
sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
body_types = ['hourglass', 'pear', 'rectangular', 'apple', 'inverted triangle']
skin_tones = ['fair', 'medium', 'dark']
styles = ['casual', 'formal', 'sporty', 'elegant', 'streetwear']

# Streamlit app
st.title("Personalized Outfit Recommendation System")

# Collect user inputs
size = st.selectbox("Select your size", sizes)
body_type = st.selectbox("Select your body type", body_types)
skin_tone = st.selectbox("Select your skin tone", skin_tones)
style = st.selectbox("Select your preferred style", styles)

# Prepare the input data
input_data = pd.DataFrame({
    'size': [size],
    'body_type': [body_type],
    'skin_tone': [skin_tone],
    'style': [style]
})

# Load the model
model = load_model()

# Function to get top N recommendations
def get_top_n_recommendations(model, new_data, n=3):
    probabilities = model.predict_proba(new_data)
    classes = model.classes_
    top_n_indices = probabilities.argsort()[0, -n:][::-1]
    top_n_classes = classes[top_n_indices]
    return top_n_classes

# Make predictions when the button is clicked
if st.button("Get Recommendations"):
    top_recommendations = get_top_n_recommendations(model, input_data)
    st.write("Top Recommendations:")
    for i, recommendation in enumerate(top_recommendations, 1):
        st.write(f"{i}. {recommendation}")
