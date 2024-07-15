import streamlit as st
import joblib
import pandas as pd
import os
import urllib.request

# Define the URL of the model file on GitHub
model_url = 'https://github.com/Joshikakuna/Fashion-Fit/raw/main/outfit_recommendation_model.pkl'

# Function to download the model file
def download_model(url, save_path):
    try:
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return False

# Download the model file to a local directory
model_directory = '/mount/src/fashion-fit/models'
model_path = os.path.join(model_directory, 'outfit_recommendation_model.pkl')

if not os.path.exists(model_path):
    st.info("Downloading model file...")
    if download_model(model_url, model_path):
        st.success("Model file downloaded successfully.")
    else:
        st.error("Failed to download model file. Please check the URL.")

# Debugging: Check current directory and directory contents
current_directory = os.getcwd()
st.write("Current directory:", current_directory)
st.write("Directory contents:", os.listdir(current_directory))

# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    try:
        st.write("Attempting to load model from:", model_path)
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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
input_data = {
    'size': [size],
    'body_type': [body_type],
    'skin_tone': [skin_tone],
    'style': [style]
}

# Load the model
model = load_model(model_path)

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
        input_data_df = pd.DataFrame(input_data)
        top_recommendations = get_top_n_recommendations(model, input_data_df)
        st.write("Top Recommendations:")
        for i, recommendation in enumerate(top_recommendations, 1):
            st.write(f"{i}. {recommendation}")
