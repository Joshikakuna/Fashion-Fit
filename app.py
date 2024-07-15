import streamlit as st
import joblib
import os

# Assuming the model file path
model_path = '/mount/src/fashion-fit/outfit_recommendation_model.pkl'

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please check the file path.")
    st.stop()

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Streamlit app
st.title("Personalized Outfit Recommendation System")

# Load the model
model = load_model()

# Function to get top N recommendations
def get_top_n_recommendations(model, new_data, n=3):
    probabilities = model.predict_proba(new_data)
    classes = model.classes_
    top_n_indices = probabilities.argsort()[0, -n:][::-1]
    top_n_classes = classes[top_n_indices]
    return top_n_classes

# Collect user inputs (example)
sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
size = st.selectbox("Select your size", sizes)

if model:
    if st.button("Get Recommendations"):
        input_data = {'size': [size]}  # Example data
        input_data_df = pd.DataFrame(input_data)
        top_recommendations = get_top_n_recommendations(model, input_data_df)
        st.write("Top Recommendations:")
        for i, recommendation in enumerate(top_recommendations, 1):
            st.write(f"{i}. {recommendation}")


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
model_path = '/content/outfit_recommendation_model.pkl'
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
        input_data_df = pd.DataFrame(input_data)
        top_recommendations = get_top_n_recommendations(model, input_data_df)
        st.write("Top Recommendations:")
        for i, recommendation in enumerate(top_recommendations, 1):
            st.write(f"{i}. {recommendation}")
