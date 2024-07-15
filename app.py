
import streamlit as st
import pandas as pd

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = joblib.load('outfit_recommendation_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Streamlit app
def main():
    st.title("Personalized Outfit Recommendation System")

    # Load the model
    model = load_model()

    if model:
        # Your application logic here
        st.write("Model loaded successfully!")
    else:
        st.error("Failed to load model. Check logs for details.")

if __name__ == '__main__':
    main()
