
import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('outfit_recommendation_model.pkl')

# Function to recommend outfit based on user input
def recommend_outfit(user_input):
    # Process user input (example: convert to DataFrame if needed)
    user_data = pd.DataFrame([user_input])

    # Predict outfit recommendation
    prediction = model.predict(user_data)
    return prediction[0]

# Streamlit app code
st.title('Personalized Outfit Recommendation System')

# Example user input (replace with actual input form or widget)
user_input = {
    'size': 'M',
    'body_type': 'hourglass',
    'skin_tone': 'fair',
    'style': 'casual'
}

# Display user input
st.subheader('User Input')
st.write(user_input)

# Show recommended outfit
recommended_outfit = recommend_outfit(user_input)
st.subheader('Recommended Outfit')
st.write(recommended_outfit)
