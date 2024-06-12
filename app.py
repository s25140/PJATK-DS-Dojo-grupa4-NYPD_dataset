import joblib
import numpy as np
import pandas as pd
import streamlit as st

data = pd.read_csv('data/selected_data.csv')

X_test = joblib.load('data/X_test.pkl')
y_test = joblib.load('data/y_test.pkl')

model = joblib.load('models/lgbm_model.pkl')

encoders = {'time_of_call': None, 'vic_age_group': None, 'vic_race': None,
            'vic_sex': None,
            'suspect_race': None, 'suspect_sex': None, 'precinct': None,
            'borough': None, 'location_of_occurrence': None,
            'premises': None,
            'offence_description': None}

unique_sets = encoders.copy()

for column_name in data.columns:
    unique_sets[column_name] = set(data[column_name])

unique_sets.pop('time_of_call')


for name, encoder in encoders.items():
    encoders[name] = joblib.load(f'encoders/{name}.pkl')


st.title('NYC 911 offence predictor')
st.write('-------')

st.header('Examples')
st.dataframe(data.sample(5))
st.write('-------')

st.header('Predict yourself')

current_time = st.selectbox('**Current time**', options=range(0, 24))

col1, col2, col3 = st.columns(3)
with col1:
    vic_sex = st.selectbox('**Victim sex**', options=unique_sets.get('vic_sex'), index=None)
    vic_age_group = st.selectbox('**Victim age group**', options=unique_sets.get('vic_age_group'), index=None)
    vic_race = st.selectbox('**Victim race**', options=unique_sets.get('vic_race'), index=None)
with col2:
    suspect_race = st.selectbox('**Suspect race**', options=unique_sets.get('suspect_race'), index=None)
    suspect_sex = st.selectbox('**Suspect sex**', options=unique_sets.get('suspect_sex'), index=None)
    premises = st.selectbox('**Premises**', options=unique_sets.get('premises'), index=None)
with col3:
    precinct = st.selectbox('**Precinct**', options=unique_sets.get('precinct'), index=None)
    borough = st.selectbox('**Borough**', options=unique_sets.get('borough'), index=None)
    location_of_occurrence = st.selectbox('**Location of occurrence**', options=unique_sets.get('location_of_occurrence'), index=None)

predict = st.button('predict')
if predict:
    features = [
        encoders.get('time_of_call').transform([[current_time]]).flatten(),
        encoders.get('vic_age_group').transform([[vic_age_group]]).flatten(),
        encoders.get('vic_race').transform([[vic_race]]).flatten(),
        encoders.get('vic_sex').transform([[vic_sex]]).flatten(),
        encoders.get('suspect_race').transform([[suspect_race]]).flatten(),
        encoders.get('suspect_sex').transform([[suspect_sex]]).flatten(),
        encoders.get('precinct').transform([[precinct]]).flatten(),
        encoders.get('borough').transform([[borough]]).flatten(),
        encoders.get('location_of_occurrence').transform([[location_of_occurrence]]).flatten(),
        encoders.get('premises').transform([[premises]]).flatten()
    ]

    # Concatenate all features into a single 2D array
    features = np.concatenate(features).reshape(1, -1)

    probabilities = model.predict_proba(features)[0]

    # Get indices of the top 5 probabilities
    top_indices = probabilities.argsort()[-5:][::-1]

    # Get the top 5 probabilities
    top_probabilities = probabilities[top_indices]

    # Display the top 5 events and their probabilities
    st.header("Top 4 Most Probable Events:")
    st.write('----')
    col1, col2 = st.columns(2)
    with col1:
        for i in range(2):
            st.write(f"**{encoders.get('offence_description').inverse_transform([top_indices[i]])[0]}**: **{round(top_probabilities[i]*100)}%**")
    with col2:
        for i in range(2, 4):
            st.write(
                f"**{encoders.get('offence_description').inverse_transform([top_indices[i]])[0]}**: **{round(top_probabilities[i] * 100)}%**")

