import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained Random Forest model
model = joblib.load('random_forest_model.pkl')
df = pd.read_excel("Catalogue_Clusters.xlsx")

# Create a LabelEncoder for each categorical column
label_encoder = LabelEncoder()

# Create a function to get user input and make predictions
def predict_cluster(age, sexe, taux, situationFamiliale, nbEnfantsAcharge, deuxieme_voiture):
    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'age': [age],
        'sexe': [sexe],
        'taux': [taux],
        'situationFamiliale': [situationFamiliale],
        'nbEnfantsAcharge': [nbEnfantsAcharge],
        '2eme voiture': [deuxieme_voiture]
    })
    
    # Encode categorical columns using the fitted LabelEncoders
    input_data['sexe'] = label_encoder.fit_transform(input_data['sexe'])
    input_data['situationFamiliale'] = label_encoder.fit_transform(input_data['situationFamiliale'])
    input_data['2eme voiture'] = label_encoder.fit_transform(input_data['2eme voiture'])

    # Make predictions
    prediction = model.predict(input_data)

    return prediction[0]


def step1():
    st.header("Personal Information")
    
    age = st.slider('Age', min_value=18, max_value=100, value=30)
    sexe = st.radio('Sexe', ['F', 'M'])
    taux = st.slider('Taux', min_value=500, max_value=1000, value=700)
    situation_familiale = st.selectbox('Situation Familiale', ['Célibataire', 'En Couple', 'Divorcée'])
    nb_enfants_a_charge = st.slider('Nombre d\'enfants à charge', min_value=0, max_value=20, value=1)
    deuxieme_voiture = st.checkbox('Deuxième voiture')
    
    # # Make prediction when the user clicks the "Predict" button
    # if st.form_submit_button('Predict'):
    # # Map boolean values to strings for the 'deuxieme_voiture' feature
    deuxieme_voiture_str = 'true' if deuxieme_voiture else 'false'

    #  # Call the prediction function
    #     prediction_result = predict_cluster(age, sexe, taux, situation_familiale, nb_enfants_a_charge, deuxieme_voiture_str)

    # # Display the prediction result
    #     st.success(f'The predicted cluster is: {prediction_result}')
    return age, sexe, taux, situation_familiale, nb_enfants_a_charge, deuxieme_voiture_str

def step2():
    st.header("Car status")
    selected_occasion = st.selectbox("Select Occasion", ["False", "True"])
    return selected_occasion


def step3():
    st.header("Car colour")
    selected_colour = st.selectbox("Select Colour", ["bleu", "noir", "gris", "rouge", "blanc"])
    submit_button = st.form_submit_button("Submit")
    return selected_colour, submit_button



def show_data(age, 
              sexe, 
              taux, 
              situation_familiale, 
              nb_enfants_a_charge, 
              deuxieme_voiture_str, 
              selected_occasion, 
              selected_colour):
     # Call the prediction function
    prediction_result = predict_cluster(age, sexe, taux, situation_familiale, nb_enfants_a_charge, deuxieme_voiture_str)
    df['occasion'] = df['occasion'].astype(str)
    df['couleur'] = df['couleur'].astype(str)

    filtered_df = df.loc[
        (df['cluster']== prediction_result) &
        (df['occasion']== selected_occasion) &
        (df['couleur']== selected_colour) 
    ]
    selected_attributes = ['marque', 'nom', 'puissance', 'longueur', 'nbPlaces', 'nbPortes', 'prix']
    filtered_df = filtered_df[selected_attributes]
    
    st.success(f' {prediction_result}')

    st.write(filtered_df)




def main():
    st.title("Car Prediction App")

    # Initialize variables with default values
    age = 30
    sexe = 'F'
    taux = 700
    situation_familiale = 'Célibataire'
    nb_enfants_a_charge = 1
    deuxieme_voiture_str = 'false'
    selected_occasion = 'False'
    selected_colour = 'bleu'

    with st.form(key="my_form"):
        step = st.session_state.step if "step" in st.session_state else 1

        if step == 1:
            age, sexe, taux, situation_familiale, nb_enfants_a_charge, deuxieme_voiture_str = step1()
        elif step == 2:
            selected_occasion = step2()
        elif step == 3:
            selected_colour, submit_button = step3()
            if submit_button:
                show_data(age, 
                          sexe, 
                          taux, 
                          situation_familiale, 
                          nb_enfants_a_charge, 
                          deuxieme_voiture_str, 
                          selected_occasion, 
                          selected_colour)

        # Navigation buttons
        if step < 3:
            next_button = st.form_submit_button("Next")
            if next_button:
                st.session_state.step = step + 1

        if step > 1:
            prev_button = st.form_submit_button("Previous")
            if prev_button:
                st.session_state.step = step - 1

if __name__ == "__main__":
    main()
