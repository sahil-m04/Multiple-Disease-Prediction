
"""
@author: Sahil Mehta
"""


import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Load the models
heart_disease_model = pickle.load(open('D:/Multiple Disease Prediction/savedmodel/heart_disease_model.sav', 'rb'))
diabetes_model = pickle.load(open('D:/Multiple Disease Prediction/savedmodel/diabetes_prediction_trained_model.sav', 'rb'))
scaler = pickle.load(open('D:/Multiple Disease Prediction/savedmodel/scaler.sav', 'rb'))


# Initialize session state for inputs

if 'heart_inputs' not in st.session_state:
    st.session_state.heart_inputs = {
        'age': '',
        'sex': '',
        'cp': '',
        'trestbps': '',
        'chol': '',
        'fbs': '',
        'restecg': '',
        'thalach': '',
        'exang': '',
        'oldpeak': '',
        'slope': '',
        'ca': '',
        'thal': ''
    }


def clear_heart_inputs():
    st.session_state.heart_inputs = {
        'age': '',
        'sex': '',
        'cp': '',
        'trestbps': '',
        'chol': '',
        'fbs': '',
        'restecg': '',
        'thalach': '',
        'exang': '',
        'oldpeak': '',
        'slope': '',
        'ca': '',
        'thal': ''
    }

if 'diabetes_inputs' not in st.session_state:
    st.session_state.diabetes_inputs = {
        'Pregnancies': '',
        'Glucose': '',
        'BloodPressure': '',
        'SkinThickness': '',
        'Insulin': '',
        'BMI': '',
        'DiabetesPedigreeFunction': '',
        'Age': ''
    }

def clear_diabetes_inputs():
    st.session_state.diabetes_inputs = {
        'Pregnancies': '',
        'Glucose': '',
        'BloodPressure': '',
        'SkinThickness': '',
        'Insulin': '',
        'BMI': '',
        'DiabetesPedigreeFunction': '',
        'Age': ''
    }

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Heart Disease Prediction','Diabetes Prediction'],
                           menu_icon='hospital-fill',
                           icons=['heart','activity'],
                           default_index=0)


if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.session_state.heart_inputs['age'] = st.text_input('Age', st.session_state.heart_inputs['age'])

    with col2:
        st.session_state.heart_inputs['sex'] = st.text_input('Sex', st.session_state.heart_inputs['sex'])

    with col3:
        st.session_state.heart_inputs['cp'] = st.text_input('Chest Pain types', st.session_state.heart_inputs['cp'])

    with col1:
        st.session_state.heart_inputs['trestbps'] = st.text_input('Resting Blood Pressure', st.session_state.heart_inputs['trestbps'])

    with col2:
        st.session_state.heart_inputs['chol'] = st.text_input('Serum Cholestoral in mg/dl', st.session_state.heart_inputs['chol'])

    with col3:
        st.session_state.heart_inputs['fbs'] = st.text_input('Fasting Blood Sugar > 120 mg/dl', st.session_state.heart_inputs['fbs'])

    with col1:
        st.session_state.heart_inputs['restecg'] = st.text_input('Resting Electrocardiographic results', st.session_state.heart_inputs['restecg'])

    with col2:
        st.session_state.heart_inputs['thalach'] = st.text_input('Maximum Heart Rate achieved', st.session_state.heart_inputs['thalach'])

    with col3:
        st.session_state.heart_inputs['exang'] = st.text_input('Exercise Induced Angina', st.session_state.heart_inputs['exang'])

    with col1:
        st.session_state.heart_inputs['oldpeak'] = st.text_input('ST depression induced by exercise', st.session_state.heart_inputs['oldpeak'])

    with col2:
        st.session_state.heart_inputs['slope'] = st.text_input('Slope of the peak exercise ST segment', st.session_state.heart_inputs['slope'])

    with col3:
        st.session_state.heart_inputs['ca'] = st.text_input('Major vessels colored by fluoroscopy', st.session_state.heart_inputs['ca'])

    with col1:
        st.session_state.heart_inputs['thal'] = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect', st.session_state.heart_inputs['thal'])

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            if any(not field for field in st.session_state.heart_inputs.values()):
                st.error('Please enter values for all fields')
            else:
                user_input = [float(st.session_state.heart_inputs[field]) for field in st.session_state.heart_inputs]
                heart_prediction = heart_disease_model.predict([user_input])

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'

            st.success(heart_diagnosis)

        except ValueError:
            st.error('Please enter valid numerical values')
        except Exception as e:
            st.error(f'An unexpected error occurred: {e}')

    if st.button('Clear'):
        clear_heart_inputs()

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction Model')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.session_state.diabetes_inputs['Pregnancies'] = st.text_input('Number of Pregnancies', st.session_state.diabetes_inputs['Pregnancies'])

    with col2:
        st.session_state.diabetes_inputs['Glucose'] = st.text_input('Glucose Level', st.session_state.diabetes_inputs['Glucose'])

    with col3:
        st.session_state.diabetes_inputs['BloodPressure'] = st.text_input('Blood Pressure value', st.session_state.diabetes_inputs['BloodPressure'])

    with col1:
        st.session_state.diabetes_inputs['SkinThickness'] = st.text_input('Skin Thickness Value', st.session_state.diabetes_inputs['SkinThickness'])

    with col2:
        st.session_state.diabetes_inputs['Insulin'] = st.text_input('Insulin Level', st.session_state.diabetes_inputs['Insulin'])

    with col3:
        st.session_state.diabetes_inputs['BMI'] = st.text_input('BMI value', st.session_state.diabetes_inputs['BMI'])

    with col1:
        st.session_state.diabetes_inputs['DiabetesPedigreeFunction'] = st.text_input('Diabetes Pedigree Function Value', st.session_state.diabetes_inputs['DiabetesPedigreeFunction'])

    with col2:
        st.session_state.diabetes_inputs['Age'] = st.text_input('Age of the Person', st.session_state.diabetes_inputs['Age'])

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            if any(not field for field in st.session_state.diabetes_inputs.values()):
                st.error('Please enter values for all fields')
            else:
                user_input = [float(st.session_state.diabetes_inputs[field]) for field in st.session_state.diabetes_inputs]
                input_data_as_numpy_array = np.asarray(user_input)
                input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
                input_data_df = pd.DataFrame(input_data_reshape, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

                std_data = scaler.transform(input_data_df)
                std_data_df = pd.DataFrame(std_data, columns=input_data_df.columns)

                diab_prediction = diabetes_model.predict(std_data_df)

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'

            st.success(diab_diagnosis)

        except ValueError:
            st.error('Please enter valid numerical values')
        except Exception as e:
            st.error(f'An unexpected error occurred: {e}')

    if st.button('Clear'):
        clear_diabetes_inputs()

