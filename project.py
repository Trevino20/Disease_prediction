import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np


# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")


# loading the saved models

diabetes_model =  pickle.load(open('./model/dia/best_mlp','rb'))
# heart_disease_model1 = pickle.load(open('./model/heart/best_rf','rb'))
heart_disease_model = pickle.load(open('./model/heart/best_svm.sav','rb'))



# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart'],
                           default_index=0)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#                     Diabetes Prediction 

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    if st.button('Diabetes Test Result'):
        try:
            # Convert user inputs to float
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]

            # Make prediction
            diab_prediction = diabetes_model.predict([user_input])

            # Display result
            if diab_prediction[0] == 1:
                st.error("The person is predicted to have diabetes")
            else:
                st.success("The person is predicted not to have diabetes")
        except ValueError as e:
            st.error(f"Error: {e}. Please enter valid numeric values.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#                   Heart Disease Prediction Page

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
        # age = int(age)


    with col2:
        gender = st.selectbox('Select your gender:', ('Male', 'Female'))

        if gender == 'Male':
            sex = 0
        else:
            sex = 1


    with col3:
        cp_type = st.selectbox( "Chest Pain types",(" Typical angina","Atypical angina", "Non-anginal pain", "Asymptomatic"))

        cp_1 = False
        cp_2 = False
        cp_3 = False

        if cp_type == "Atypical angina":
                cp_1 = True
        elif cp_type == "Non-anginal pain":
                cp_2 = True
        else:  # Asymptomatic
                cp_3 = True

    with col1:
        trestbps = st.text_input('Resting Blood Pressure',value=0)
        # trestbps = int(trestbps)
       
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        # chol = int(chol)
        

    with col3:
        fbss = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('High', 'Low'))

        if fbss == 'Low':
            fbs = 0
        else:
            fbs = 1
        

    with col1:
        restecg_type = st.selectbox("Resting Electrocardiographic Results",("Normal","Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy"))

        restecg_1 = False
        restecg_2 = False
        if restecg_type == "Having ST-T wave abnormality":
            restecg_1 = True
        else:  # Showing probable or definite left ventricular hypertrophy
            restecg_2 = True



        
       

    with col2:
         thalach = st.text_input('Maximum Heart Rate achieved')
        #  thalach = int(thalach)
         
        
    with col3:
        exangs = st.selectbox('Exercise Induced Angina', ('YES', 'NO'))

        if exangs == 'NO':
            exang = 0
        else:
            exang = 1
         
        

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        # oldpeak = int(oldpeak)
       

    with col2:
        slope = st.selectbox( "Slope of the peak exercise ST segment",("UP Sloping","Flat","Down Sloping"))


        if slope == "UP Sloping":
                slop = 0
        elif cp_type == "Flat":
                slop = 1
        else:  # Asymptomatic
                slop = 2
      


    with col3:
         ca = st.text_input('Major vessels colored by flourosopy')
        #  ca = int(ca)

        

    with col1:
        thal_type = st.selectbox("Thalassemia",("Normal","Fixed defect", "Reversible defect", "Not described"))

        thal_1 = False
        thal_2 = False
        thal_3 = False
        if thal_type == "Fixed defect":
            thal_1 = True
        elif thal_type == "Reversible defect":
            thal_2 = True
        else:  # Not described
            thal_3 = True



    if st.button("Heart Disease Test Result"):
        try:

            features = [
                int(age), int(sex), int(trestbps), int(chol), int(fbs), int(thalach), int(exang), 
                float(oldpeak), int(slop), int(ca), cp_1, cp_2, cp_3, restecg_1, restecg_2, thal_1, thal_2, thal_3]

            features = np.array(features).reshape(1, -1)

            
            
            # Convert features to a numpy array and reshape for prediction
            print("Features array:", features)
            
            # Make prediction
            prediction = heart_disease_model.predict(features)
            lc = [str(i) for i in prediction]
            ans = int("".join(lc))

            # Display result
            if ans == 1:
                st.error("The person is having heart disease")
            elif ans == 0:
                st.success("The person does not have any heart disease")
        except ValueError as e:
            st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")