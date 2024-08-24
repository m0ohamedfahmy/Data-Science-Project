## Import Libraries
import streamlit as st
import joblib
import numpy as np
from utils import process_now

# Load the model
model =joblib.load(r'..\models/xgboost.pkl')


def Obesity_classification():

    ## Title
    st.title('Obesity Classification Prediction ....')
    st.markdown('<hr>', unsafe_allow_html=True)

    ## Input fields

    family_history_with_overweight = st.selectbox('Family history with overweight', options=['yes', 'no'])
    favc = st.selectbox('Frequent consumption of high caloric food', options=['yes', 'no'])
    fcvc = st.number_input('Usually eat vegetables in your meals', step=0.01)
    ncp = st.number_input('Number of main meals', step=0.01)
    
    ch2o = st.number_input('Consumption of water daily', step=0.01)
    caec = st.selectbox('Consumption of food between meals', options=['Sometimes', 'Frequently','Always','no'])
    scc = st.selectbox('Calories consumption monitoring', options=['yes','no'])
    calc = st.selectbox('Consumption of alcohol', options=['Sometimes', 'Frequently','no'])
    faf = st.number_input('Physical activity frequency', step=0.01)
    tue= st.number_input('Time using technology devices', step=0.01)
    mtrans=st.selectbox('Consumption of alcohol', options=['Public_Transportation', 'Automobile','Walking','Motorbike','Bike'])
    height=st.number_input('Height', step=0.01)
    weight=st.number_input('Weight', step=0.01)
    
    st.markdown('<hr>', unsafe_allow_html=True)


    if st.button('Predict Obesity ...'):

        ## Concatenate the users data
        new_data = np.array([family_history_with_overweight, favc, fcvc, ncp, caec, ch2o, scc, faf, tue, calc, mtrans, height, weight])
        
        ## Call the function from utils.py to apply the pipeline
        X_processed = process_now(x_new=new_data)

        ## Predict using Model

        y_pred = model.predict(X_processed)
        
        ## Mapping of target labels
        map_target = {
            0: 'Insufficient Weight',
            1: 'Normal_Weight',
            2: 'Overweight_Level_I',
            3: 'Overweight_Level_II',
            4: 'Obesity_Type_I',
            5: 'Obesity_Type_II',
            6: 'Obesity_Type_III'
        }
        y_pred_mapped = [map_target[val] for val in y_pred]

        ## Display Results
        st.success(f'Obesity Prediction is ... {y_pred_mapped}')



if __name__ == '__main__':
    ## Call the function
    Obesity_classification()

