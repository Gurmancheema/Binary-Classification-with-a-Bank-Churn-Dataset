import streamlit as st
import pickle
import pandas as pd

def main():

    encoder = pickle.load(open('onehotencoder.pkl','rb'))
    ml_model = pickle.load(open('xgboostclassifier.pkl','rb'))
    # st.write("Encoder & Model loaded successfully")


    # Creating input fields for user to enter data

    credit_score = st.number_input('Credit Score', min_value=0)
    geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
    gender = st.radio('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=18, max_value=100)
    tenure = st.number_input('Tenure', min_value=0)
    balance = st.number_input('Balance', min_value=0.0)
    num_of_products = st.number_input('Number of Products', min_value=1)
    has_credit_card = st.checkbox('Has Credit Card')
    is_active_member = st.checkbox('Is Active Member')
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

    # Convert boolean values to binary (0 or 1)
    has_credit_card = 1 if has_credit_card else 0
    is_active_member = 1 if is_active_member else 0

    # Store the input values in a dictionary
    user_data = {
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_credit_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(user_data)

    # Display the DataFrame
    st.write('User DataFrame:', df)

    # Encoding the categorical values
    encoded_df = encoder.transform(df[['Geography','Gender']])

    final_encoded_df = pd.DataFrame(encoded_df.toarray(),columns=encoder.get_feature_names_out(['Geography','Gender']))
    final_encoded_df.index = df.index

    selected_columns_df = df[['Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
    final_encoded_df = pd.concat([selected_columns_df,final_encoded_df], axis=1)

    st.write('Encoded DataFrame:', final_encoded_df)

    # Adding a button to trigger model prediction
    if st.button('Get Churn Probability'):

        # Model prediction
        prediction = ml_model.predict_proba(final_encoded_df)

        # Extract the prediction probability for the first class
        predict_probability = prediction[0][1] * 100  # getting the probability of the positive class or churn customers class

        # Display the prediction probability
        st.write(f"Churn Probability: {predict_probability:.2f}%")


if __name__=='__main__':
    main()