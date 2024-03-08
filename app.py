import streamlit as st
import pickle
import pandas as pd

# Set page title and favicon
st.set_page_config(page_title="Bank Customer Churn Prediction", page_icon=":bank:")

def main():
    st.title("Bank Customer Churn Prediction")

    # Load encoder and ML model
    encoder = pickle.load(open('onehotencoder.pkl','rb'))
    ml_model = pickle.load(open('xgboostclassifier.pkl','rb'))

    st.write("This machine learning model is trained on synthetic data from the Playground competitions.Using synthetic data allows for a balance between real-world scenarios and ensuring test labels are not publicly available.")
    st.write("While synthetic data generation has challenges, efforts have been made to minimize artifacts in the dataset.")
    st.write("Please enter the following information about the customer:")

    # Input fields for user to enter data
    credit_score = st.number_input('1.Credit Score', min_value=0, help="The credit score of the customer.")
    geography = st.selectbox('2.Home Branch Country', ['France', 'Spain', 'Germany'], help="The country where the customer resides.")
    gender = st.radio('3.Gender', ['Male', 'Female'], help="The gender of the customer.")
    age = st.number_input('4.Age', min_value=18, max_value=100, help="The age of the customer.")
    tenure = st.number_input('5.Tenure', min_value=0, help="The number of years the customer has been with the bank.")
    balance = st.number_input('6.Account Balance', min_value=0.0, help="The account balance of the customer.")
    num_of_products = st.number_input('7.Number of Services', min_value=1, help="The number of services opted by the customer.")
    has_credit_card = st.checkbox('8.Credit Card Status', help="Whether the customer has a credit card or not.")
    is_active_member = st.checkbox('9.Membership status', help="Whether the customer is an active member or not.")
    estimated_salary = st.number_input('10.Estimated Salary', min_value=0.0, help="The estimated salary of the customer.")

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

    # Encoding the categorical values
    encoded_df = encoder.transform(df[['Geography','Gender']])
    final_encoded_df = pd.DataFrame(encoded_df.toarray(),columns=encoder.get_feature_names_out(['Geography','Gender']))
    final_encoded_df.index = df.index

    selected_columns_df = df[['Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
    final_encoded_df = pd.concat([selected_columns_df,final_encoded_df], axis=1)

    # Adding a button to trigger model prediction
    if st.button('Get Churn Probability'):

        # Model prediction
        prediction = ml_model.predict_proba(final_encoded_df)

        # Extract the prediction probability for the first class
        predict_probability = prediction[0][1] * 100  # getting the probability of the positive class or churn customers class

        # Display the prediction probability
        st.success(f"Churn Probability: {predict_probability:.2f}%")

if __name__=='__main__':
    main()
