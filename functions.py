from deepchecks.tabular import Dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from deepchecks.tabular.suites import data_integrity
from deepchecks.tabular.suites import train_test_validation
from deepchecks.tabular.suites import model_evaluation
from joblib import dump, load
import pandas as pd
import os
import re

# Load sklearn dataframe
def load_Empdata_df(filename):
    dataset_dir = 'Dataset'
    # Define the path to the "empfile.csv" file within the "dataset" directory
    file_path = os.path.join(dataset_dir, filename)
    df = pd.read_csv(file_path)
    return df


# Split Dataframe into train and test dataframe with columns id removed 
def split_dataframe():
    df = load_Empdata_df(filename='Train-employee-salary.csv')
    # Mapping categorical values to numeric values
    blood_type_mapping = {'A': 0, 'B': 1, 'AB': 2, 'O': 3}
    df['groups'] = df['groups'].map(blood_type_mapping)
    X = df.drop(columns=['id','salary'], axis=1)  # removed ID and Salary columns
    y = df['salary']
    # Split the Test data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create two DataFrames from the split data
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    return df_train, df_test

#Converting Full Dataframe to full Dataset for Data integrity check used in Deepchecks library
def load_dataset():
    df_train, df_test = split_dataframe()  
    # Mapping categorical values to numeric values

    # Create Dataset objects from dataframe for deepcheck usage
    train_ds = Dataset(df_train.iloc[:,:-1], label=df_train['salary'], cat_features=['groups','healthy_eating','active_lifestyle'])
    test_ds = Dataset(df_test.iloc[:,:-1], label=df_test['salary'], cat_features=['groups','healthy_eating','active_lifestyle'])
    return train_ds, test_ds


# Training Linear Model
def train_linear_model(filename='trained_SalaryPrediction_linear_model'):
    try:
        df_train, df_test = split_dataframe()
        # Check data types and valid test_frac value
        assert isinstance(df_train, pd.DataFrame), "X_train must be a DataFrame"
        assert isinstance(df_test, pd.DataFrame), "X_train must be a DataFrame"
        assert isinstance(df_train['salary'], pd.Series), "y_train must be a Series"
        assert isinstance(df_test['salary'], pd.Series), "y_train must be a Series"
        assert isinstance(filename, str), "Filename must be a string"

        # Spliting dataframe into X and y
        X_train = df_train.drop('salary',axis=1)
        X_test = df_test.drop('salary',axis=1)
        y_train = df_train['salary']
        y_test = df_test['salary']

        # Linear Regression Model creation
        model = LinearRegression()

        # Fit the model with training data
        model.fit(X_train, y_train)  # Assuming 'target' is the column to predict

        # Save the trained model to a file
        fname = filename + '.joblib'
        dump(model, fname)
        return model
   
    except AssertionError as msg:
        print(msg)
        return msg
    

# Assuming 'model' is your trained regression model

def evaluate_salary_prediction(input_features, actual_salary):
    try:
        model = train_linear_model()
        # Prepare input features for prediction
        input_data = [input_features]  # Assuming input_features is already a list

        # Make prediction
        predicted_salary = model.predict(input_data)[0]

        # Calculate accuracy
        accuracy = (1 - abs(predicted_salary - actual_salary) / actual_salary) * 100
        return predicted_salary, accuracy
    except AssertionError as msg:
        print(msg)
        return msg

# Function Overloading , f the test file doesn't have the salary values, and  we need predict the salary for each row using your model
def salary_prediction(input_features):
    try:
        model = train_linear_model()
        # Prepare input features for prediction
        input_data = [input_features]  # Assuming input_features is already a list
        # Make prediction
        predicted_salary = model.predict(input_data)[0]        
        return predicted_salary
    except AssertionError as msg:
        print(msg)
        return msg


# Function to calculate and return R-squared scores
def Rsquared_linear_model():
    try:
        df_train, df_test = split_dataframe()
        model = train_linear_model()
        
        # Splitting dataframe into X and y
        X_train = df_train.drop('salary', axis=1)
        X_test = df_test.drop('salary', axis=1)
        y_train = df_train['salary']
        y_test = df_test['salary']

        # Compute R-squared scores for training and test data
        r2_train = model.score(X_train, y_train)
        r2_test = model.score(X_test, y_test)  # Assuming 'Salary' is the column to predict

        print("Train R-squared:", r2_train)
        print("Test R-squared:", r2_test)

        # Return scores in a dictionary
        score = {'Train-score': r2_train, 'Test-score': r2_test}
        return score
    except AssertionError as msg:
        print(msg)
        return msg

# Deepcheck Data integrity check and saving the report into HTML
def data_integrity_check():
    train_ds,test_ds = load_dataset()
    # Run Suite:
    integ_suite = data_integrity()
    suite_result1 = integ_suite.run(train_ds)
    suite_result2 = integ_suite.run(test_ds)

    # Save the result report as an HTML file
    suite_result1.save_as_html("Train_data_integrity_report.html")
    suite_result2.save_as_html("Test_data_integrity_report.html")
 

# Deepcheck to verify the train and test dataset - here we are using Dataframe however Deepchekc works well with Dataset for train_test_validation
def train_test_dataset_validation():
    validation_suite = train_test_validation()
    train_ds,test_ds = load_dataset()
    suite_result = validation_suite.run(train_ds, test_ds)
    # Note: the result can be saved as html using suite_result.save_as_html()
    # or exported to json using suite_result.to_json()
    suite_result.save_as_html("Train_Test_dataset_Validation_Suite.html")


def model_evaluation_suite():
    train_ds, test_ds = load_dataset()
    # Load the saved model from the joblib file
    lg = load("trained_SalaryPrediction_linear_model.joblib")

    evaluation_suite = model_evaluation()
    suite_result = evaluation_suite.run(train_ds, test_ds, lg )
    # Note: the result can be saved as html using suite_result.save_as_html()
    # or exported to json using suite_result.to_json()
    # or to show the result suite_result.show()
    suite_result.save_as_html("model_evaluation_suite.html")