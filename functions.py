from deepchecks.tabular import Dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from deepchecks.tabular.suites import data_integrity
from joblib import dump
import pandas as pd
import os

# Load sklearn dataframe
def load_Empdata_df(filename):
    dataset_dir = 'dataset'
    # Define the path to the "empfile.csv" file within the "dataset" directory
    file_path = os.path.join(dataset_dir, filename)
    df = pd.read_csv(file_path)
    return df


# Split Dataframe
def split_dataframe():
    df = load_Empdata_df('Train-employee-salary.csv')
    # Mapping categorical values to numeric values
    blood_type_mapping = {'A': 0, 'B': 1, 'AB': 2, 'O': 3}
    df['groups'] = df['groups'].map(blood_type_mapping)
    X = df.drop(columns='salary', axis=1)
    y = df['salary']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#Converting Dataframe to Dataset for Deepcheck uses
def load_dataset():
    df = load_Empdata_df('Train-employee-salary.csv')
    # Create Dataset objects for the diabetes dataframe
    ds = Dataset(df, label='salary', cat_features=['groups','healthy_eating','active_lifestyle'])
    return ds

# Training Linear Model
def train_linear_model(filename='trained_SalaryPrediction_linear_model'):
    try:
        X_train, X_test, y_train, y_test = split_dataframe()
        # Check data types and valid test_frac value
        assert isinstance(X_train, pd.DataFrame), "X_train must be a DataFrame"
        assert isinstance(y_train, pd.Series), "y_train must be a Series"
        assert isinstance(X_test, pd.DataFrame), "X_train must be a DataFrame"
        assert isinstance(y_test, pd.Series), "y_train must be a Series"
        assert isinstance(filename, str), "Filename must be a string"

        # Instantiate a Linear Regression model
        model = LinearRegression()

        # Fit the model with training data
        model.fit(X_train, y_train)  # Assuming 'target' is the column to predict

        # Save the trained model to a file
        fname = filename + '.joblib'
        dump(model, fname)

        # Compute R-squared scores for training and test data
        r2_train = model.score(X_train, y_train)
        r2_test = model.score(X_test, y_test)  # Assuming 'target' is the column to predict

        print("Train R-squared:", r2_train)
        print("Test R-squared:", r2_test)

        # Return scores in a dictionary
        return {'Train-score': r2_train, 'Test-score': r2_test}

    except AssertionError as msg:
        print(msg)
        return msg

# Deepcheck Data integrity check and saving the report into HTML
def data_integrity_check():
    ds = load_dataset()

    # Run Suite:
    integ_suite = data_integrity()
    suite_result = integ_suite.run(ds)

    # Save the result report as an HTML file
    suite_result.save_as_html("data_integrity_report.html")
