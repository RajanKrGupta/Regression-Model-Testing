import pytest
from functions import load_Empdata_df, split_dataframe, train_linear_model, data_integrity_check

# Test load_Empdata_df function
def test_load_Empdata_df():
    df = load_Empdata_df('Train-employee-salary.csv')
    assert df is not None
    assert 'salary' in df
    assert len(df) > 0
    # Add more specific assertions as needed

# Test split_dataframe function
def test_split_dataframe():
    X_train, X_test, y_train, y_test = split_dataframe()
    assert X_train is not None
    assert y_train is not None
    assert X_test is not None
    assert y_test is not None
    assert len(X_train) > 0
    assert len(y_train) > 0
    assert len(X_test) > 0
    assert len(y_test) > 0
    # Add more specific assertions as needed

# Test train_linear_model function
def test_train_linear_model():
    X_train, X_test, y_train, y_test = split_dataframe()
    result = train_linear_model()
    assert 'Train-score' in result
    assert 'Test-score' in result
    assert 0 <= result['Train-score'] <= 1
    assert 0 <= result['Test-score'] <= 1
    print(result['Train-score'])
    print(result['Test-score'])
    # Add more specific assertions as needed

# Test data_integrity_check function
def test_data_integrity_check():
    data_integrity_check()
    # You can add assertions for the result of data integrity check, e.g., check for specific issues

# Additional custom test cases

# Test if the saved model file exists
def test_saved_model_file_exists():
    import os
    assert os.path.isfile('trained_SalaryPrediction_linear_model.joblib')

# Test if the HTML report for data integrity check exists
def test_data_integrity_report_exists():
    import os
    assert os.path.isfile('data_integrity_report.html')

# Test if data integrity check reports any issues
def test_data_integrity_report_issues():
    # Implement assertions to check for specific issues reported in the data integrity report
    pass

# You can add more custom test cases as needed
