import pytest
import os
import re
from functions import load_Empdata_df, model_evaluation_suite, split_dataframe, train_linear_model, data_integrity_check, train_test_dataset_validation

# Test presenec of Training Empdata_df function
def test_load_Train_Empdata_df():
    df = load_Empdata_df('Train-employee-salary.csv')
    assert df is not None
    assert 'salary' in df
    assert len(df) > 0
    # Add more specific assertions as needed

# Test presenec of Test Empdata_df function
def test_load_Train_Empdata_df():
    df = load_Empdata_df('Test-employee-salary.csv')
    assert df is not None
    #assert 'salary' not in df
    assert len(df) > 0


# Test split_dataframe function
def test_split_dataframe():
    train_df, test_df = split_dataframe()
    assert train_df is not None
    assert test_df is not None
    assert len(train_df) > 0
    assert len(test_df) > 0
    # Add more specific assertions as needed

# Test train_linear_model function
def test_train_linear_model():
    test_df, test_df = split_dataframe()
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
    assert os.path.isfile('trained_SalaryPrediction_linear_model.joblib')

# Test if the HTML report for data integrity check exists
# Test if the HTML report for data integrity check exists
def test_data_integrity_report_exists():
    import os
    assert os.path.isfile('Train_data_integrity_report.html')

# To validate two data subsets
# Comparing distributions across different train-test splits (e.g. before training a model or when splitting data for cross-validation)
# Comparing a new data batch to previous data batche
def test_train_test_dataset_validation():
    # Implement assertions to check for specific issues reported in the data integrity report
    train_test_dataset_validation()


def test_model_evaluation_suite():
    # Implement assertions to check for specific issues reported in the data integrity report
    model_evaluation_suite()

# You can add more custom test cases as needed
