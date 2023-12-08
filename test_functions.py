import pytest
import os
import re
import functions
#from functions import Rsquared_linear_model, evaluate_salary_prediction, load_Empdata_df, model_evaluation_suite, split_dataframe, train_linear_model, data_integrity_check, train_test_dataset_validation

# Test presence of Training Empdata_df function
def test_load_Train_Empdata_df():
    df = functions.load_Empdata_df('Train-employee-salary.csv')
    assert df is not None
    assert 'salary' in df
    assert len(df) > 0
    # Add more specific assertions as needed

# Test presence of Test Empdata_df function
def test_load_Train_Empdata_df():
    df = functions.load_Empdata_df('Test-employee-salary.csv')
    assert df is not None
    assert 'salary' in df.columns
    assert len(df) > 0


# Test split_dataframe function
def test_split_dataframe():
    train_df, test_df = functions.split_dataframe()
    assert train_df is not None
    assert test_df is not None
    assert len(train_df) > 0
    assert len(test_df) > 0
    # Add more specific assertions as needed

# Test train_linear_model function for R-squared scores for training and test data
@pytest.mark.prediction
def test_Rsquared_scores_train_test():
    result = functions.Rsquared_linear_model()
    assert 'Train-score' in result
    assert 'Test-score' in result
    assert 0 <= result['Train-score'] <= 1
    assert 0 <= result['Test-score'] <= 1
    print(result['Train-score'])
    print(result['Test-score'])
    # Add more specific assertions as needed

# Test data_integrity_check function
@pytest.mark.skip(reason="Working Time Consuming")
def test_data_integrity_check():
    functions.data_integrity_check()
    # You can add assertions for the result of data integrity check, e.g., check for specific issues

# Additional custom test cases

# Test if the saved model file exists
def test_saved_model_file_exists():
    assert os.path.isfile('trained_SalaryPrediction_linear_model.joblib')

# Test if the HTML report for data integrity check exists
# Test if the HTML report for data integrity check exists
@pytest.mark.skip(reason="Working Time Consuming")
@pytest.mark.report
def test_data_integrity_report_exists():
    import os
    assert os.path.isfile('Train_data_integrity_report.html')

# To validate two data subsets
# Comparing distributions across different train-test splits (e.g. before training a model or when splitting data for cross-validation)
# Comparing a new data batch to previous data batche

@pytest.mark.skip(reason="Working Time Consuming")
def test_train_test_dataset_validation():
    # Implement assertions to check for specific issues reported in the data integrity report
    functions.train_test_dataset_validation()

@pytest.mark.skip(reason="Working Time Consuming")
def test_model_evaluation_suite():
    # Implement assertions to check for specific issues reported in the data integrity report
    functions.model_evaluation_suite()

# You can add more custom test cases as needed

@pytest.mark.prediction
def test_positive_salary_prediction(input_features=[0, 36, 5, 5], actual_salary = 2297 ):
    predicted_salary, accuracy = functions.evaluate_salary_prediction(input_features, actual_salary)
    print(f"Actual Salary: {actual_salary}")
    print(f"Predicted Salary: {predicted_salary}")
    print(f"Accuracy: {accuracy:.2f}%")
    assert predicted_salary > 2000
    assert 0< accuracy <100


@pytest.mark.testdf_prediction
def test_checking_predicted_salary_from_actual():
    # Load input data from CSV
    df_train, df_test = functions.split_dataframe()

    # Iterate over rows in the CSV file
    for index, row in df_test.iterrows():
        input_features = [row['groups'], row['age'], row['healthy_eating'], row['active_lifestyle']]
        actual_salary = row['salary']

        # Perform salary prediction for each row
        predicted_salary, accuracy = functions.evaluate_salary_prediction(input_features, actual_salary)

        # Add assertions based on your requirements
        # assert accuracy > 0  # Add your specific assertions here

        # Optionally, print results for each row
        print(f"Row {index + 1} - Predicted Salary: {predicted_salary}, Actual Salary: {actual_salary}, Accuracy: {accuracy}%\n")


@pytest.mark.new
def test_salary_prediction_from_csv():
    # Load input data from CSV
    input_data = functions.load_Empdata_df('Test-employee-salary.csv')
    input_data = input_data.drop(columns='id', axis=1)
    
     # Mapping categorical values to numeric values
    blood_type_mapping = {'A': 0, 'B': 1, 'AB': 2, 'O': 3}
    input_data['groups'] = input_data['groups'].map(blood_type_mapping)

    # Iterate over rows in the CSV file
    for index, row in input_data.iterrows():
        input_features = [row['groups'], row['age'], row['healthy_eating'], row['active_lifestyle']]

        # Perform salary prediction for each row
        predicted_salary = functions.salary_prediction(input_features)

        # Add assertions based on your requirements
        assert predicted_salary is not None  # Add your specific assertions here

        # Optionally, print results for each row
        print(f"Row {index + 1} - Predicted Salary: {predicted_salary}")

# ... other test cases ...
