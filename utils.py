import os

# file names
root = os.getcwd()
data_folder = "data"
models_folder = "models"
data_out_folder = "data_out"
dataset_filename = "HR_capstone_dataset.csv"
train_filename = "train.csv"
valid_filename = "validation.csv"
test_filename = "test.csv"
best_model_filename = "Gradient Boosting - With amh_x_last_eval.joblib"
rf_filename = "random_forest.joblib"
brf_filename = "balanced_random_forest.joblib"
gb_filename = "gradient_boosting.joblib"
adaboost_filename = "adaboost.joblib"
lr_filename = "logistic_regression.joblib"

# dataset file paths
train_filepath = f"{root}/{data_folder}/{train_filename}"
valid_filepath = f"{root}/{data_folder}/{valid_filename}"
test_filepath = f"{root}/{data_folder}/{test_filename}"

# models
model_filepath = f"{root}/{models_folder}/{gb_filename}"
