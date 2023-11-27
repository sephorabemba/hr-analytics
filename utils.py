import os
from enum import Enum

# file names
root = os.getcwd()
data_folder = "data"
models_folder = "models"
data_out_folder = "data_out"
dataset_filename = "HR_capstone_dataset.csv"
train_filename = "train_full.csv"
valid_filename = "validation_full.csv"
test_filename = "test_full.csv"
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

# mappings

dept_map = {
    "management": "Management",
    "support": "Support",
    "technical": "Technical",
    "sales": "Sales",
    "product_mng": "Product Management",
    "marketing": "Marketing",
    "hr": "HR",
    "IT": "IT",
    "accounting": "Accounting",
    "RandD": "Other",

}


class Department(Enum):
    def __init__(self, id, raw, formatted):
        self.id = id
        self.raw = raw
        self.formatted = formatted

    MANAGEMENT = 0, "management", "Management"
    SUPPORT = 1, "support", "Support"
    TECHNICAL = 2, "technical", "Technical"
    SALES = 3, "sales", "Sales"
    PRODUCT_MANAGEMENT = 4, "product_mng", "Product Management"
    MARKETING = 5, "marketing", "Marketing"
    HR = 6, "hr", "HR"
    IT = 7, "IT", "IT"
    ACCOUNTING = 8, "accounting", "Accounting"
    RANDD = 9, "RandD", "Other"


Department._col_name = "department_fmt"


class Tenure(Enum):
    def __init__(self, id, formatted):
        self.id = id, id
        self.formatted = formatted

    BETWEEN_0_AND_2Y = 0, "Between 0 and 2 years"
    BETWEEN_3_AND_5Y = 1, "Between 3 and 5 years"
    FROM_6Y_AND_MORE = 2, "6 years and more"

    @staticmethod
    def format_tenure(tenure):
        if tenure <= 2:
            return "Between 0 and 2 years"
        if tenure <= 5:
            return "Between 3 and 5 years"
        return "6 years and more"


Tenure._col_name = "tenure_fmt"


class Perf(Enum):
    def __init__(self, id, formatted):
        self.id = id, id
        self.formatted = formatted

    LOW = 0, "Low Performance"
    MIDDLE = 1, "Middle Performance"
    HIGH = 2, "High Performance"
    VERY_HIGH = 3, "Very High Performance"

    @staticmethod
    def format_perf(perf):
        if perf <= 0.5:
            return Perf.LOW.formatted
        if perf <= 0.7:
            return Perf.MIDDLE.formatted
        if perf <= 0.9:
            return Perf.HIGH.formatted
        return Perf.VERY_HIGH.formatted


Perf._col_name = "perf_fmt"


class Salary(Enum):
    def __init__(self, id, formatted):
        self.id = id, id
        self.formatted = formatted

    LOW = 0, "Low Salary"
    MEDIUM = 1, "Medium Salary"
    HIGH = 2, "High Salary"

    @staticmethod
    def format_salary(salary):
        return f"{salary.capitalize()} Salary"


Salary._col_name = "salary_fmt"


class Factor(Enum):
    def __init__(self, id, raw, formatted):
        self.id = id
        self.raw = raw
        self.formatted = formatted

    SATISFACTION_LEVEL = 0, "satisfaction_level", "Satisfaction Level"
    NUMBER_PROJECT = 1, "number_project", "Number of Projects"
    TENURE = 2, "tenure", "Tenure"
    AMH_X_LAST_EVAL = 3, "amh_x_last_eval", "Achiever Type (Nb hours worked x Performance)"
    AVERAGE_MONTHLY_HOURS = 4, "average_monthly_hours", "Monthly Hours Worked",
    LAST_EVALUATION = 5, "last_evaluation", "Performance"
    WORK_ACCIDENT = 6, "work_accident", "Work Accident"
    PROMOTION_LAST_5_YEARS = 7, "promotion_last_5years", "Promoted"

    @staticmethod
    def format_factor(factor):
        return Factor._factors[factor]


Factor._factors = {
    "satisfaction_level": Factor.SATISFACTION_LEVEL.formatted,
    "number_project": Factor.NUMBER_PROJECT.formatted,
    "tenure": Factor.TENURE.formatted,
    "amh_x_last_eval": Factor.AMH_X_LAST_EVAL.formatted,
    "average_monthly_hours": Factor.AVERAGE_MONTHLY_HOURS.formatted,
    "last_evaluation": Factor.LAST_EVALUATION.formatted,
    "work_accident": Factor.WORK_ACCIDENT.formatted,
    "promotion_last_5years": Factor.PROMOTION_LAST_5_YEARS.formatted,
}

segments = {
    # "Department": dept_map,
    "Department": Department,
    "Tenure": Tenure,
    "Performance": Perf,
    "Salary": Salary,
}
