# HR Analytics: Employee Turnover Prediction

Analysis and prediction of employee turnover on the "Hr Analytics Job Prediction" Kaggle dataset.

## Project Overview

The goal of this project is to predict employee turnover on the "Hr Analytics Job Prediction" [Kaggle dataset](
https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction) using
employee survey results and employee characteristics such as tenure, performance, number of projects, etc.

To predict the turnover effectively, we built a Gradient Boosting model that achieved an average precision of 0.95 on
unseen data.

## Business Understanding

Retaining talents in a company is important to build steady businesses. Many companies struggle with preventing
turnover.
The main reasons are a lack of understanding the underlying factors and not addressing the core problems.

High turnover situations must be solved as they cause the following problems to companies, among others:

- Disruption for the teams and negative impact on group morale.
- High company cost, one to two times an employee’s annual salary per leaver (
  source: [Josh Bersin - Deloitte](https://www.linkedin.com/pulse/20130816200159-131079-employee-retention-now-a-big-issue-why-the-tide-has-turned/)).
- Lost productivity.

## Data Understanding

The dataset has 9 columns and 1 categorical target `left`.

| Column name           | Type  | Description                                                       |
|-----------------------|-------|-------------------------------------------------------------------|
| satisfaction_level    | int64 | The employee’s self-reported satisfaction level [0-1]             |
| last_evaluation       | int64 | Score of employee's last performance review [0–1]                 |
| number_project        | int64 | Number of projects employee contributes to                        |
| average_monthly_hours | int64 | Average number of hours employee worked per month                 |
| time_spend_company    | int64 | How long the employee has been with the company (years)           |
| work_accident         | int64 | Whether or not the employee experienced an accident while at work |
| left                  | int64 | Whether or not the employee left the company                      |
| promotion_last_5years | int64 | Whether or not the employee was promoted in the last 5 years      |
| department            | str   | The employee's department                                         |
| salary                | str   | The employee's salary (low, medium, or high)                      |

The dataset is imbalanced with a turnover of 16.6%.

[Pie chart of the target variable]("img/es-left-pie.png")

## Modeling and Evaluation

### Evaluation Metrics

We picked the `average precision` as our main evaluation metric for the following reasons:

- The business problem requires to spot as many risky employees as possible to be able to retain them (cf. high recall).
- It also requires to understand perfectly why the rest of the employees want to stay to put in place a lean action
  plan. Therefore, it's also important to minimize false positives  (cf. high precision).
- Average precision gives stable insights on how a model performs balancing minimizing false positive and negatives at
  different decision thresholds. It's especially useful for imbalanced datasets like the one at hand.

### Cross Validation and Model Selection

* We performed a Grid Search cross validation to evaluate different models with different hyperparameter sets, including
  ensemble, tree-based and regression models.
* The Gradient Boosting algorithm was selected as the best algorithm with a best model achieving an average precision of
  0.95.

## Conclusion

Employee satisfaction, work load, tenure and performance evaluation strongly impact employee turnover.

[Feature importance](img/es-importance.png)

The leadership team should put in place actions to tackle these areas. Here are a few data-driven recommendations:
- Implement regular employee feedback sessions to address concerns and improve job satisfaction.
- Provide professional development opportunities and clear career paths.
- Introduce flexible work arrangements for better work-life balance.

We've created a Streamlit App to:
* Test the model.
* Summarize the story of this dataset.
* Provide customized recommendations to decrease employee turnover.

**Try it out below!**

[Streamlit demo](img/es-streamlit.png)

## Run the Streamlit Demo

1. Open the demo on Streamlit Cloud...

2. ...Or clone and run locally

```
git clone https://github.com/sephorabemba/hr-analytics.git

streamlit run streamlit-app/Employee_Turnover_Prediction_App.py
```
