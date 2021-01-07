# Salary Prediction based on Job descriptions
## Problem Overview
The goal of this project is to predict the missing salaries of the unseen job postings by analyzing a dataset about job postings and their corresponding salaries.

The possible use-case for this project is for the websites showing job postings to give a better estimate of the salaries and to give job seekers an idea about the salaries based on their credentials.

## Checking out the data
Following datasets are provided for this project in the CSV format:
- `train_features.csv`: As name suggests, this dataset contains the training data in tabular form with all the features. It contains 1 million records and each one of them corresponds to a job listing. All the records contain 8 features which are described as follows:
    
    * `jobId` (str): Unique identifier for each job listing
    
    The following features are the categorical variables in the dataset:
    * `companyId` (str): Categorical variable representing a unique identifier for each company. Total of 63 different companies have their job listings in the dataset
    * `jobType` (str): Examples: CEO, CFO, Manager, etc.
    * `degree` (str): Examples: Bachelors, Masters, Doctoral, etc.
    * `major` (str): Examples: Biology, Engineering, Math, etc.
    * `industry` (str): Examples: Web, Finance, Health, etc.

    There are two numeric features in the dataset:
    * `yearsExperience` (int): Number of years of experience
    * `milesFromMetropolis` (int): Distance between a job's location and nearest metropolis

- `train_salaries.csv`: Dataset containing the target variable (`salary`) for each job listing of the `train_features.csv`.