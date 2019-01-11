# Bosch Production Line Performance Internal Failure Predictor
 In this project, I will predict internal failures of Bosch using thousands of measurements and tests made for each component along the assembly line.

 A good chocolate souffle is decadent, delicious, and delicate. But, it's a challenge to prepare. When you pull a disappointingly deflated dessert out of the oven, you instinctively retrace  your steps to identify at what point you went wrong. Bosch, one of the world's leading manufacturing companies, has an imperative to ensure that the recipes for the production of its advanced mechanical components are of the highest quality and safety standards. Part of doing so is closely monitoring its parts as they progress through the manufacturing processes.

 Because Bosch records data at every step along its assembly lines, they have the ability to apply advanced analytics to improve these manufacturing processes. However, the intricacies of the data and complexities of the production line pose problems for current methods.

# Objective: Prediction of Internal Failures to reduce Manufacturing Failure

- This project will cover the following skills: 
  - Understanding about Manufacturing domain and its failures
  - Working with large real-time dataset
  - Feature Engineering
  - Working on Extratree Classifier
  - Working on Ensembles methods
  - New evaluation metrics
  - Python
  - Univariate Characteristics
  - Naive Bayes KDE: Decision Tree Classifer
  - Extra Tree classifer 
  - Random Forest
  - Grid Search CV
  - XGBoost
  
# Dataset Introduction

 The data for this competition represents measurements of parts as they move through Bosch's production lines. Each part has a unique Id. The goal is to predict which parts will fail quality control (represented by a 'Response' = 1).

 The dataset contains an extremely large number of anonymized features. Features are named according to a convention that tells you the production line, the station on the line, and a feature number. E.g. L3_S36_F3939 is a feature measured on line 3, station 36, and is feature number 3939.

 On account of the large size of the dataset, we have separated the files by the type of feature they contain: numerical, categorical, and finally, a file with date features. The date features provide a timestamp for when each measurement was taken. Each date column ends in a number that corresponds to the previous feature number. E.g. the value of L0_S0_D1 is the time at which L0_S0_F0 was taken.

 In addition to being one of the largest datasets (in terms of number of features) ever hosted on Kaggle, the ground truth for this competition is highly imbalanced. Together, these two attributes are expected to make this a challenging problem.

 File descriptions train_numeric.csv - the training set numeric features (this file contains the 'Response' variable) test_numeric.csv - the test set numeric features (you must predict the 'Response' for these Ids) train_categorical.csv - the training set categorical features test_categorical.csv - the test set categorical features train_date.csv - the training set date features test_date.csv - the test set date features sample_submission.csv - a sample submission file in the correct format

[Dominic Pagan](https://www.ingenium-ai.com/) is the Lead [Machine Learning](https://www.ingenium-ai.com/) and [Artificial Intelligence](https://www.ingenium-ai.com/) [Developer](https://www.ingenium-ai.com/) at [Ingenium A.I.](https://www.ingenium-ai.com/) LLC. 
