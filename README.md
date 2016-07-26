# General-Social-Survey

Team project using General Social Survey (GSS) data to predict happiness.

# Summary

We set out to build a model to predict whether a person is unhappy, based on demographic and biographic information. The intended application of our model is to identify people who may be at higher risk for depression, for instance among a doctor's patient population, so that they can be proactively screened (through an interview).

We used data from the GSS, a survey conducted every 1-2 years since 1972 to gather comprehensive data on attitudes and demographic information among the American public.

# Data Cleaning and Assumptions

Over the lifetime of the GSS, over 5000 distinct questions or attributes have been asked. We whittled this down to 28 variables for initial consideration, and mapped the response codes to appropriate categorical values or intuitive numerical values for responses on a scale.

Unfortunately, we had to drop the Income variable, because the GSS asked for income in buckets rather than as a specific number, so we are unable to inflation-adjust the values over time.

Additional cleaning steps included filling N/A values with the median (to represent the "average person") for numerical, or mode for categorical variables. Before modeling (but after performing exploratory data analysis), we also converted Happiness from a 3-category variable to a binary variable, grouping "Very Happy" and "Pretty Happy" together as "Happy", to make a clear distinction from people who identify as "Unhappy".

# Select Exploratory Data Visualizations

Below is shown average happiness plotted against education level, demonstrating that more highly-educated individuals tend to be happier, particularly among women. The size of the bubbles corresponds to the number of samples, with large groups at education levels 12 (completed high school) and 16 (completed 4-year college). While we're not focused on causation in this project, it is worth noting that more highly educated people are also more likely to have higher income, and it may be the higher income that leads to increased happiness rather than the education itself.


<img src=https://github.com/cmgerr/General-Social-Survey/raw/master/Images/Happiness_by_Education_Level.png width="600">


Interestingly, despite the rise in average education level over time, average happiness has not increased:


<img src=https://github.com/cmgerr/General-Social-Survey/raw/master/Images/Average_Education_and_Happiness_over_Time.png width="600">


Marital status, or more precisely, whether or not a person is married, also appears to make a difference in average happiness:


<img src=https://github.com/cmgerr/General-Social-Survey/raw/master/Images/Happiness_by_Marital_Status.png width="600">


# Approaches to dealing with Highly Dimensional Data

## Eliminating highly correlated variables

To reduce our feature space, we examined pairs of variables with correlations over 0.4, with those correlations shown below:

<img src=https://github.com/cmgerr/General-Social-Survey/raw/master/Images/Correlations.png width="300">

Using these correlations, along with common sense judgment, we dropped education variables aside from the survey respondent, as well as home population and household earners variables.

## Principal Component Analysis

We also attempted PCA as a technique for feature reduction in modeling. However, we found that there was not a major concentration of explained variance among the top principal components. This is not very surprising, because PCA is generally not as effective with discrete and categorical features, which is all our dataset contains.

The below plots show the Happiness variable plotted against the top two principal components:

<img src=https://github.com/cmgerr/General-Social-Survey/raw/master/Images/PCA.png width="400">

# Modeling

For modeling purposes, we used data from 2006 onward. This is because we wanted to identify current relationships between our features and the probability that an individual is unhappy. We trained and tested three models to predict happiness: Random Forest, Logistic Regression, and Support Vector Machine.

Our class weights are quite imbalanced (14% Unhappy / 86% Happy), so we selected the Receiver Operating Characteristic Area Under the Curve (ROC AUC) as our evaluation metric. This is preferable to accuracy, because it is not sensitive to class weights. After using GridSearch to identify optimal model hyperparameters, the best performing models were Random Forest and Logistic Regression, both of which had 0.68 ROC AUC score. SVM performed significantly worse, with a score of only 0.58.

Below is a plot of the ROC Curve (true positive rate TPR vs false positive rate FPR tradeoff) for the Random Forest and Logistic Regression models, along with an orange line representing the baseline (what we would expect if we were guessing). The further an ROC curve is away from that baseline, the better a model has performed. A perfect model would have a TPR of 1 and FPR of 0. Using out-of-sample data, the Logistic Regression model performed slightly better than Random Forest (0.71 versus 0.70).


<img src=https://github.com/cmgerr/General-Social-Survey/raw/master/Images/ROC_Curve.png width="600">

In order to select the recommended probability threshold, we also plotted the True Positive Rate, False Positive Rate, and Precision (percent of predicted unhappy people who are truly unhappy) against the model's predicted probability. By shifting the threshold used to predict unhappiness, we can therefore shift where we lie on each of these curves. 


<img src=https://github.com/cmgerr/General-Social-Survey/raw/master/Images/Model_Value_Add.png width="600">

Our recommendation is to use the Logistic Regression model with a probability threshold for predicting unhappiness of 0.15. This threshold yields TPR of 0.68, FPR of 0.36 and Precision of 0.24.

# Further Enhancements

Given the success of both the Random Forest model and the Logistic Regression model, one potential enhancement is the use of a Voting Classifier model that incorporates both the Random Forest and Logistic Regression.

The models could also likely be further strengthened by the addition of features that would be available to health practitioners, for instance: past incidence of particular health conditions, BMI, and income or credit score information.

