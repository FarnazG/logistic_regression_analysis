
# Customer Churn Analysis


## Introduction

In this project, we are going to build a binary classifier to predict whether a customer will "soon" stop doing business with SyriaTel, a telecommunications company. 


## Project Outlines:

1.Data cleaning and pre-processing

2.Modeling

3.Evaluating and choosing the final model

4.Recommendations

5.Future works


## Project Breakdown:

### 1. Data cleaning and preprocessing

Importing libraries and available data files and Checking for:

* Missing data and placeholders
* Data types
* Multicollinearity
* Duplicates and outliers
* Data distributions
* Data range and scalling
* Categorcal data 

In this project, after pre-processing steps, 4 different data-sets was prepared:

data = original dataframe

data1 = a copy of the original dataframe data in which:
* log transfer was applied on skewed distributions
  
data2 = a copy of the original dataframe data in which:
* features were scaled

data3 = a copy of the original dataframe data in which:
* features were log-transfered and scaled


### 2. Modeling

* Separating feaures and target columns in the dataset
* Splitting train and test data
* Working on rebalancing the class imbalance for each dataset

![alt text](https://github.com/FarnazG/project003/blob/master/images/class-imbalance.png)

* Building the basic model(s)(Logistic Regression) for each dataset
* Creating confusion matrix and obtaining classification report for each dataset
* Creating ROC curve and AUC for each dataset

At this step we compare and choose the best dataset to work with

![alt text](https://github.com/FarnazG/project003/blob/master/images/ROC-curve.png)

* Hyperparameter tuning
* Feature selection

![alt text](https://github.com/FarnazG/project003/blob/master/images/feature_importance.png)


### 3. Evaluating models and choosing the final model 

* Exploring a few models considering different features and hyperparameters
* Visualizing scatter plots of each feature with the probabilities of leaving

![alt text](https://github.com/FarnazG/project003/blob/master/images/customer_service_calls.png)

![alt text](https://github.com/FarnazG/project003/blob/master/images/total-day-charge.png)

* Choosing the best model
* Evaluating the financial impact of the final model prediction on the SyriaTel company


### 4. Recommendations for the company

By default, we do not know the budget of company to distribute promotion and offers and their ability to change any charge rates, so we only suggest our formula based on the raw input

The policy can be offering promo plans to potential leaving customers to motivates them to stay:

Promo_Plan to keep customer motivated = [(potential yearly promo) + (12* potential monthly discount)]


To choose the optimal model based on the confusion matrix info, the model should maximize the benefit equation:

1. True Negative predictions of the model (TN) : cost 

* Revenue loss/year = TN x ($monthly contract rate*12)
* The company will lose its customer through canceling the service and discontinuing the subscription

2. False Positive predictions of model(FP) : cost  

* Revenue loss/yearly = FP x ($monthly contract rate*12)
* The model predicts that SyriaTel costumer will continue doing business with the company but they don’t ! In this case, the model will instruct the website not to offer a promo plan or discount, thus it will lose the  opportunity to keep making revenue from a potential customer

3. False Negative (FN ) : Benefit  

* Benefit on revenue & loss on promo plans = FN x [($ monthly contract rate*12) - ($ promo plan)]
* The company will earn money in revenue if the model predicts a customer will be leaving , thus a promo plan or discount is offered to motivate the customer to stay. the customer is actually staying so the company does not lose the revenue but it will lose money on the promo plans and discounts

4. True Positive (TP) : Benefit 

* Revenue gain/year = TP x ($ monthly contract*12)
* The company will earn in revenue if our model correctly predicts a customer continues doing business with them, thus no discount or promo is offered


**Benefit equation:** 

[TP*(monthly_contract*12)]-[TN*(monthly_contract*12)]-[FP*(monthly_contract*12)]+[FN*[(monthly_contract*12)-(promo_plan)]]

So, the predicting model will best benefit the company if limits the false positive predictions. 


**Final model:**

most important features: 

1. International plan

2. Total day charge

3. Customer service calls

4. Number vmail messages


* Logistic regression with tuned Hyperparameters:

```javascript
{
    'C': 0.1, 
    'class_weight': 'balanced',  
    'penalty': 'l1', 
    'random_state': 10, 
    'solver': 'liblinear'
} 
```
* Model Test Data Precision: 0.345
* Model Test Data Accuracy: 0.738 
* Model Test Data AUC: 0.845
  
![alt text](https://github.com/FarnazG/project003/blob/master/images/confusion_matrix.png)

![alt text](https://github.com/FarnazG/project003/blob/master/images/model-ROC-curve.png)


## Non-technical Presentation

[customer-churn-analysis-presentation](https://github.com/FarnazG/project003/blob/master/customer-churn-analysis-presentation.pdf)


### 5. Future Work:

* Testing other classification algorithms and compare the results to our existing model. Decision Trees, Random Forests, and Support Vector Machines are a few other classifiers to consider testing.

* Defining a function to show the affect of location and each specific state on the leaving probability when all other features are the same.

* Identifying new promotion plans to higher the benefit and lower the probability of leaving the company.

