#1.Importing Libraries
#Datarames and Computation
import numpy as np
import pandas as pd

#Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk")

#machine-learning toolkit:
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import accuracy_score,confusion_matrix, recall_score, precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings('ignore')

#train test split
from sklearn.model_selection import train_test_split

#k-fold cross validation
from sklearn.model_selection import cross_val_score

#PICKLE for saving objects
import pickle

#smote for balancing data
import imblearn
from imblearn.over_sampling import SMOTE
########################################

#2.Importing Data
data = pd.read_csv("bigml_59c28831336c6604c800002a.csv")

#Changin column names to a python format
data.columns = data.columns.str.strip().str.replace(' ', '_')
data.columns
########################################

#3.Data Cleaning and preprocessing

#Checking for data types
data.info()

#Changing data types
data["international_plan"]=data.international_plan.replace("yes","1").replace("no","0").astype(int)
data["voice_mail_plan"]=data.voice_mail_plan.replace("yes","1").replace("no","0").astype(int)
data["churn"]=data.churn.replace("True","1").replace("False","0")
data["churn"]=data.churn.astype(int)

data=data.drop(['phone_number'], axis=1)

#Changing state format from text to numbers
data.state.unique()
data_dummies=pd.get_dummies(data, columns=['state'], drop_first=True)
data_dummies.head()

#Checking for missing data and placeholders
data.shape
data.isna().sum()

#check for place holders
for column in data:
    columnSeriesdf = data[column]
    print('Colunm Name : ', column)
    print('Column Contents : ', columnSeriesdf.unique())

#we have a dataset with no missing data and no place holders

########################################

#4.Initial check for collinearity
# without considering state , phone_number
data.drop(['state'],axis=1).corr()['churn'].sort_values(ascending=False)

#Heatmap of all correlation coefficients
plt.figure(figsize=(50,45))
ax= sns.heatmap(data.corr(), annot=True, cmap=sns.color_palette('coolwarm'), center=0, linewidths=.7, square=True, annot_kws={"size":7})
plt.xticks(size = 9)
plt.yticks(size = 9)
plt.show()

data=data.drop(['voice_mail_plan','total_day_minutes','total_eve_minutes','total_night_minutes','total_intl_minutes'], axis=1)

########################################

#Check for distributions, outliers and categorical data
fig = data.hist(bins=50, figsize=(12,9), grid=True)
plt.tight_layout();
plt.show()

#categorical data
data.area_code.value_counts()
data =pd.get_dummies(data, columns=['state', 'area_code'], drop_first=True)

#checking for outliers
#creating box plots for features with similar distribution range:
columns=['account_length','total_day_calls','total_eve_calls','total_night_calls']
data[columns].boxplot(figsize=(15,5), rot = 45)
plt.xlabel('Features', size = 13)
plt.title('Features Box Plot', size = 16)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.ylim(0,250);
plt.show()

# dropping outliers from the features:
data = data.loc[data['account_length']<=200]
data = data[(data['total_day_calls'] >= 50) & (data['total_day_calls'] <= 150)]
data = data[(data['total_eve_calls'] >= 50) & (data['total_eve_calls'] <= 150)]
data = data[(data['total_night_calls'] >=50) & (data['total_night_calls'] <= 150)]

# creating another box plots for the rest of featurs with similar distribution range:
columns1=['number_vmail_messages','total_day_charge','total_eve_charge','total_night_charge','total_intl_calls','customer_service_calls']
data[columns1].boxplot(figsize=(15,5), rot = 45)
plt.xlabel('Features', size = 13)
plt.title('Features Box Plot', size = 16)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.ylim(0,60);
plt.show()

# dropping outliers from the features:
data = data.loc[data['number_vmail_messages']<=50]
data = data[(data['total_day_charge'] >= 6) & (data['total_day_charge'] <= 55)]
data = data[(data['total_eve_charge'] >=7) & (data['total_eve_charge'] <= 27)]
data = data[(data['total_night_charge'] >=3) & (data['total_night_charge'] <= 15)]
data = data[(data['total_intl_calls'] <=10)]
data = data[(data['total_intl_charge'] >=1) & (data['total_intl_charge'] <= 4.5)]
data = data[(data['customer_service_calls'] <=4)]

#checking for data distributions
data.number_vmail_messages.value_counts().sort_index()

#log(x+1) to have a smoother distribution on number_vmail_messages feature.
#data1 = a copy of the original dataframe data in which 'number_vmail_messages' Log transfered
data1 = data.copy(deep=True)
data1['number_vmail_messages'] = data['number_vmail_messages'].apply(lambda x: np.log(x+1))
sns.distplot(data['number_vmail_messages'], bins=30, kde=True, rug=False, color="purple")
plt.show()

########################################

#5.spliting data into train/test

# separating data features and target:
x = data.drop(columns = 'churn')
y = data.churn

# separating data1 features and target:
x1 = data1.drop(columns = 'churn')
y1 = data1.churn
x1.head()

# creating a list of different datasets and their train/test data
x_train_list= list()
y_train_list= list()
x_test_list= list()
y_test_list= list()

# split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20)
# get the dimenssion of each ub dataframe
x_train.shape
y_train.shape
x_test.shape
y_test.shape
# add train and test data to the list
x_train_list.append(x_train)
y_train_list.append(y_train)
x_test_list.append(x_test)
y_test_list.append(y_test)

# split the data1
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = .20)
# get the dimenssion of each ub dataframe
x1_train.shape
y1_train.shape
x1_test.shape
y1_test.shape
# add train and test data1 to the list
x_train_list.append(x1_train)
y_train_list.append(y1_train)
x_test_list.append(x1_test)
y_test_list.append(y1_test)

########################################

#6.scalling

#create a copy of data to perform MINMAX scaler:
data2 = data.copy(deep=True)
# separating data2 features and target:
x2 = data2.drop(columns = 'churn')
y2 = data2.churn
x2.head()
# split the data2
from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size = .20)

#MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x2_train= scaler.fit_transform(x2_train)
x2_test= scaler.transform(x2_test)

#x2_train.columns are equal to x2.columns
x2_train = pd.DataFrame(x2_train, columns = x2.columns)
# x2_train.head() = x_train_list[2].head()
# y2_train.head()

# add train and test data2 to the list
x_train_list.append(x2_train)
y_train_list.append(y2_train)
x_test_list.append(x2_test)
y_test_list.append(y2_test)

x_train_list[2].head()

# take a quick look at our normalized feature variables:
x2_train.hist(bins= 40, figsize  = [26, 26])
plt.show();


#create a copy of data1 to perform MINMAX scaler:
data3 = data1.copy(deep=True)
# separating data3 features and target:
x3 = data3.drop(columns = 'churn')
y3 = data3.churn
x3.head()
# split the data3
from sklearn.model_selection import train_test_split
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size = .20)

# MINMAXScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x3_train = scaler.fit_transform(x3_train)
x3_test = scaler.transform(x3_test)

# x2_train.columns are equal to x2.columns
x3_train = pd.DataFrame(x3_train, columns = x3.columns)
x3_train.head()
#y3_train.head()

# append train and test data3 to the list:
x_train_list.append(x3_train)
y_train_list.append(y3_train)
x_test_list.append(x3_test)
y_test_list.append(y3_test)

x_train_list[3].head()

########################################

#7. Check for class imbalance

# for the original data before splitting
x = data.drop(columns = 'churn')
y = data.churn
y.value_counts()

plt.figure(figsize=(4,6))
sns.countplot(y, palette='Reds')

plt.xticks(size = 13)
plt.yticks(size = 13)

plt.title('Number of Customers Leaving vs Staying in Business', size = 15)

positions = (0,1)
labels = ("Staying=2487","Leaving=357")
plt.xticks(positions, labels)

plt.xlabel('churn', size = 13)
plt.ylabel('Count', size = 13);

#for training data
y_train.value_counts()
plt.figure(figsize=(4,6))
sns.countplot(y_train, palette='Blues')
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.title('Target Variable', size = 15)
plt.xlabel('Churn', size = 13)
plt.ylabel('Count', size = 13);

# a for loop to apply SMOTE on each dataset in the list of train and tests:
from imblearn.over_sampling import SMOTE, ADASYN

x_trainb_list = list()
y_trainb_list = list()

for i in range(4):
    x_train_balanced, y_train_balanced = SMOTE().fit_sample(x_train_list[i], y_train_list[i])
    print("dataset: ", i)
    print(y_train_list[i].value_counts())
    print("balanced")
    print(pd.Series(y_train_balanced).value_counts(), "\n")

    # new balanced features and target data
    x_trainb = pd.DataFrame(x_train_balanced, columns=x_train_list[i].columns)
    y_trainb = y_train_balanced

    # add the balanced training data for all 4 datasets into the list
    x_trainb_list.append(x_trainb)
    y_trainb_list.append(y_trainb)

########################################

# Data Preprocessing is completed, to compare the best pre-process for the model, we are working with all these different datasets:

#the original data balanced:x_trainb_list[0], y_trainb_list[0]
#the original data log-transfered and balanced: x_trainb_list[1], y_trainb_list[1]
#the original data standardscaled and balanced: x_trainb_list[2], y_trainb_list[2]
#the original data log-transfered, standardscaled and balanced: x_trainb_list[3], y_trainb_list[3]

########################################

#MODELING
#1.Basic Logistic Regression Model:
from sklearn.linear_model import LogisticRegression

y_pred_list = list()
for i in range(4):
    logreg = LogisticRegression(random_state=10, solver= "liblinear")
    model = logreg.fit(x_trainb_list[i], y_trainb_list[i])
    y_pred=logreg.predict(x_test_list[i])
    y_pred_list.append(y_pred)
    score=logreg.score(x_test_list[i], y_test_list[i])
    print('logistic regression score, dataset:',i)
    print(score)

#2.Evaluating Predictions and Confusion Matrix
#2a)Classification Report
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,classification_report
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

def report(y_test,y_pred):
    print("Classification report dataset: ", i)
    print("Accuracy_score:",accuracy_score(y_test,y_pred),"\n")
    print(classification_report(y_test,y_pred), "\n")

for i in range(4):
    report(y_test_list[i], y_pred_list[i])

#2b)Confusion Matrix
def confusion_matrix_report(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm_norm.round(4)
    print("Confusion Matrix dataset: ", i)
    print(cm, "\n")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, cmap='Blues', linewidths=10, center=True)
    plt.yticks(np.arange(2) + 0.5, ('0=cancellation', '1=continuation'), rotation=0, fontsize="15", va="center")
    plt.xticks(np.arange(2) + 0.5, ('0=cancellation', '1=continuation'), rotation=0, fontsize="15", va="center")

    plt.title('Confusion Matrix Rate', size=15)
    plt.xlabel('Predicted label', size=13)
    plt.ylabel('True label', size=13);

    # # fix for mpl bug that cuts off top/bottom of seaborn viz
    # b, t = plt.ylim()  # discover the values for bottom and top
    # b += 0.5  # Add 0.5 to the bottom
    # t -= 0.5  # Subtract 0.5 from the top
    # plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.show()

for i in range(4):
    confusion_matrix_report(y_test_list[i], y_pred_list[i])


#3.ROC Curve
def ROC_report(x_test, y_test, i):
    y_pred_prob = logreg.decision_function(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc_label = 'Initial Model data AUC ' + str(i) + ' : '
    print(auc_label, auc(fpr, tpr))
    legend_label = 'Initial Model ROC curve dataset: ' + str(i)
    cmap = mpl.cm.jet
    plt.plot(fpr, tpr, color = cmap(float(i) / float(4.0)), lw = 2, label =legend_label)

import matplotlib as mpl

plt.figure(figsize=(12,8))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks([i/20.0 for i in range(21)])
plt.xticks([i/20.0 for i in range(21)])
plt.xlabel('False Positive Rate', size = 10)
plt.ylabel('True Positive Rate', size = 10)
plt.title('Receiver operating characteristic (ROC) Curve', size = 14)

for i in range(4):
    cm = ROC_report(x_test_list[i], y_test_list[i], i)

plt.legend(loc="lower right");
plt.show()

#ROC curve shows the best preprocessing for the logistic regression has been done on dataset 2, we continue working on our model with the dataset 2

#4.Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

grid = {"C":[0.001, 0.01, 0.1, 1, 10, 100], "penalty":['l1', 'l2', 'none'],
        'class_weight': ['balanced', None], "solver" : ['liblinear', 'saga'],
        "random_state":[10,15, None]}

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=3)
logreg_cv.fit(x_trainb_list[2], y_trainb_list[2])

print("Tuned Hyperparameters: ",logreg_cv.best_params_)
print("Accuracy: ",logreg_cv.best_score_)

#5.feature selection

#5a) Recursive Feature Elimination with Cross-Validation for n-of-features to include in model:
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import KFold

rfecv = RFECV(estimator=logreg, step=2, cv=StratifiedKFold(5), scoring="accuracy")
#rfecv = RFECV(estimator=logreg, step=1, cv=KFold(10), scoring="accuracy")

rfecv.fit(x_trainb_list[2], y_trainb_list[2])
print("Optimal number of features: {}".format(rfecv.n_features_))

# Plotting the best number of features with respect to the Cross Validation Score:
plt.figure(figsize=(9,5))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (# of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#5b)Recursive Feature Elimination for finding top features:
# Feature extraction on balanced data2
from sklearn.feature_selection import RFECV, RFE

model = LogisticRegression(solver='liblinear')
selector = RFE(estimator= model, n_features_to_select=15, step=0.8)
fit = selector.fit(x_train_list[2], y_train_list[2])

print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

#Explore the top most significant features based upon their coefficients and correlation with the target:
logreg = LogisticRegression(random_state=10, solver="liblinear")
model = logreg.fit(x_trainb_list[2], y_trainb_list[2])
coefs = logreg.coef_.T

#visualizing feature importances based on their coefficients to identify the most impactful feature:
feature_importance = abs(logreg.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure(figsize=(20,20))
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(x_train.columns)[sorted_idx], fontsize=10)
featax.set_xlabel('Relative Feature Importance')

plt.tight_layout()
plt.show()

#According to our different feature selection methods:
#1."international_plan","total_day_charge","total_eve_charge" are the most significant features among all.
#2.location and "state" is also an important feature.
#3."customer_service_calls", "number_vmail_messages", "total_intl_charge" are some other important features to explore.

# exploring some logistic models considering different features and hyperparameters:
def model(logreg, x_train, y_train):
    x_train_balanced, y_train_balanced = SMOTE().fit_sample(x_train, y_train)
    x_train = pd.DataFrame(x_train_balanced, columns=x_train.columns)
    y_train = y_train_balanced

    model_log = logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    y_pred_prob = logreg.decision_function(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    print('Model Test Data Precision: {}'.format(precision_score(y_test, y_pred).round(3)))
    print('Model Test Data Accuracy: {}'.format(accuracy_score(y_test, y_pred).round(3)))
    print('model Test Data AUC: {}'.format(auc(fpr, tpr).round(3)))

    # Confusion matrix Using Heatmap
    cnf_matrix = confusion_matrix(y_test, y_pred)
    tn_cost = (cnf_matrix[0][0] / len(y_test))
    fp_cost = (cnf_matrix[0][1] / len(y_test))
    fn_ben = (cnf_matrix[1][0] / len(y_test))
    tp_ben = (cnf_matrix[1][1] / len(y_test))
    # print("TN+FP=",tn_cost+fp_cost)
    # print("TP+FN=",tp_ben+fn_ben)

    cnf_matrix_norm = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cnf_matrix_norm, annot=True, cmap='Blues', linewidths=10, center=True)
    plt.yticks(np.arange(2) + 0.05, ('0=cancellation', '1=continuation'), rotation=0, fontsize="15", va="center")
    plt.xticks(np.arange(2) + 0.05, ('0=cancellation', '1=continuation'), rotation=0, fontsize="15", va="center")
    plt.title('Confusion Matrix', size=15)
    plt.xlabel('Predicted label', size=13)
    plt.ylabel('True label', size=13);

    #fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.05  # Add 0.5 to the bottom
    t -= 0.05  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.show()
    print()

    # ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i / 20.0 for i in range(21)])
    plt.xticks([i / 20.0 for i in range(21)])
    plt.xlabel('False Positive Rate', size=12)
    plt.ylabel('True Positive Rate', size=12)
    plt.title('Receiver operating characteristic (ROC) Curve', size=14)
    plt.legend(loc="lower right");
    plt.show()


#Tuned Hyperparameters:
# {
#     'C': 10,
#     'class_weight': 'balanced',
#     'penalty': 'l1',
#     'random_state': 10,
#     'solver': 'liblinear'
# }

#Logistic regression model #1:
x = data2[['total_day_charge','international_plan']]
y = data2.churn
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20)
#logreg=LogisticRegression(penalty='l1',class_weight="balanced",C=100,solver='liblinear',random_state=15)
logreg=LogisticRegression(class_weight="balanced",solver='liblinear',C=0.1, penalty='l2')
model(logreg,x_train,y_train)

#Logistic regression model #2:
x = data2[['total_day_charge','international_plan','total_eve_charge']]
y = data2.churn
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20)
#logreg=LogisticRegression(penalty = 'l1', class_weight = "balanced", C = 100, solver= 'liblinear', random_state=10)
logreg=LogisticRegression(class_weight="balanced",solver='liblinear',C=10, penalty='l1')
model(logreg,x_train,y_train)

#Logistic regression model #3:
x = data2[['total_day_charge','international_plan','total_eve_charge','customer_service_calls']]
y = data2.churn
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20)
#logreg=LogisticRegression(penalty = 'l1', class_weight = "balanced", C = 100, solver= 'liblinear', random_state=10)
logreg=LogisticRegression(class_weight="balanced",solver='liblinear',C=1, penalty='l1')
model(logreg,x_train,y_train)

#Logistic regression model #4:
x = data2[['total_day_charge','international_plan','total_eve_charge','customer_service_calls','number_vmail_messages']]
y = data2.churn
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20)
logreg=LogisticRegression(penalty='l1',class_weight="balanced",C=1,solver='liblinear',random_state=0)
model(logreg,x_train,y_train)

#Logistic regression model #5:
x = data2[['total_day_charge','international_plan','total_eve_charge','customer_service_calls','number_vmail_messages']]
y = data2.churn
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20)
logreg=LogisticRegression(penalty='l1',class_weight="balanced",C=0.1,solver='liblinear',random_state=0)
model(logreg,x_train,y_train)

#6.visualizing scatter plots of each feature with the probabilities.

#creating a dataframe with the most important features.
x = data2[['total_day_charge','international_plan','total_eve_charge','customer_service_calls','number_vmail_messages']]

# we add a column to the x dataframe called prob which is the probability of leaving and will be calculated through the logreg.predict_proba for all the columns of x dataframe.
x['y_prob']= logreg.predict_proba(x)[:, 1]

# 1.we add a column to the x dataframe called target which is the true churn column
# 2.we pass the true churn value to the c,so it shows highlight color for 1 (leaving) and darker color for 0(staying).
x['target']= y.values

for col in x.drop(['y_prob', 'target'], axis=1).columns:
    plt.figure(figsize=(8,5))
    plt.scatter(col, 'y_prob', data=x, c='target', cmap= 'coolwarm', alpha=.6)
    plt.xlabel(col)
    plt.ylabel('Probability of leaving')
    plt.axhline(y=0.5, color='g', linestyle='-')
    plt.show()

#Checking scatter plots of each state with every important features:
#create a dataframe with our most important features.
x = data2.iloc[:,[1,4,11,16,34]]
x['y_prob']= logreg.predict_proba(x)[:, 1]
x['target']= y.values

for col in x.drop(['y_prob', 'target'], axis=1).columns:
    for col1 in x.drop(['y_prob', 'target'], axis=1).columns:
        if col == col1:
            continue
        plt.figure(figsize=(8,5))
        plt.scatter(col, col1, data=x, c='target', cmap='coolwarm', alpha=.3)
        plt.xlabel(col)
        plt.ylabel(col1)
        plt.title('leaving customers highlighted')
        plt.show()


#evaluating the result:
#Final model : model #2
# most important features: ['total_day_calls','international_plan','customer_service_calls']
# Logistic regression with tuned Hyperparameters:
# {
#     'C': 0.1,
#     'class_weight': 'balanced',
#     'penalty': 'l1',
#     'random_state': 10,
#     'solver': 'liblinear'
# }
# Model Test Data Precision: 0.345
# Model Test Data Accuracy: 0.738
# Model Test Data AUC: 0.845

