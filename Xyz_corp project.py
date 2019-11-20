#%%
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#to show all columns
pd.set_option('display.max_columns',None)
#%%
#importing dataset
df = pd.read_table(r'C:\Users\Rohan\Desktop\Data science\DataSets\XYZCorp_LendingData.txt',sep='\t')
df.head()
#%%
#shape of the data set
df.shape
#%%
## Pre processing the data
#create a copy of the dataframe
df_rev = pd.DataFrame.copy(df)
#checking null values
df_rev.isnull().sum()
#%%
#creating dataframe of variables of df_rev which have more than 40% of null values
na = df_rev.isnull().sum()
na=na[na.values > (0.4*len(df))]

#variables names which have more than 40% null values
na.index
#%%
"""droping above varibales and some other variables which are insignificant 
like id,member_id etc
id and member_id  have unique values so they will not give any significant 
info to model
policy_code for all observation is same so it is irrelevant
emp_title and his working experience i.e.emp_length will not help us to 
predict that the person is defaul or not
zip code and addr_state are irrelevantas they will not tell us 
about person is going to be defaulter or not
pymnt_plan has only two values 'y' and 'n' but it highly biased 
to 'n' as there are only 5 rows has pymnt_plan 'y'
"""
df_rev.drop(['desc', 'mths_since_last_delinq', 'mths_since_last_record',
       'mths_since_last_major_derog','dti_joint',
       'verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
       'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
       'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi',
       'total_cu_tl', 'inq_last_12m','id','member_id','policy_code','emp_title',
       'title','zip_code','next_pymnt_d','addr_state','emp_length','pymnt_plan','acc_now_delinq'],
       axis=1,inplace=True)
#checking new shape of dataset 
df_rev.shape
#%%
#droping rows which has null values since both are date variables it doesn't
# make sense to impute them 
df_rev.dropna(subset=['last_pymnt_d'],how='all',inplace=True)
df_rev.dropna(subset=['last_credit_pull_d'],how='all',inplace=True)
#%%
#converting some variables to date format as problem statement say we have to split data on the basis of issue date
df_rev['issue_d']=pd.to_datetime(df_rev['issue_d'])
#checking data types of variables 
df_rev.dtypes
#head of the dateset
df_rev.head()
#%%
#checking for null values
df_rev.isnull().sum()
#%%
#Since there are no missing values in object variables 
#now we have null values in numeric variables only so we will imputing that 
#missing values with median
for x in df_rev.columns[:]:
    if df_rev[x].dtype=='int64' or df_rev[x].dtype=='float64':
        df_rev[x].fillna(df_rev[x].median(),inplace=True)
		
#checking whether null are still there or not?
df_rev.isnull().sum()
#%%
#checking desciption of data set
df_rev.describe()
#%%
# so if you notice the min value in annual income is 0 which is not possible,
#no company will give you loan if you have 0 income
df_rev[df_rev['annual_inc']==0.0]   #462577  508976
#%%
#deleting that two rows
df_rev.drop([462577,508976],axis=0,inplace=True)
#%%
#fetching out the categorical variables from dataset for label encoding
colname =[]
for j in df_rev.columns[:]:
    if df_rev[j].dtype == 'object':
        colname.append(j)
colname
#%%
# label encoding the object variable using LabelEncoder()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for j in colname:
    df_rev[j] = le.fit_transform(df_rev[j].astype(str))

#%%
#converting datetime var to integer using toordinal() function 
#it counts the day from 1 jan 0001
df_rev['issue_d']=df_rev['issue_d'].apply(lambda x:x.toordinal())
df_rev.head()
#%%
#creating indepedant(x) and dependant(y) variables in form array just to make model process time less
X = df_rev.values[:,:-1]
Y = df_rev.values[:,-1]
X
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
print(X)
#%%
#converting Y to integer
Y=Y.astype(int)
#%%
# Running a basic model
"""we will convert may 2015 to to by toordinal function and then manually calculate the 
scaled value of it by standardize method
and we got may 2015 ordinal value as 735719 and scaled value of may 2015 is
0.6548243942563992 
so whatever numbers are less than 0.6548243942563992 are months before may 2015 which is our 
training data and greater than are months after may 2015 which are testing data
and since we don't have any varibale to split y variable we will split y variable using our data!
"""
#splitting the data into trainning data and testing data!
X_train=X[X[:,11]<=0.6548243942563992,:]
Y_train=df_rev.loc[df_rev['issue_d']<=735719,'default_ind']
X_test=X[X[:,11]>0.6548243942563992,:]
Y_test=df_rev.loc[df_rev['issue_d']>735719,'default_ind']

#%%
#Model Building
from sklearn.linear_model import LogisticRegression
#create a model
classifier = LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

print(classifier.coef_)
print(classifier.intercept_)
#%%
#accuracy and confusion matrix and classification report
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report : ")

print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("Accuracy of the model : ",acc)
#%%
# Adjusting the threshold
#store the predicted probabilities
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)

# setting threshold = 0.4
y_pred_class = []
for value in y_pred_prob[:,1]:
    if value > 0.40:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)
#%%
#checking accuracy and confusion matrix for threshold=0.4
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm = confusion_matrix(Y_test,y_pred_class)
print(cfm)

print("Classification report : ")

print(classification_report(Y_test,y_pred_class))

acc = accuracy_score(Y_test,y_pred_class)
print("Accuracy of the model : ",acc)
#%%
#checking type 1 error and type 2 error for differnt threshold
for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :",
           cfm[1,0]," , type 1 error:", cfm[0,1])
#%%
#ROCR curve
from sklearn import metrics

fpr, tpr,z = metrics.roc_curve(Y_test, y_pred_class)
auc = metrics.auc(fpr,tpr)
print(auc)      #arear under the curve
print(fpr)      #false positive rate
print(tpr)      #true positive rate

#%%
#ROC curve 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
#%%
#Rea under the curve with probalities value
from sklearn import metrics

fpr, tpr,z = metrics.roc_curve(Y_test, y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc)
#print(z)

#ROC curve with Probabilities value
import matplotlib.pyplot as plt
#%matplotlib inline
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
"""
try and test the auc for multiple ambigues thresholds where errors 
are almost similar we compare there auc value with overall auc value 
and we finalize upon that threshold which gives us an auc closer to overall auc"""
#%%
#Using cross validation for logistic

classifier=(LogisticRegression())

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%%
"""Even we are getting high accuarcy for logidtic model our precision for class of importance is 
very low so we are trying new algorith
since our dataset is big SVM and KNN are time consuming we are trying decision tree"""
#%%
#splitting the data into trainning data and testing data!
#split the data into test and train
X_train=X[X[:,11]<=0.6548243942563992,:]
Y_train=df_rev.loc[df_rev['issue_d']<=735719,'default_ind'].values
X_test=X[X[:,11]>0.6548243942563992,:]
Y_test=df_rev.loc[df_rev['issue_d']>735719,'default_ind'].values

#%%
#predicting using the decision_tree_classifier
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(random_state=10,min_samples_leaf=5,max_depth=10)

#fir the model on the data and predict the values
dt.fit(X_train,Y_train)
#%%
#PREDICTEING y VALUE 
Y_pred=dt.predict(X_test)
#%%
#checking accuracy and confusion matrix for decision tree
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report : ")

print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("Accuracy of the model : ",acc)
#%%
#features impoertance
print(list(zip(colname,dt.feature_importances_)))	
#%%
#Using cross validation

classifier=(DecisionTreeClassifier())

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=5)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

#%%
#_______________________________________________________

for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])


Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report : ")

print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("Accuracy of the model : ",acc)

#%%
from sklearn import tree
with open("dt.txt","w") as f:
    
    f = tree.export_graphviz(dt,feature_names=colname[:-1],out_file=f)
    
#generate the file and upload the code in webgraphviz.com to plot the decision tree
#C:\Users\Admin\.ipynb_checkpoints
#----------------------------------THANK YOU----------------------------------                

