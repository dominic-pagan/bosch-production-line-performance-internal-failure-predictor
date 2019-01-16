
# coding: utf-8


# train_numeric.csv -
# train_date.csv

# In[1]:


# Import the packages
import pandas as pd
import numpy as np 
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn import grid_search
from sklearn import preprocessing


# In[2]:


DataPath='C:\\Users\\Jameel shaik\\Downloads\\Dezyre\\HackerDay\\Bosch Performance line\\'


# In[3]:


train_numeric=pd.read_csv(DataPath+'train_numeric.csv',nrows=10000)
train_date=pd.read_csv(DataPath+'train_date.csv',nrows=10000)
#train_cat=pd.read_csv(DataPath+'train_cat.csv',nrows=10000)


# In[4]:


train_numeric.shape  # (Rows,Col)


# In[5]:


train_numeric.head()


# In[6]:


train_date.head()


# In[7]:


train_numeric.describe


# In[8]:


data_merge = pd.merge(train_numeric,train_date,on = 'Id')
data_merge.head()


# In[9]:


dataclean = data_merge.dropna(axis=1,thresh = int(len(data_merge)*0.5))
dataclean = dataclean.fillna(0)


# In[10]:


dataclean.head()


# In[11]:


1 column: 50% data whih is filled and 50 % are emtpy
1) Data imbalance : Noisy, overfit,  


# In[12]:


# label the encoder  ( aligning he labels in order)
 
le = preprocessing.LabelEncoder()
dataclean['Id'] = le.fit_transform(dataclean.Id)


# In[13]:


# Splitting my data into Training and testing  by ignoring ID column as its Identical column
featurelist =  list(dataclean.columns.values)
featurelist.remove('Id')
featurelist.remove('Response')
features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(dataclean[featurelist],
                                                              dataclean['Response'], test_size=0.1, random_state=42)


# Training data
# features_train  # ind columns
# labels_train  # dependent columns
# Testing Data
# features_test # ind columns
# labels_test# dependent columns

# In[14]:


# 10k -- accuracy 92 % 99 %   (on sample the accuray may be higher but when we consider total amount of data we the accuray can goes down to )

# 80%   --- 89% 


# In[15]:


#########################
######### Naive Bayes###########
##################################


# In[16]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB


# In[17]:


naive_bayes = BernoulliNB()
naive_bayes.fit(features_train,labels_train)


# In[18]:


p_station = naive_bayes.predict_proba(features_test)
p_station


# In[28]:


# 0 = Not failure, 1  = Failure
pred = naive_bayes.predict(features_test)
pred


# In[29]:


labels_test.shape


# In[30]:


pred.shape


# In[32]:


accuracy = accuracy_score(labels_test,pred)
accuracy


# In[ ]:


############# 
## Random Forest Classifier################
#####################


# In[33]:


from sklearn.ensemble import RandomForestClassifier


# In[34]:


clf = RandomForestClassifier(100, max_depth = 20, n_jobs =3)


# In[35]:


clf


# In[36]:


clf.fit(features_train,labels_train)


# In[40]:


accuracy = accuracy_score(labels_test,pred)
accuracy


# In[37]:


pred = clf.predict(features_test)
pred


# In[ ]:


##################
### Grid Search##############
#################


# In[45]:


param_grid= {  "criterion" : ['gini','entropy'],
                 "min_samples_split": [2,4,5,6,7,8,9,10],
                 "max_depth" : [None,2,4],
                 "min_samples_leaf" :[1,3,5,6,7,8,10],
                 'n_estimators':[20,30,50,70],
                     'n_jobs' :[-1]
             
}


# In[46]:


modeloptimal = grid_search.GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, scoring='f1', cv=5)
modeloptimal


# In[ ]:


modeloptimal.fit(features_train, labels_train)


# In[ ]:


# Using this we can find the best acuracy model from the above parameter estimators
clf = modeloptimal.best_estimator_
clf


# In[ ]:


pred = clf.predict(features_test)
pred


# In[ ]:


accuracy = accuracy_score(labels_test)
accuracy


# In[19]:


#######################################
### Extra Tree Classifier################
##################################


# In[23]:


import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


# In[24]:


clf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1,min_samples_leaf= 10, verbose = 1)


# In[25]:


clf.fit(features_train,labels_train)


# In[26]:


pred = clf.predict(features_test)
pred


# In[27]:


accuracy = accuracy_score(labels_test, pred)
accuracy


# In[28]:


################################
## xgboost ###########
####################################


# In[29]:


import xgboost as xgb
from sklearn.grid_search import GridSearchCV


# In[30]:


cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1) 


# In[ ]:


optimized_GBM.fit(features_train, labels_train)


# In[ ]:


optimized_GBM.grid_scores_


# In[ ]:


cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth': 3, 'min_child_weight': 1}


optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
optimized_GBM.fit(features_train, labels_train)


# In[ ]:


optimized_GBM.grid_scores_


# There are a few other parameters we could tune in theory to squeeze out further performance, but this is a good enough starting point.
# 
# To increase the performance of XGBoost’s speed through many iterations of the training set, and since we are using only XGBoost’s API and not sklearn’s anymore, we can create a DMatrix. This sorts the data initially to optimize for XGBoost when it builds trees, making the algorithm more efficient. This is especially helpful when you have a very large number of training examples. To create a DMatrix:

# In[ ]:


xgdmat = xgb.DMatrix(features_train, labels_train) # Create our DMatrix to make XGBoost more efficient


# In[ ]:


our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1} 
# Grid Search CV optimized settings

cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 3000, nfold = 5,
                metrics = ['error'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error


# We can look at our CV results to see how accurate we were with these settings. The output is automatically saved into a pandas dataframe for us.

# In[ ]:


cv_xgb.tail(5)


# In[ ]:


Now that we have our best settings, let’s create this as an XGBoost object model that we can reference later.


# In[ ]:


our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1} 

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 432)


# In[31]:


get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set(font_scale = 1.5)


# In[ ]:


xgb.plot_importance(final_gb)


# In[ ]:


importances = final_gb.get_fscore()
importances


# In[ ]:


importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)
importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')


# Analyzing Performance on Test Data
# 
# The model has now been tuned using cross-validation grid search through the sklearn API and early stopping through the built-in XGBoost API. Now, we can see how it finally performs on the test set. Does it match our CV performance? First, create another DMatrix (this time for the test data).

# In[ ]:


testdmat = xgb.DMatrix(features_test, labels_test)


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = final_gb.predict(testdmat) # Predict using our testdmat
y_pred


# In[ ]:


accuracy_score(y_pred, labels_test), 1-accuracy_score(y_pred, labels_test)


# In[32]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# In[ ]:


# test set 
test_numeric = pd.read_csv(Datapath+'test_numeric.csv')
test_date = pd.read_csv(Datapath+'test_date.csv')
data_merge = pd.merge(test_numeric, test_date, on='Id',suffixes=('num', 'date'))
# 


# In[ ]:


def makesubmit(clf,testdf,featurelist,output="submit.csv"):
    testdf = testdf.fillna(0)
    feature_test = testdf[featurelist]
    
    pred = clf.predict(feature_test)
    
    ids = list(testdf['Id'])
    
    fout = open(output,'w')
    fout.write("Id,Response\n")
    for i,id in enumerate(ids):
        fout.write('%s,%s\n' % (str(id),str(pred[i])))
    fout.close()


# In[ ]:


makesubmit(clf,data_merge,featurelist,output="submit.csv")


# In the first step, we import standard libraries and fix the most essential features as suggested by an XGB

# In[33]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import seaborn as sns

feature_names = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114']


# We determine the indices of the most important features. After that the training data is loaded

# In[38]:


numeric_cols = pd.read_csv(DataPath+"train_numeric.csv", nrows = 10000).columns.values
imp_idxs = [np.argwhere(feature_name == numeric_cols)[0][0] for feature_name in feature_names]
train = pd.read_csv(DataPath+"train_numeric.csv", 
                index_col = 0, header = 0, usecols = [0, len(numeric_cols) - 1] + imp_idxs)
train = train[feature_names + ['Response']]


# The data is split into positive and negative samples.

# In[39]:


X_neg, X_pos = train[train['Response'] == 0].iloc[:, :-1], train[train['Response']==1].iloc[:, :-1]


# # Univariate characteristics

# In order to understand better the predictive power of single features, we compare the univariate distributions of the most important features. First, we divide the train data into batches column-wise to prepare the data for plotting.

# In[40]:


BATCH_SIZE = 5
train_batch =[pd.melt(train[train.columns[batch: batch + BATCH_SIZE].append(np.array(['Response']))], 
                      id_vars = 'Response', value_vars = feature_names[batch: batch + BATCH_SIZE])
              for batch in list(range(0, train.shape[1] - 1, BATCH_SIZE))]


# After this split, we can now draw violin plots. Due to memory reasons, we have to split the presentation into several cells. For many of the distributions there is no clear difference between the positive and negative samples.

# In[41]:


FIGSIZE = (12,16)
_, axs = plt.subplots(len(train_batch), figsize = FIGSIZE)
plt.suptitle('Univariate distributions')
for data, ax in zip(train_batch, axs):
    sns.violinplot(x = 'variable',  y = 'value', hue = 'Response', data = data, ax = ax, split =True)


# # Correlation structure

# In the previous section we have seen differences between negative and positive samples for univariate characteristics. We go down the rabbit hole a little further and analyze covariances for the negative and positive samples separately.

# In[42]:


FIGSIZE = (13,4)
_, (ax1, ax2) = plt.subplots(1,2, figsize = FIGSIZE)
MIN_PERIODS = 100

triang_mask = np.zeros((X_pos.shape[1], X_pos.shape[1]))
triang_mask[np.triu_indices_from(triang_mask)] = True

ax1.set_title('Negative Class')
sns.heatmap(X_neg.corr(min_periods = MIN_PERIODS), mask = triang_mask, square=True,  ax = ax1)

ax2.set_title('Positive Class')
sns.heatmap(X_pos.corr(min_periods = MIN_PERIODS), mask = triang_mask, square=True,  ax = ax2)


# The difference between the two matrices is sparse except for three specific feature combinations.

# In[43]:


sns.heatmap(X_pos.corr(min_periods = MIN_PERIODS) -X_neg.corr(min_periods = MIN_PERIODS), 
             mask = triang_mask, square=True)


# Finally, as in the univariate case, we analyze correlations between missing values in different features.

# In[44]:


nan_pos, nan_neg = np.isnan(X_pos), np.isnan(X_neg)

triang_mask = np.zeros((X_pos.shape[1], X_pos.shape[1]))
triang_mask[np.triu_indices_from(triang_mask)] = True

FIGSIZE = (13,4)
_, (ax1, ax2) = plt.subplots(1,2, figsize = FIGSIZE)
MIN_PERIODS = 100

ax1.set_title('Negative Class')
sns.heatmap(nan_neg.corr(),   square=True, mask = triang_mask, ax = ax1)

ax2.set_title('Positive Class')
sns.heatmap(nan_pos.corr(), square=True, mask = triang_mask,  ax = ax2)


# For the difference of the missing-value correlation matrices, a striking pattern emerges. A further and more systematic analysis of such missing-value patterns has the potential to beget powerful features.

# In[45]:


sns.heatmap(nan_neg.corr() - nan_pos.corr(), mask = triang_mask, square=True)


# #Hope you have enjoyed the session.. Have a great day ahead.....!!!
