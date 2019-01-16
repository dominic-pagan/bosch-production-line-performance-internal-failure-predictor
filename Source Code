# Dataset Introduction

The data for this competition represents measurements of parts as they move through Bosch's production lines. Each part has a unique Id. The goal is to predict which parts will fail quality control (represented by a 'Response' = 1).

The dataset contains an extremely large number of anonymized features. Features are named according to a convention that tells you the production line, the station on the line, and a feature number. E.g. L3_S36_F3939 is a feature measured on line 3, station 36, and is feature number 3939.

On account of the large size of the dataset, we have separated the files by the type of feature they contain: numerical, categorical, and finally, a file with date features. The date features provide a timestamp for when each measurement was taken. Each date column ends in a number that corresponds to the previous feature number. E.g. the value of L0_S0_D1 is the time at which L0_S0_F0 was taken.

In addition to being one of the largest datasets (in terms of number of features) ever hosted on Kaggle, the ground truth for this competition is highly imbalanced. Together, these two attributes are expected to make this a challenging problem.

File descriptions
train_numeric.csv - the training set numeric features (this file contains the 'Response' variable)
test_numeric.csv - the test set numeric features (you must predict the 'Response' for these Ids)
train_categorical.csv - the training set categorical features
test_categorical.csv - the test set categorical features
train_date.csv - the training set date features
test_date.csv - the test set date features
sample_submission.csv - a sample submission file in the correct format

### Road map for the Project: ###
- features, dates 2189 features

- Large dimensional data 

- Import the packages

- pandas dataframe:
  - merging the data
    
- Training data : 
  - Summary Statistics
  - Preprocesing 
  - Data Imputation 

- Modelling : 
  - x1 -  0, 1
    
 - a Naive Bayes KDE: 
      - Decision Tree Classifer
        - 2) Extra Tree classifer 
        - 3) Random Forest
        - 4) Grid Search CV
        - 5) XGBoost
 
 - Evaluation:
   - which one has the best accuray(prediction)
   - prediction result (internal part) 
   - feature significance
   - top Features : iedentify there ditributions, correlation(density),
   - submit and the result/file     
            
    
    
    


train_numeric.csv -
train_date.csv


```python
# Import the packages
import pandas as pd
import numpy as np 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
```


```python
DataPath='/Users/dominic/Downloads/'
```


```python
train_numeric=pd.read_csv(DataPath+'train_numeric.csv',nrows=10000)
train_date=pd.read_csv(DataPath+'train_date.csv',nrows=10000)
#train_cat=pd.read_csv(DataPath+'train_cat.csv',nrows=10000)
```


```python
train_numeric.shape  # (Rows,Col)
```




    (10000, 970)




```python
train_numeric.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>L0_S0_F0</th>
      <th>L0_S0_F2</th>
      <th>L0_S0_F4</th>
      <th>L0_S0_F6</th>
      <th>L0_S0_F8</th>
      <th>L0_S0_F10</th>
      <th>L0_S0_F12</th>
      <th>L0_S0_F14</th>
      <th>L0_S0_F16</th>
      <th>...</th>
      <th>L3_S50_F4245</th>
      <th>L3_S50_F4247</th>
      <th>L3_S50_F4249</th>
      <th>L3_S50_F4251</th>
      <th>L3_S50_F4253</th>
      <th>L3_S51_F4256</th>
      <th>L3_S51_F4258</th>
      <th>L3_S51_F4260</th>
      <th>L3_S51_F4262</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>  4</td>
      <td> 0.030</td>
      <td>-0.034</td>
      <td>-0.197</td>
      <td>-0.179</td>
      <td> 0.118</td>
      <td> 0.116</td>
      <td>-0.015</td>
      <td>-0.032</td>
      <td> 0.020</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>  6</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>  7</td>
      <td> 0.088</td>
      <td> 0.086</td>
      <td> 0.003</td>
      <td>-0.052</td>
      <td> 0.161</td>
      <td> 0.025</td>
      <td>-0.015</td>
      <td>-0.072</td>
      <td>-0.225</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>  9</td>
      <td>-0.036</td>
      <td>-0.064</td>
      <td> 0.294</td>
      <td> 0.330</td>
      <td> 0.074</td>
      <td> 0.161</td>
      <td> 0.022</td>
      <td> 0.128</td>
      <td>-0.026</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 11</td>
      <td>-0.055</td>
      <td>-0.086</td>
      <td> 0.294</td>
      <td> 0.330</td>
      <td> 0.118</td>
      <td> 0.025</td>
      <td> 0.030</td>
      <td> 0.168</td>
      <td>-0.169</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> 0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 970 columns</p>
</div>




```python
train_date.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>L0_S0_D1</th>
      <th>L0_S0_D3</th>
      <th>L0_S0_D5</th>
      <th>L0_S0_D7</th>
      <th>L0_S0_D9</th>
      <th>L0_S0_D11</th>
      <th>L0_S0_D13</th>
      <th>L0_S0_D15</th>
      <th>L0_S0_D17</th>
      <th>...</th>
      <th>L3_S50_D4246</th>
      <th>L3_S50_D4248</th>
      <th>L3_S50_D4250</th>
      <th>L3_S50_D4252</th>
      <th>L3_S50_D4254</th>
      <th>L3_S51_D4255</th>
      <th>L3_S51_D4257</th>
      <th>L3_S51_D4259</th>
      <th>L3_S51_D4261</th>
      <th>L3_S51_D4263</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>  4</td>
      <td>   82.24</td>
      <td>   82.24</td>
      <td>   82.24</td>
      <td>   82.24</td>
      <td>   82.24</td>
      <td>   82.24</td>
      <td>   82.24</td>
      <td>   82.24</td>
      <td>   82.24</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>  6</td>
      <td>     NaN</td>
      <td>     NaN</td>
      <td>     NaN</td>
      <td>     NaN</td>
      <td>     NaN</td>
      <td>     NaN</td>
      <td>     NaN</td>
      <td>     NaN</td>
      <td>     NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>  7</td>
      <td> 1618.70</td>
      <td> 1618.70</td>
      <td> 1618.70</td>
      <td> 1618.70</td>
      <td> 1618.70</td>
      <td> 1618.70</td>
      <td> 1618.70</td>
      <td> 1618.70</td>
      <td> 1618.70</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>  9</td>
      <td> 1149.20</td>
      <td> 1149.20</td>
      <td> 1149.20</td>
      <td> 1149.20</td>
      <td> 1149.20</td>
      <td> 1149.20</td>
      <td> 1149.20</td>
      <td> 1149.20</td>
      <td> 1149.20</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 11</td>
      <td>  602.64</td>
      <td>  602.64</td>
      <td>  602.64</td>
      <td>  602.64</td>
      <td>  602.64</td>
      <td>  602.64</td>
      <td>  602.64</td>
      <td>  602.64</td>
      <td>  602.64</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1157 columns</p>
</div>




```python
train_numeric.describe
```




    <bound method DataFrame.describe of          Id  L0_S0_F0  L0_S0_F2  L0_S0_F4  L0_S0_F6  L0_S0_F8  L0_S0_F10  \
    0         4     0.030    -0.034    -0.197    -0.179     0.118      0.116   
    1         6       NaN       NaN       NaN       NaN       NaN        NaN   
    2         7     0.088     0.086     0.003    -0.052     0.161      0.025   
    3         9    -0.036    -0.064     0.294     0.330     0.074      0.161   
    4        11    -0.055    -0.086     0.294     0.330     0.118      0.025   
    5        13     0.003     0.019     0.294     0.312     0.031      0.161   
    6        14       NaN       NaN       NaN       NaN       NaN        NaN   
    7        16       NaN       NaN       NaN       NaN       NaN        NaN   
    8        18    -0.016    -0.041    -0.179    -0.179    -0.056      0.161   
    9        23       NaN       NaN       NaN       NaN       NaN        NaN   
    10       26     0.016     0.093    -0.015    -0.016     0.031      0.116   
    11       27    -0.062    -0.153    -0.197    -0.179    -0.187     -0.384   
    12       28    -0.075    -0.093     0.367     0.348    -0.056      0.025   
    13       31    -0.003    -0.093    -0.161    -0.216     0.118     -0.021   
    14       34    -0.016    -0.138    -0.197    -0.179     0.118     -0.112   
    15       38     0.252     0.250     0.003    -0.016     0.118     -0.294   
    16       41       NaN       NaN       NaN       NaN       NaN        NaN   
    17       44    -0.016    -0.041     0.003    -0.016    -0.143     -0.339   
    18       47       NaN       NaN       NaN       NaN       NaN        NaN   
    19       49     0.088     0.033     0.330     0.348    -0.056      0.161   
    20       52       NaN       NaN       NaN       NaN       NaN        NaN   
    21       55       NaN       NaN       NaN       NaN       NaN        NaN   
    22       56     0.003     0.056    -0.033    -0.034    -0.056      0.116   
    23       57     0.010    -0.034    -0.197    -0.197     0.118      0.025   
    24       63       NaN       NaN       NaN       NaN       NaN        NaN   
    25       68     0.036     0.026     0.312     0.294    -0.013      0.070   
    26       70     0.062     0.071    -0.179    -0.216     0.161      0.025   
    27       71    -0.167    -0.168     0.276     0.330     0.074      0.161   
    28       72       NaN       NaN       NaN       NaN       NaN        NaN   
    29       73       NaN       NaN       NaN       NaN       NaN        NaN   
    ...     ...       ...       ...       ...       ...       ...        ...   
    9970  19865       NaN       NaN       NaN       NaN       NaN        NaN   
    9971  19869       NaN       NaN       NaN       NaN       NaN        NaN   
    9972  19870       NaN       NaN       NaN       NaN       NaN        NaN   
    9973  19871     0.075     0.086    -0.343    -0.325     0.074      0.161   
    9974  19873       NaN       NaN       NaN       NaN       NaN        NaN   
    9975  19874       NaN       NaN       NaN       NaN       NaN        NaN   
    9976  19878    -0.082    -0.064    -0.197    -0.161     0.031      0.161   
    9977  19881    -0.160    -0.175    -0.197    -0.179     0.031      0.161   
    9978  19882       NaN       NaN       NaN       NaN       NaN        NaN   
    9979  19885    -0.010    -0.026    -0.179    -0.197     0.031     -0.066   
    9980  19886       NaN       NaN       NaN       NaN       NaN        NaN   
    9981  19887     0.016     0.041    -0.197    -0.161     0.031      0.161   
    9982  19888     0.069     0.131    -0.052    -0.052    -0.013     -0.066   
    9983  19889     0.088     0.093    -0.015    -0.034     0.074      0.116   
    9984  19893    -0.180    -0.190     0.003    -0.016    -0.143      0.070   
    9985  19895       NaN       NaN       NaN       NaN       NaN        NaN   
    9986  19896       NaN       NaN       NaN       NaN       NaN        NaN   
    9987  19898    -0.108    -0.220    -0.233    -0.179    -0.013      0.116   
    9988  19899       NaN       NaN       NaN       NaN       NaN        NaN   
    9989  19902       NaN       NaN       NaN       NaN       NaN        NaN   
    9990  19904       NaN       NaN       NaN       NaN       NaN        NaN   
    9991  19905       NaN       NaN       NaN       NaN       NaN        NaN   
    9992  19906       NaN       NaN       NaN       NaN       NaN        NaN   
    9993  19909    -0.003    -0.026    -0.052    -0.016    -0.013      0.116   
    9994  19910    -0.023    -0.049    -0.179    -0.161     0.031      0.025   
    9995  19912       NaN       NaN       NaN       NaN       NaN        NaN   
    9996  19915    -0.147    -0.168    -0.033    -0.016     0.074      0.161   
    9997  19917    -0.095     0.004     0.330     0.312    -0.143     -0.339   
    9998  19921       NaN       NaN       NaN       NaN       NaN        NaN   
    9999  19923    -0.003    -0.019    -0.015    -0.016     0.118      0.161   
    
          L0_S0_F12  L0_S0_F14  L0_S0_F16    ...     L3_S50_F4245  L3_S50_F4247  \
    0        -0.015     -0.032      0.020    ...              NaN           NaN   
    1           NaN        NaN        NaN    ...              NaN           NaN   
    2        -0.015     -0.072     -0.225    ...              NaN           NaN   
    3         0.022      0.128     -0.026    ...              NaN           NaN   
    4         0.030      0.168     -0.169    ...              NaN           NaN   
    5         0.022      0.088     -0.005    ...              NaN           NaN   
    6           NaN        NaN        NaN    ...              NaN           NaN   
    7           NaN        NaN        NaN    ...              NaN           NaN   
    8        -0.007     -0.032     -0.082    ...              NaN           NaN   
    9           NaN        NaN        NaN    ...              NaN           NaN   
    10       -0.007     -0.072      0.209    ...              NaN           NaN   
    11        0.000      0.088     -0.199    ...              NaN           NaN   
    12        0.008      0.048     -0.031    ...              NaN           NaN   
    13       -0.015      0.048     -0.031    ...              NaN           NaN   
    14       -0.007      0.088     -0.194    ...              NaN           NaN   
    15       -0.044     -0.232     -0.179    ...              NaN           NaN   
    16          NaN        NaN        NaN    ...              NaN           NaN   
    17        0.000      0.048      0.081    ...              NaN           NaN   
    18          NaN        NaN        NaN    ...              NaN           NaN   
    19        0.008      0.088     -0.112    ...              NaN           NaN   
    20          NaN        NaN        NaN    ...              NaN           NaN   
    21          NaN        NaN        NaN    ...              NaN           NaN   
    22        0.000     -0.072      0.056    ...              NaN           NaN   
    23       -0.015     -0.032      0.112    ...              NaN           NaN   
    24          NaN        NaN        NaN    ...              NaN           NaN   
    25        0.015      0.088     -0.097    ...              NaN           NaN   
    26       -0.022     -0.112     -0.174    ...              NaN           NaN   
    27        0.052      0.248      0.163    ...              NaN           NaN   
    28          NaN        NaN        NaN    ...              NaN           NaN   
    29          NaN        NaN        NaN    ...              NaN           NaN   
    ...         ...        ...        ...    ...              ...           ...   
    9970        NaN        NaN        NaN    ...              NaN           NaN   
    9971        NaN        NaN        NaN    ...              NaN           NaN   
    9972        NaN        NaN        NaN    ...              NaN           NaN   
    9973     -0.030     -0.192      0.132    ...              NaN           NaN   
    9974        NaN        NaN        NaN    ...              NaN           NaN   
    9975        NaN        NaN        NaN    ...              NaN           NaN   
    9976      0.008      0.008      0.056    ...              NaN           NaN   
    9977      0.022      0.128     -0.072    ...              NaN           NaN   
    9978        NaN        NaN        NaN    ...              NaN           NaN   
    9979     -0.007     -0.032     -0.097    ...              NaN           NaN   
    9980        NaN        NaN        NaN    ...              NaN           NaN   
    9981     -0.015     -0.072     -0.010    ...              NaN           NaN   
    9982     -0.015     -0.112      0.112    ...              NaN           NaN   
    9983     -0.015     -0.072     -0.046    ...              NaN           NaN   
    9984      0.037      0.168     -0.189    ...              NaN           NaN   
    9985        NaN        NaN        NaN    ...              NaN           NaN   
    9986        NaN        NaN        NaN    ...              NaN           NaN   
    9987      0.015      0.128     -0.138    ...              NaN           NaN   
    9988        NaN        NaN        NaN    ...              NaN           NaN   
    9989        NaN        NaN        NaN    ...              NaN           NaN   
    9990        NaN        NaN        NaN    ...              NaN           NaN   
    9991        NaN        NaN        NaN    ...              NaN           NaN   
    9992        NaN        NaN        NaN    ...              NaN           NaN   
    9993      0.000      0.008     -0.092    ...              NaN           NaN   
    9994     -0.007     -0.032      0.153    ...              NaN           NaN   
    9995        NaN        NaN        NaN    ...              NaN           NaN   
    9996      0.030      0.128      0.056    ...              NaN           NaN   
    9997      0.037      0.128      0.061    ...              NaN           NaN   
    9998        NaN        NaN        NaN    ...              NaN           NaN   
    9999      0.000      0.008      0.020    ...              NaN           NaN   
    
          L3_S50_F4249  L3_S50_F4251  L3_S50_F4253  L3_S51_F4256  L3_S51_F4258  \
    0              NaN           NaN           NaN           NaN           NaN   
    1              NaN           NaN           NaN           NaN           NaN   
    2              NaN           NaN           NaN           NaN           NaN   
    3              NaN           NaN           NaN           NaN           NaN   
    4              NaN           NaN           NaN           NaN           NaN   
    5              NaN           NaN           NaN           NaN           NaN   
    6              NaN           NaN           NaN           NaN           NaN   
    7              NaN           NaN           NaN           NaN           NaN   
    8              NaN           NaN           NaN           NaN           NaN   
    9              NaN           NaN           NaN           NaN           NaN   
    10             NaN           NaN           NaN           NaN           NaN   
    11             NaN           NaN           NaN           NaN           NaN   
    12             NaN           NaN           NaN           NaN           NaN   
    13             NaN           NaN           NaN           NaN           NaN   
    14             NaN           NaN           NaN           NaN           NaN   
    15             NaN           NaN           NaN           NaN           NaN   
    16             NaN           NaN           NaN           NaN           NaN   
    17             NaN           NaN           NaN           NaN           NaN   
    18             NaN           NaN           NaN           NaN           NaN   
    19             NaN           NaN           NaN           NaN           NaN   
    20             NaN           NaN           NaN           NaN           NaN   
    21             NaN           NaN           NaN           NaN           NaN   
    22             NaN           NaN           NaN           NaN           NaN   
    23             NaN           NaN           NaN           NaN           NaN   
    24             NaN           NaN           NaN           NaN           NaN   
    25             NaN           NaN           NaN           NaN           NaN   
    26             NaN           NaN           NaN             0             0   
    27             NaN           NaN           NaN           NaN           NaN   
    28             NaN           NaN           NaN           NaN           NaN   
    29             NaN           NaN           NaN           NaN           NaN   
    ...            ...           ...           ...           ...           ...   
    9970           NaN           NaN           NaN           NaN           NaN   
    9971           NaN           NaN           NaN           NaN           NaN   
    9972           NaN           NaN           NaN           NaN           NaN   
    9973           NaN           NaN           NaN           NaN           NaN   
    9974           NaN           NaN           NaN           NaN           NaN   
    9975           NaN           NaN           NaN           NaN           NaN   
    9976           NaN           NaN           NaN           NaN           NaN   
    9977           NaN           NaN           NaN             0             0   
    9978           NaN           NaN           NaN           NaN           NaN   
    9979           NaN           NaN           NaN           NaN           NaN   
    9980           NaN           NaN           NaN           NaN           NaN   
    9981           NaN           NaN           NaN           NaN           NaN   
    9982           NaN           NaN           NaN           NaN           NaN   
    9983           NaN           NaN           NaN           NaN           NaN   
    9984           NaN           NaN           NaN           NaN           NaN   
    9985           NaN           NaN           NaN           NaN           NaN   
    9986           NaN           NaN           NaN           NaN           NaN   
    9987           NaN           NaN           NaN           NaN           NaN   
    9988           NaN           NaN           NaN           NaN           NaN   
    9989           NaN           NaN           NaN           NaN           NaN   
    9990           NaN           NaN           NaN           NaN           NaN   
    9991           NaN           NaN           NaN             0             0   
    9992           NaN           NaN           NaN           NaN           NaN   
    9993           NaN           NaN           NaN           NaN           NaN   
    9994           NaN           NaN           NaN           NaN           NaN   
    9995           NaN           NaN           NaN           NaN           NaN   
    9996           NaN           NaN           NaN           NaN           NaN   
    9997           NaN           NaN           NaN           NaN           NaN   
    9998           NaN           NaN           NaN           NaN           NaN   
    9999           NaN           NaN           NaN             0             0   
    
          L3_S51_F4260  L3_S51_F4262  Response  
    0              NaN           NaN         0  
    1              NaN           NaN         0  
    2              NaN           NaN         0  
    3              NaN           NaN         0  
    4              NaN           NaN         0  
    5              NaN           NaN         0  
    6              NaN           NaN         0  
    7              NaN           NaN         0  
    8              NaN           NaN         0  
    9              NaN           NaN         0  
    10             NaN           NaN         0  
    11             NaN           NaN         0  
    12             NaN           NaN         0  
    13             NaN           NaN         0  
    14             NaN           NaN         0  
    15             NaN           NaN         0  
    16             NaN           NaN         0  
    17             NaN           NaN         0  
    18             NaN           NaN         0  
    19             NaN           NaN         0  
    20             NaN           NaN         0  
    21             NaN           NaN         0  
    22             NaN           NaN         0  
    23             NaN           NaN         0  
    24             NaN           NaN         0  
    25             NaN           NaN         0  
    26               0             0         0  
    27             NaN           NaN         0  
    28             NaN           NaN         0  
    29             NaN           NaN         0  
    ...            ...           ...       ...  
    9970           NaN           NaN         0  
    9971           NaN           NaN         0  
    9972           NaN           NaN         0  
    9973           NaN           NaN         0  
    9974           NaN           NaN         0  
    9975           NaN           NaN         0  
    9976           NaN           NaN         0  
    9977             0             0         0  
    9978           NaN           NaN         0  
    9979           NaN           NaN         0  
    9980           NaN           NaN         0  
    9981           NaN           NaN         0  
    9982           NaN           NaN         0  
    9983           NaN           NaN         0  
    9984           NaN           NaN         0  
    9985           NaN           NaN         0  
    9986           NaN           NaN         0  
    9987           NaN           NaN         0  
    9988           NaN           NaN         0  
    9989           NaN           NaN         0  
    9990           NaN           NaN         0  
    9991             0             0         0  
    9992           NaN           NaN         0  
    9993           NaN           NaN         0  
    9994           NaN           NaN         0  
    9995           NaN           NaN         0  
    9996           NaN           NaN         0  
    9997           NaN           NaN         0  
    9998           NaN           NaN         0  
    9999             0             0         0  
    
    [10000 rows x 970 columns]>




```python
data_merge = pd.merge(train_numeric,train_date,on = 'Id')
data_merge.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>L0_S0_F0</th>
      <th>L0_S0_F2</th>
      <th>L0_S0_F4</th>
      <th>L0_S0_F6</th>
      <th>L0_S0_F8</th>
      <th>L0_S0_F10</th>
      <th>L0_S0_F12</th>
      <th>L0_S0_F14</th>
      <th>L0_S0_F16</th>
      <th>...</th>
      <th>L3_S50_D4246</th>
      <th>L3_S50_D4248</th>
      <th>L3_S50_D4250</th>
      <th>L3_S50_D4252</th>
      <th>L3_S50_D4254</th>
      <th>L3_S51_D4255</th>
      <th>L3_S51_D4257</th>
      <th>L3_S51_D4259</th>
      <th>L3_S51_D4261</th>
      <th>L3_S51_D4263</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>  4</td>
      <td> 0.030</td>
      <td>-0.034</td>
      <td>-0.197</td>
      <td>-0.179</td>
      <td> 0.118</td>
      <td> 0.116</td>
      <td>-0.015</td>
      <td>-0.032</td>
      <td> 0.020</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>  6</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>   NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>  7</td>
      <td> 0.088</td>
      <td> 0.086</td>
      <td> 0.003</td>
      <td>-0.052</td>
      <td> 0.161</td>
      <td> 0.025</td>
      <td>-0.015</td>
      <td>-0.072</td>
      <td>-0.225</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>  9</td>
      <td>-0.036</td>
      <td>-0.064</td>
      <td> 0.294</td>
      <td> 0.330</td>
      <td> 0.074</td>
      <td> 0.161</td>
      <td> 0.022</td>
      <td> 0.128</td>
      <td>-0.026</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 11</td>
      <td>-0.055</td>
      <td>-0.086</td>
      <td> 0.294</td>
      <td> 0.330</td>
      <td> 0.118</td>
      <td> 0.025</td>
      <td> 0.030</td>
      <td> 0.168</td>
      <td>-0.169</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2126 columns</p>
</div>




```python
dataclean = data_merge.dropna(axis=1,thresh = int(len(data_merge)*0.5))
dataclean = dataclean.fillna(0)
```


```python
dataclean.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>L0_S0_F0</th>
      <th>L0_S0_F2</th>
      <th>L0_S0_F4</th>
      <th>L0_S0_F6</th>
      <th>L0_S0_F8</th>
      <th>L0_S0_F10</th>
      <th>L0_S0_F12</th>
      <th>L0_S0_F14</th>
      <th>L0_S0_F16</th>
      <th>...</th>
      <th>L3_S34_D3877</th>
      <th>L3_S34_D3879</th>
      <th>L3_S34_D3881</th>
      <th>L3_S34_D3883</th>
      <th>L3_S37_D3942</th>
      <th>L3_S37_D3943</th>
      <th>L3_S37_D3945</th>
      <th>L3_S37_D3947</th>
      <th>L3_S37_D3949</th>
      <th>L3_S37_D3951</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>  4</td>
      <td> 0.030</td>
      <td>-0.034</td>
      <td>-0.197</td>
      <td>-0.179</td>
      <td> 0.118</td>
      <td> 0.116</td>
      <td>-0.015</td>
      <td>-0.032</td>
      <td> 0.020</td>
      <td>...</td>
      <td>   87.28</td>
      <td>   87.28</td>
      <td>   87.28</td>
      <td>   87.28</td>
      <td>   87.29</td>
      <td>   87.29</td>
      <td>   87.29</td>
      <td>   87.29</td>
      <td>   87.29</td>
      <td>   87.29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>  6</td>
      <td> 0.000</td>
      <td> 0.000</td>
      <td> 0.000</td>
      <td> 0.000</td>
      <td> 0.000</td>
      <td> 0.000</td>
      <td> 0.000</td>
      <td> 0.000</td>
      <td> 0.000</td>
      <td>...</td>
      <td> 1315.75</td>
      <td> 1315.75</td>
      <td> 1315.75</td>
      <td> 1315.75</td>
      <td> 1315.75</td>
      <td> 1315.75</td>
      <td> 1315.75</td>
      <td> 1315.75</td>
      <td> 1315.75</td>
      <td> 1315.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>  7</td>
      <td> 0.088</td>
      <td> 0.086</td>
      <td> 0.003</td>
      <td>-0.052</td>
      <td> 0.161</td>
      <td> 0.025</td>
      <td>-0.015</td>
      <td>-0.072</td>
      <td>-0.225</td>
      <td>...</td>
      <td> 1624.42</td>
      <td> 1624.42</td>
      <td> 1624.42</td>
      <td> 1624.42</td>
      <td> 1624.42</td>
      <td> 1624.42</td>
      <td> 1624.42</td>
      <td> 1624.42</td>
      <td> 1624.42</td>
      <td> 1624.42</td>
    </tr>
    <tr>
      <th>3</th>
      <td>  9</td>
      <td>-0.036</td>
      <td>-0.064</td>
      <td> 0.294</td>
      <td> 0.330</td>
      <td> 0.074</td>
      <td> 0.161</td>
      <td> 0.022</td>
      <td> 0.128</td>
      <td>-0.026</td>
      <td>...</td>
      <td> 1154.15</td>
      <td> 1154.15</td>
      <td> 1154.15</td>
      <td> 1154.15</td>
      <td> 1154.16</td>
      <td> 1154.16</td>
      <td> 1154.16</td>
      <td> 1154.16</td>
      <td> 1154.16</td>
      <td> 1154.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 11</td>
      <td>-0.055</td>
      <td>-0.086</td>
      <td> 0.294</td>
      <td> 0.330</td>
      <td> 0.118</td>
      <td> 0.025</td>
      <td> 0.030</td>
      <td> 0.168</td>
      <td>-0.169</td>
      <td>...</td>
      <td>  606.01</td>
      <td>  606.01</td>
      <td>  606.01</td>
      <td>  606.01</td>
      <td>  606.02</td>
      <td>  606.02</td>
      <td>  606.02</td>
      <td>  606.02</td>
      <td>  606.02</td>
      <td>  606.02</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 328 columns</p>
</div>




```python
### 1 column: 50% data whih is filled and 50 % are emtpy ###
### 1) Data imbalance : Noisy, overfit, ### 
```





```python
### label the encoder  ( aligning he labels in order) ###
 
le = preprocessing.LabelEncoder()
dataclean['Id'] = le.fit_transform(dataclean.Id)

```


```python
### Splitting my data into Training and testing  by ignoring ID column as its Identical column ###
featurelist =  list(dataclean.columns.values)
featurelist.remove('Id')
featurelist.remove('Response')
features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(dataclean[featurelist],
                                                              dataclean['Response'], test_size=0.1, random_state=42)

```

Training data
features_train  # ind columns
labels_train  # dependent columns
Testing Data
features_test # ind columns
labels_test# dependent columns


```python
### 10k -- accuracy 92 % 99 %   (on sample the accuray may be higher but when we consider total amount of data we the accuray can goes down to ) ###

### 80%   --- 89% ###
```


```python

### Naive Bayes###

```


```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB
```

    :0: FutureWarning: IPython widgets are experimental and may change in the future.



```python
naive_bayes = BernoulliNB()
naive_bayes.fit(features_train,labels_train)
```




    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)




```python
p_station = naive_bayes.predict_proba(features_test)
p_station
```




    array([[  1.30977963e-21,   1.00000000e+00],
           [  9.96157867e-01,   3.84213254e-03],
           [  9.99999998e-01,   2.25669252e-09],
           ..., 
           [  9.82437565e-01,   1.75624351e-02],
           [  9.99999555e-01,   4.45174560e-07],
           [  9.99999993e-01,   6.72477614e-09]])




```python
# 0 = Not failure, 1  = Failure
pred = naive_bayes.predict(features_test)
pred
```




    array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64)




```python
labels_test.shape
```




    (1000L,)




```python
pred.shape
```




    (1000L,)




```python
accuracy = accuracy_score(labels_test,pred)
accuracy
```




    0.92900000000000005




```python
 
### Random Forest Classifier ###

```


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
clf = RandomForestClassifier(100, max_depth = 20, n_jobs =3)
```


```python
clf
```




    RandomForestClassifier(bootstrap=True, compute_importances=None,
                criterion='gini', max_depth=20, max_features='auto',
                max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
                min_samples_split=2, n_estimators=100, n_jobs=3,
                oob_score=False, random_state=None, verbose=0)




```python
clf.fit(features_train,labels_train)
```




    RandomForestClassifier(bootstrap=True, compute_importances=None,
                criterion='gini', max_depth=20, max_features='auto',
                max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
                min_samples_split=2, n_estimators=100, n_jobs=3,
                oob_score=False, random_state=None, verbose=0)




```python
accuracy = accuracy_score(labels_test,pred)
accuracy
```




    0.99399999999999999




```python
pred = clf.predict(features_test)
pred
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64)




```python

### Grid Search###

```


```python
param_grid= {  "criterion" : ['gini','entropy'],
                 "min_samples_split": [2,4,5,6,7,8,9,10],
                 "max_depth" : [None,2,4],
                 "min_samples_leaf" :[1,3,5,6,7,8,10],
                 'n_estimators':[20,30,50,70],
                     'n_jobs' :[-1]
             
}
```


```python
modeloptimal = grid_search.GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, scoring='f1', cv=5)
modeloptimal
```




    GridSearchCV(cv=5,
           estimator=RandomForestClassifier(bootstrap=True, compute_importances=None,
                criterion='gini', max_depth=None, max_features='auto',
                max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
                min_samples_split=2, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0),
           fit_params={}, iid=True, loss_func=None, n_jobs=1,
           param_grid={'n_jobs': [-1], 'min_samples_leaf': [1, 3, 5, 6, 7, 8, 10], 'n_estimators': [20, 30, 50, 70], 'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 4, 5, 6, 7, 8, 9, 10], 'max_depth': [None, 2, 4]},
           pre_dispatch='2*n_jobs', refit=True, score_func=None, scoring='f1',
           verbose=0)




```python
modeloptimal.fit(features_train, labels_train)
```


```python
### Using this we can find the best acuracy model from the above parameter estimators ###
clf = modeloptimal.best_estimator_
clf
```


```python
pred = clf.predict(features_test)
pred
```


```python
accuracy = accuracy_score(labels_test)
accuracy
```


```python
### Extra Tree Classifier ###
```


```python
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
```


```python
clf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1,min_samples_leaf= 10, verbose = 1)
```


```python
clf.fit(features_train,labels_train)
```

    [Parallel(n_jobs=4)]: Done   1 out of   4 | elapsed:    0.8s remaining:    2.5s
    [Parallel(n_jobs=4)]: Done   4 out of   4 | elapsed:    0.8s finished





    ExtraTreesClassifier(bootstrap=False, compute_importances=None,
               criterion='gini', max_depth=None, max_features='auto',
               max_leaf_nodes=None, min_density=None, min_samples_leaf=10,
               min_samples_split=2, n_estimators=50, n_jobs=-1,
               oob_score=False, random_state=None, verbose=1)




```python
pred = clf.predict(features_test)
pred
```

    [Parallel(n_jobs=4)]: Done   1 out of   4 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=4)]: Done   4 out of   4 | elapsed:    0.0s finished





    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64)




```python
accuracy = accuracy_score(labels_test, pred)
accuracy
```




    0.996




```python
### xgboost ###

```


```python
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
```


```python
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1) 
```


```python
optimized_GBM.fit(features_train, labels_train)
```


```python
optimized_GBM.grid_scores_
```


```python
cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth': 3, 'min_child_weight': 1}


optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
optimized_GBM.fit(features_train, labels_train)
```


```python
optimized_GBM.grid_scores_
```

There are a few other parameters we could tune in theory to squeeze out further performance, but this is a good enough starting point.

To increase the performance of XGBoost’s speed through many iterations of the training set, and since we are using only XGBoost’s API and not sklearn’s anymore, we can create a DMatrix. This sorts the data initially to optimize for XGBoost when it builds trees, making the algorithm more efficient. This is especially helpful when you have a very large number of training examples. To create a DMatrix:


```python
xgdmat = xgb.DMatrix(features_train, labels_train) # Create our DMatrix to make XGBoost more efficient
```


```python
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1} 
# Grid Search CV optimized settings

cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 3000, nfold = 5,
                metrics = ['error'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error
```

We can look at our CV results to see how accurate we were with these settings. The output is automatically saved into a pandas dataframe for us.


```python
cv_xgb.tail(5)
```


```python
Now that we have our best settings, let’s create this as an XGBoost object model that we can reference later.
```


```python
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1} 

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 432)
```


```python
%matplotlib inline
import seaborn as sns
sns.set(font_scale = 1.5)
```


```python
xgb.plot_importance(final_gb)
```


```python
importances = final_gb.get_fscore()
importances
```


```python
importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)
importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')
```

Analyzing Performance on Test Data

The model has now been tuned using cross-validation grid search through the sklearn API and early stopping through the built-in XGBoost API. Now, we can see how it finally performs on the test set. Does it match our CV performance? First, create another DMatrix (this time for the test data).


```python
testdmat = xgb.DMatrix(features_test, labels_test)
```


```python
from sklearn.metrics import accuracy_score
y_pred = final_gb.predict(testdmat) # Predict using our testdmat
y_pred
```


```python
accuracy_score(y_pred, labels_test), 1-accuracy_score(y_pred, labels_test)
```


```python
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
```


```python
### test set ###
test_numeric = pd.read_csv(Datapath+'test_numeric.csv')
test_date = pd.read_csv(Datapath+'test_date.csv')
data_merge = pd.merge(test_numeric, test_date, on='Id',suffixes=('num', 'date'))
### test set ###

```


```python
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

```


```python
makesubmit(clf,data_merge,featurelist,output="submit.csv")
```

In the first step, we import standard libraries and fix the most essential features as suggested by an XGB


```python
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns

feature_names = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114']
```

We determine the indices of the most important features. After that the training data is loaded


```python
numeric_cols = pd.read_csv(DataPath+"train_numeric.csv", nrows = 10000).columns.values
imp_idxs = [np.argwhere(feature_name == numeric_cols)[0][0] for feature_name in feature_names]
train = pd.read_csv(DataPath+"train_numeric.csv", 
                index_col = 0, header = 0, usecols = [0, len(numeric_cols) - 1] + imp_idxs)
train = train[feature_names + ['Response']]
```

The data is split into positive and negative samples.


```python
 X_neg, X_pos = train[train['Response'] == 0].iloc[:, :-1], train[train['Response']==1].iloc[:, :-1]
```

# Univariate characteristics

In order to understand better the predictive power of single features, we compare the univariate distributions of the most important features. First, we divide the train data into batches column-wise to prepare the data for plotting.


```python
BATCH_SIZE = 5
train_batch =[pd.melt(train[train.columns[batch: batch + BATCH_SIZE].append(np.array(['Response']))], 
                      id_vars = 'Response', value_vars = feature_names[batch: batch + BATCH_SIZE])
              for batch in list(range(0, train.shape[1] - 1, BATCH_SIZE))]
```

After this split, we can now draw violin plots. Due to memory reasons, we have to split the presentation into several cells. For many of the distributions there is no clear difference between the positive and negative samples.


```python
FIGSIZE = (12,16)
_, axs = plt.subplots(len(train_batch), figsize = FIGSIZE)
plt.suptitle('Univariate distributions')
for data, ax in zip(train_batch, axs):
    sns.violinplot(x = 'variable',  y = 'value', hue = 'Response', data = data, ax = ax, split =True)
```


![png](output_86_0.png)


# Correlation structure

In the previous section we have seen differences between negative and positive samples for univariate characteristics. We go down the rabbit hole a little further and analyze covariances for the negative and positive samples separately.


```python
FIGSIZE = (13,4)
_, (ax1, ax2) = plt.subplots(1,2, figsize = FIGSIZE)
MIN_PERIODS = 100

triang_mask = np.zeros((X_pos.shape[1], X_pos.shape[1]))
triang_mask[np.triu_indices_from(triang_mask)] = True

ax1.set_title('Negative Class')
sns.heatmap(X_neg.corr(min_periods = MIN_PERIODS), mask = triang_mask, square=True,  ax = ax1)

ax2.set_title('Positive Class')
sns.heatmap(X_pos.corr(min_periods = MIN_PERIODS), mask = triang_mask, square=True,  ax = ax2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1cbc71d0>




![png](output_89_1.png)


The difference between the two matrices is sparse except for three specific feature combinations.


```python
sns.heatmap(X_pos.corr(min_periods = MIN_PERIODS) -X_neg.corr(min_periods = MIN_PERIODS), 
             mask = triang_mask, square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21ba3ef0>




![png](output_91_1.png)


Finally, as in the univariate case, we analyze correlations between missing values in different features.


```python
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
```




    <matplotlib.axes._subplots.AxesSubplot at 0x36c6a4a8>




![png](output_93_1.png)


For the difference of the missing-value correlation matrices, a striking pattern emerges. A further and more systematic analysis of such missing-value patterns has the potential to beget powerful features.


```python
sns.heatmap(nan_neg.corr() - nan_pos.corr(), mask = triang_mask, square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ca66630>




![png](output_95_1.png)


### Hope you have enjoyed. Have a great day! ###
