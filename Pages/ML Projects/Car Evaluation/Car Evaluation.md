<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Car-Evaluation" data-toc-modified-id="Car-Evaluation-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Car Evaluation</a></span></li><li><span><a href="#Importing-the-necessary-libraries" data-toc-modified-id="Importing-the-necessary-libraries-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Importing the necessary libraries</a></span></li><li><span><a href="#Loading-the-data-set" data-toc-modified-id="Loading-the-data-set-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Loading the data set</a></span></li><li><span><a href="#Dataset-Information" data-toc-modified-id="Dataset-Information-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Dataset Information</a></span></li><li><span><a href="#Attribute-Information" data-toc-modified-id="Attribute-Information-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Attribute Information</a></span></li><li><span><a href="#Data-Preprocessing" data-toc-modified-id="Data-Preprocessing-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Data Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Converting-to-Dataframe-Format" data-toc-modified-id="Converting-to-Dataframe-Format-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Converting to Dataframe Format</a></span></li><li><span><a href="#Converting-columns-to-categorical-data-types" data-toc-modified-id="Converting-columns-to-categorical-data-types-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Converting columns to categorical data types</a></span></li></ul></li><li><span><a href="#Univariate-Analysis-:-Categorical-Variables" data-toc-modified-id="Univariate-Analysis-:-Categorical-Variables-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Univariate Analysis : Categorical Variables</a></span><ul class="toc-item"><li><span><a href="#Overview-of-all-the-categorical-variables" data-toc-modified-id="Overview-of-all-the-categorical-variables-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Overview of all the categorical variables</a></span></li><li><span><a href="#Price-vs-Class-heatmap" data-toc-modified-id="Price-vs-Class-heatmap-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Price vs Class heatmap</a></span><ul class="toc-item"><li><span><a href="#Buying-Price-vs-Class" data-toc-modified-id="Buying-Price-vs-Class-7.2.1"><span class="toc-item-num">7.2.1&nbsp;&nbsp;</span>Buying Price vs Class</a></span></li><li><span><a href="#Maintenance-vs-Class" data-toc-modified-id="Maintenance-vs-Class-7.2.2"><span class="toc-item-num">7.2.2&nbsp;&nbsp;</span>Maintenance vs Class</a></span></li><li><span><a href="#Buying-vs-Maintenance" data-toc-modified-id="Buying-vs-Maintenance-7.2.3"><span class="toc-item-num">7.2.3&nbsp;&nbsp;</span>Buying vs Maintenance</a></span></li><li><span><a href="#Price-(Buying-Price-and-Maintenance)-vs-Class" data-toc-modified-id="Price-(Buying-Price-and-Maintenance)-vs-Class-7.2.4"><span class="toc-item-num">7.2.4&nbsp;&nbsp;</span>Price (Buying Price and Maintenance) vs Class</a></span></li></ul></li><li><span><a href="#Comfort-vs-Class-Heatmaps" data-toc-modified-id="Comfort-vs-Class-Heatmaps-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Comfort vs Class Heatmaps</a></span></li></ul></li><li><span><a href="#Defining-the-Problem" data-toc-modified-id="Defining-the-Problem-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Defining the Problem</a></span></li><li><span><a href="#Approach-for-imbalanced-datasets" data-toc-modified-id="Approach-for-imbalanced-datasets-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Approach for imbalanced datasets</a></span></li><li><span><a href="#Applying-Machine-Learning-Algorithms-for-classification" data-toc-modified-id="Applying-Machine-Learning-Algorithms-for-classification-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Applying Machine Learning Algorithms for classification</a></span><ul class="toc-item"><li><span><a href="#Training-and-testing-the-data" data-toc-modified-id="Training-and-testing-the-data-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Training and testing the data</a></span></li><li><span><a href="#Preparing-the-data" data-toc-modified-id="Preparing-the-data-10.2"><span class="toc-item-num">10.2&nbsp;&nbsp;</span>Preparing the data</a></span></li><li><span><a href="#Feature-Selection-(Optional)" data-toc-modified-id="Feature-Selection-(Optional)-10.3"><span class="toc-item-num">10.3&nbsp;&nbsp;</span>Feature Selection (Optional)</a></span><ul class="toc-item"><li><span><a href="#example-of-mutual-information-feature-selection-for-categorical-data" data-toc-modified-id="example-of-mutual-information-feature-selection-for-categorical-data-10.3.1"><span class="toc-item-num">10.3.1&nbsp;&nbsp;</span>example of mutual information feature selection for categorical data</a></span></li><li><span><a href="#example-of-chi-squared-feature-selection-for-categorical-data" data-toc-modified-id="example-of-chi-squared-feature-selection-for-categorical-data-10.3.2"><span class="toc-item-num">10.3.2&nbsp;&nbsp;</span>example of chi squared feature selection for categorical data</a></span></li></ul></li><li><span><a href="#Popular-algorithms-for-multi-class-classification-include:" data-toc-modified-id="Popular-algorithms-for-multi-class-classification-include:-10.4"><span class="toc-item-num">10.4&nbsp;&nbsp;</span>Popular algorithms for multi-class classification include:</a></span></li><li><span><a href="#Logistic-Regression-Classifier" data-toc-modified-id="Logistic-Regression-Classifier-10.5"><span class="toc-item-num">10.5&nbsp;&nbsp;</span>Logistic Regression Classifier</a></span></li><li><span><a href="#Decision-Tree-Classifier" data-toc-modified-id="Decision-Tree-Classifier-10.6"><span class="toc-item-num">10.6&nbsp;&nbsp;</span>Decision Tree Classifier</a></span></li><li><span><a href="#K-Nearest-Neighbors-Classifier" data-toc-modified-id="K-Nearest-Neighbors-Classifier-10.7"><span class="toc-item-num">10.7&nbsp;&nbsp;</span>K Nearest Neighbors Classifier</a></span></li><li><span><a href="#Naive-Bayes-Classifier" data-toc-modified-id="Naive-Bayes-Classifier-10.8"><span class="toc-item-num">10.8&nbsp;&nbsp;</span>Naive Bayes Classifier</a></span></li><li><span><a href="#Random-Forest-Classifier" data-toc-modified-id="Random-Forest-Classifier-10.9"><span class="toc-item-num">10.9&nbsp;&nbsp;</span>Random Forest Classifier</a></span></li><li><span><a href="#Linear-SVC-Classifier" data-toc-modified-id="Linear-SVC-Classifier-10.10"><span class="toc-item-num">10.10&nbsp;&nbsp;</span>Linear SVC Classifier</a></span></li><li><span><a href="#Gradient-Boosting" data-toc-modified-id="Gradient-Boosting-10.11"><span class="toc-item-num">10.11&nbsp;&nbsp;</span>Gradient Boosting</a></span></li><li><span><a href="#Listing-the-performance-from-all-the-models" data-toc-modified-id="Listing-the-performance-from-all-the-models-10.12"><span class="toc-item-num">10.12&nbsp;&nbsp;</span>Listing the performance from all the models</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Conclusion</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Resources-to-follow-for-imbalanced-learning" data-toc-modified-id="Resources-to-follow-for-imbalanced-learning-11.0.1"><span class="toc-item-num">11.0.1&nbsp;&nbsp;</span>Resources to follow for imbalanced learning</a></span><ul class="toc-item"><li><span><a href="#Datasets" data-toc-modified-id="Datasets-11.0.1.1"><span class="toc-item-num">11.0.1.1&nbsp;&nbsp;</span>Datasets</a></span></li><li><span><a href="#Real-life-scenarios-with-data-imbalance" data-toc-modified-id="Real-life-scenarios-with-data-imbalance-11.0.1.2"><span class="toc-item-num">11.0.1.2&nbsp;&nbsp;</span>Real life scenarios with data imbalance</a></span></li></ul></li></ul></li></ul></li></ul></div>

# Car Evaluation 

This dataset has been downloaded from  UC Irvine Machine Learning Repository.  
<https://archive.ics.uci.edu/ml/datasets/Car+Evaluation>  
<https://www.kaggle.com/mykeysid10/car-evaluation>

This dataset is regarding evaluation of cars.  
The target variable/label is car acceptability and has four categories : unacceptable, acceptable, good and very good.


The input attributes fall under two broad categories - Price and Technical Characteristics.  
Under Price, the attributes are buying price and maintenance price.  
Under Technical characteristics, the attributes are doors, persons, size of luggage boot and safety.  

We have identified : this is an imbalanced dataset with skewed class (output category/label) proportions. 
  
**The objective is here to build a model to give multiclass classifier model based on the input attributes.**  
  
>**Summary of Key information**

    Number of Instances/training examples          : 1728  
    Number of Instances with missing attributes    :    0  
    Number of qualified Instances/training examples :   0
    
    Number of Input Attributes                     :  6
    Number of categorical attributes               :  6
    Number of numerical attributes                 :  0
    
    Target Attribute Type                          : Multi class label
    Target Class distribution                      : 70%:22%:3.9%:3.7%
    Problem Identification                         : Multiclass Classification with imbalanced data set
    


# Importing the necessary libraries


```python
# Data Wrangling, inspection 
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

# Data preprocessing 
import category_encoders as ce 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OrdinalEncoder

# sklearn ml models
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

# Evaluation metrics
from sklearn.metrics import recall_score, precision_score, \
accuracy_score, plot_confusion_matrix, classification_report, f1_score
```

# Loading the data set


```python
pathname = "/Users/bhaskarroy/BHASKAR FILES/BHASKAR CAREER/Career/Skills/Data Science/Practise/Python/UCI Machine Learning Repository/car"
path0 = "/car.c45-names"
path1 = "/car.data"
path2 = "/car.names"


pathdata = pathname + path1
pathcolname = pathname + path0
pathdatadesc = pathname + path2
```

# Dataset Information


```python
with open(pathdatadesc) as f:
    print(f.read())
```

    1. Title: Car Evaluation Database
    
    2. Sources:
       (a) Creator: Marko Bohanec
       (b) Donors: Marko Bohanec   (marko.bohanec@ijs.si)
                   Blaz Zupan      (blaz.zupan@ijs.si)
       (c) Date: June, 1997
    
    3. Past Usage:
    
       The hierarchical decision model, from which this dataset is
       derived, was first presented in 
    
       M. Bohanec and V. Rajkovic: Knowledge acquisition and explanation for
       multi-attribute decision making. In 8th Intl Workshop on Expert
       Systems and their Applications, Avignon, France. pages 59-78, 1988.
    
       Within machine-learning, this dataset was used for the evaluation
       of HINT (Hierarchy INduction Tool), which was proved to be able to
       completely reconstruct the original hierarchical model. This,
       together with a comparison with C4.5, is presented in
    
       B. Zupan, M. Bohanec, I. Bratko, J. Demsar: Machine learning by
       function decomposition. ICML-97, Nashville, TN. 1997 (to appear)
    
    4. Relevant Information Paragraph:
    
       Car Evaluation Database was derived from a simple hierarchical
       decision model originally developed for the demonstration of DEX
       (M. Bohanec, V. Rajkovic: Expert system for decision
       making. Sistemica 1(1), pp. 145-157, 1990.). The model evaluates
       cars according to the following concept structure:
    
       CAR                      car acceptability
       . PRICE                  overall price
       . . buying               buying price
       . . maint                price of the maintenance
       . TECH                   technical characteristics
       . . COMFORT              comfort
       . . . doors              number of doors
       . . . persons            capacity in terms of persons to carry
       . . . lug_boot           the size of luggage boot
       . . safety               estimated safety of the car
    
       Input attributes are printed in lowercase. Besides the target
       concept (CAR), the model includes three intermediate concepts:
       PRICE, TECH, COMFORT. Every concept is in the original model
       related to its lower level descendants by a set of examples (for
       these examples sets see http://www-ai.ijs.si/BlazZupan/car.html).
    
       The Car Evaluation Database contains examples with the structural
       information removed, i.e., directly relates CAR to the six input
       attributes: buying, maint, doors, persons, lug_boot, safety.
    
       Because of known underlying concept structure, this database may be
       particularly useful for testing constructive induction and
       structure discovery methods.
    
    5. Number of Instances: 1728
       (instances completely cover the attribute space)
    
    6. Number of Attributes: 6
    
    7. Attribute Values:
    
       buying       v-high, high, med, low
       maint        v-high, high, med, low
       doors        2, 3, 4, 5-more
       persons      2, 4, more
       lug_boot     small, med, big
       safety       low, med, high
    
    8. Missing Attribute Values: none
    
    9. Class Distribution (number of instances per class)
    
       class      N          N[%]
       -----------------------------
       unacc     1210     (70.023 %) 
       acc        384     (22.222 %) 
       good        69     ( 3.993 %) 
       v-good      65     ( 3.762 %) 
    


# Attribute Information


```python
with open(pathcolname) as f:
    print(f.read())
```

    | names file (C4.5 format) for car evaluation domain
    
    | class values
    
    unacc, acc, good, vgood
    
    | attributes
    
    buying:   vhigh, high, med, low.
    maint:    vhigh, high, med, low.
    doors:    2, 3, 4, 5more.
    persons:  2, 4, more.
    lug_boot: small, med, big.
    safety:   low, med, high.
    


# Data Preprocessing

We will prepare the data for :
- Exploratory Data analysis (EDA) and 
- for model building

Following actions were undertaken:

- Converting to Dataframe Format
- Inspect if any missing values present
- Handling Missing values : There are no missing values. Hnece, entire data can be considered for model building
- Processing Categorical Attributes : categorical attributes have been converted to categorical data type for EDA.
- Processing Continous Attributes : not applicable as both the input and output attributes are categorical.

## Converting to Dataframe Format


```python
colnames = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv(pathdata, names = colnames, index_col = False)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying</th>
      <th>maint</th>
      <th>doors</th>
      <th>persons</th>
      <th>lug_boot</th>
      <th>safety</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>low</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>med</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>high</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>low</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>med</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1723</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>more</td>
      <td>med</td>
      <td>med</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1724</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>more</td>
      <td>med</td>
      <td>high</td>
      <td>vgood</td>
    </tr>
    <tr>
      <th>1725</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>more</td>
      <td>big</td>
      <td>low</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>1726</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>more</td>
      <td>big</td>
      <td>med</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1727</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>more</td>
      <td>big</td>
      <td>high</td>
      <td>vgood</td>
    </tr>
  </tbody>
</table>
<p>1728 rows × 7 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1728 entries, 0 to 1727
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   buying    1728 non-null   object
     1   maint     1728 non-null   object
     2   doors     1728 non-null   object
     3   persons   1728 non-null   object
     4   lug_boot  1728 non-null   object
     5   safety    1728 non-null   object
     6   class     1728 non-null   object
    dtypes: object(7)
    memory usage: 94.6+ KB



```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying</th>
      <th>maint</th>
      <th>doors</th>
      <th>persons</th>
      <th>lug_boot</th>
      <th>safety</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1728</td>
      <td>1728</td>
      <td>1728</td>
      <td>1728</td>
      <td>1728</td>
      <td>1728</td>
      <td>1728</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>top</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>4</td>
      <td>big</td>
      <td>low</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>432</td>
      <td>432</td>
      <td>432</td>
      <td>576</td>
      <td>576</td>
      <td>576</td>
      <td>1210</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dtypes
```




    buying      object
    maint       object
    doors       object
    persons     object
    lug_boot    object
    safety      object
    class       object
    dtype: object




```python
# Inspect if any missing values present
data.isnull().sum()
```




    buying      0
    maint       0
    doors       0
    persons     0
    lug_boot    0
    safety      0
    class       0
    dtype: int64



## Converting columns to categorical data types


```python
for i in data.columns :
    print(f'{i} : {data[i].unique().tolist()}')
```

    buying : ['vhigh', 'high', 'med', 'low']
    maint : ['vhigh', 'high', 'med', 'low']
    doors : ['2', '3', '4', '5more']
    persons : ['2', '4', 'more']
    lug_boot : ['small', 'med', 'big']
    safety : ['low', 'med', 'high']
    class : ['unacc', 'acc', 'vgood', 'good']



```python
cat_vars = data.columns
```


```python
catdict = {
  "buying":   ['low','med','high', 'vhigh' ],
    "maint":    ['low','med','high', 'vhigh' ],
    "doors":    ['2', '3', '4', '5more'],
    "persons" :  ['2', '4', 'more'],
    "lug_boot" : ['small', 'med', 'big'],
    "safety" :  ['low', 'med', 'high'],
    "class":['unacc', 'acc','good', 'v-good']
}

```


```python
for i in cat_vars :
    data[i] = pd.Categorical(data[i], 
                             categories=catdict[i], ordered=True)
```


```python
data.dtypes
```




    buying      category
    maint       category
    doors       category
    persons     category
    lug_boot    category
    safety      category
    class       category
    dtype: object




```python
def show(data):
  for i in data.columns[0:]:
    print("Feature: {} with {} Levels".format(i,data[i].unique()))

show(data)
```

    Feature: buying with ['vhigh', 'high', 'med', 'low']
    Categories (4, object): ['low' < 'med' < 'high' < 'vhigh'] Levels
    Feature: maint with ['vhigh', 'high', 'med', 'low']
    Categories (4, object): ['low' < 'med' < 'high' < 'vhigh'] Levels
    Feature: doors with ['2', '3', '4', '5more']
    Categories (4, object): ['2' < '3' < '4' < '5more'] Levels
    Feature: persons with ['2', '4', 'more']
    Categories (3, object): ['2' < '4' < 'more'] Levels
    Feature: lug_boot with ['small', 'med', 'big']
    Categories (3, object): ['small' < 'med' < 'big'] Levels
    Feature: safety with ['low', 'med', 'high']
    Categories (3, object): ['low' < 'med' < 'high'] Levels
    Feature: class with ['unacc', 'acc', NaN, 'good']
    Categories (3, object): ['unacc' < 'acc' < 'good'] Levels



```python
data.isnull().sum()
```




    buying       0
    maint        0
    doors        0
    persons      0
    lug_boot     0
    safety       0
    class       65
    dtype: int64



# Univariate Analysis : Categorical Variables


```python
#Accessing colors from external library Palettable
from palettable.cartocolors.qualitative import Bold_10 
colors = Bold_10.mpl_colors

#colors = plt.cm.Dark2(range(15))
#colors = plt.cm.tab20(range(15))

#.colors attribute for listed colormaps
#colors = plt.cm.tab10.colors 
#colors = plt.cm.Paired.colors
```


```python
# custom function for easy and efficient analysis of categorical univariate
def UVA_category(data_frame, var_group = [], **kargs):

  '''
  Stands for Univariate_Analysis_categorical
  takes a group of variables (category) and plot/print all the value_counts and horizontal barplot.

  - data_frame : The Dataframe
  - var_group : The list of column names for univariate plots need to be plotted

  The keyword arguments are as follows :
  - col_count : The number of columns in the plot layout. Default value set is 2.
  For instance, if there are are 4# columns in var_group, then 4# univariate plots will be plotted in 2X2 layout.
  - colwidth : width of each plot
  - rowheight : height of each plot
  - normalize : Whether to present absolute values or percentage
  - sort_by : Whether to sort the bars by descending order of values

  - axlabel_fntsize : Fontsize of x axis and y axis labels
  - infofntsize : Fontsize in the info of unique value counts
  - axlabel_fntsize : fontsize of axis labels
  - axticklabel_fntsize : fontsize of axis tick labels
  - infofntfamily : Font family of info of unique value counts.
  Choose font family belonging to Monospace for multiline alignment.
  Some choices are : 'Consolas', 'Courier','Courier New', 'Lucida Sans Typewriter','Lucidatypewriter','Andale Mono'
  https://www.tutorialbrain.com/css_tutorial/css_font_family_list/
  - max_val_counts : Number of unique values for which count should be displayed
  - nspaces : Length of each line for the multiline strings in the info area for value_counts
  - ncountspaces : Length allocated to the count value for the unique values in the info area
  - show_percentage : Whether to show percentage of total for each unique value count
  Also check link for formatting syntax : https://pyformat.info/#number
  '''

  import textwrap
  data = data_frame.copy(deep = True)
  # Using dictionary with default values of keywrod arguments
  params_plot = dict(colcount = 2, colwidth = 7, rowheight = 4, normalize = False, sort_by = "Values")
  params_fontsize =  dict(axlabel_fntsize = 10,axticklabel_fntsize = 8, infofntsize = 10)
  params_fontfamily = dict(infofntfamily = 'Andale Mono')
  params_max_val_counts = dict(max_val_counts = 10)
  params_infospaces = dict(nspaces = 10, ncountspaces = 4)
  params_show_percentage = dict(show_percentage = True)



  # Updating the dictionary with parameter values passed while calling the function
  params_plot.update((k, v) for k, v in kargs.items() if k in params_plot)
  params_fontsize.update((k, v) for k, v in kargs.items() if k in params_fontsize)
  params_fontfamily.update((k, v) for k, v in kargs.items() if k in params_fontfamily)
  params_max_val_counts.update((k, v) for k, v in kargs.items() if k in params_max_val_counts)
  params_infospaces.update((k, v) for k, v in kargs.items() if k in params_infospaces)
  params_show_percentage.update((k, v) for k, v in kargs.items() if k in params_show_percentage)

  #params = dict(**params_plot, **params_fontsize)

  # Initialising all the possible keyword arguments of doc string with updated values
  colcount = params_plot['colcount']
  colheight = params_plot['colheight']
  rowheight = params_plot['rowheight']
  normalize = params_plot['normalize']
  sort_by = params_plot['sort_by']

  axlabel_fntsize = params_fontsize['axlabel_fntsize']
  axticklabel_fntsize = params_fontsize['axticklabel_fntsize']
  infofntsize = params_fontsize['infofntsize']
  infofntfamily = params_fontfamily['infofntfamily']
  max_val_counts =  params_max_val_counts['max_val_counts']
  nspaces = params_infospaces['nspaces']
  ncountspaces = params_infospaces['ncountspaces']
  show_percentage = params_show_percentage['show_percentage']

  if len(var_group) == 0:
        var_group = df.select_dtypes(exclude = ['number']).columns.to_list()

  import matplotlib.pyplot as plt
  plt.rcdefaults()
  # setting figure_size
  size = len(var_group)
  #rowcount = 1
  #colcount = size//rowcount+(size%rowcount != 0)*1


  colcount = colcount
  #print(colcount)
  rowcount = size//colcount+(size%colcount != 0)*1

  plt.figure(figsize = (colwidth*colcount,rowheight*rowcount), dpi = 150)


  # Converting the filtered columns as categorical
  for i in var_group:
        #data[i] = data[i].astype('category')
        data[i] = pd.Categorical(data[i])


  # for every variable
  for j,i in enumerate(var_group):
    #print('{} : {}'.format(j,i))
    norm_count = data[i].value_counts(normalize = normalize).sort_index()
    n_uni = data[i].nunique()

    if sort_by == "Values":
        norm_count = data[i].value_counts(normalize = normalize).sort_values(ascending = False)
        n_uni = data[i].nunique()


  #Plotting the variable with every information
    plt.subplot(rowcount,colcount,j+1)
    sns.barplot(x = norm_count, y = norm_count.index , order = norm_count.index)

    if normalize == False :
        plt.xlabel('count', fontsize = axlabel_fntsize )
    else :
        plt.xlabel('fraction/percent', fontsize = axlabel_fntsize )
    plt.ylabel('{}'.format(i), fontsize = axlabel_fntsize )

    ax = plt.gca()

    # textwrapping
    ax.set_yticklabels([textwrap.fill(str(e), 20) for e in norm_count.index], fontsize = axticklabel_fntsize)

    #print(n_uni)
    #print(type(norm_count.round(2)))

    # Functions to convert the pairing of unique values and value_counts into text string
    # Function to break a word into multiline string of fixed width per line
    def paddingString(word, nspaces = 20):
        i = len(word)//nspaces \
            +(len(word)%nspaces > 0)*(len(word)//nspaces > 0)*1 \
            + (len(word)//nspaces == 0)*1
        strA = ""
        for j in range(i-1):
            strA = strA+'\n'*(len(strA)>0)+ word[j*nspaces:(j+1)*nspaces]

        # insert appropriate number of white spaces
        strA = strA + '\n'*(len(strA)>0)*(i>1)+word[(i-1)*nspaces:] \
               + " "*(nspaces-len(word)%nspaces)*(len(word)%nspaces > 0)
        return strA

    # Function to convert Pandas series into multi line strings
    def create_string_for_plot(ser, nspaces = nspaces, ncountspaces = ncountspaces, \
                              show_percentage =  show_percentage):
        '''
        - nspaces : Length of each line for the multiline strings in the info area for value_counts
        - ncountspaces : Length allocated to the count value for the unique values in the info area
        - show_percentage : Whether to show percentage of total for each unique value count
        Also check link for formatting syntax : https://pyformat.info/#number
        '''
        str_text = ""
        for index, value in ser.items():
            str_tmp = paddingString(str(index), nspaces)+ " : " \
                      + " "*(ncountspaces-len(str(value)))*(len(str(value))<= ncountspaces) \
                      + str(value) \
                      + (" | " + "{:4.1f}%".format(value/ser.sum()*100))*show_percentage


            str_text = str_text + '\n'*(len(str_text)>0) + str_tmp
        return str_text

    #print(create_string_for_plot(norm_count.round(2)))

    #Ensuring a maximum of 10 unique values displayed
    if norm_count.round(2).size <= max_val_counts:
        text = '{}\nn_uniques = {}\nvalue counts\n{}' \
                .format(i, n_uni,create_string_for_plot(norm_count.round(2)))
        ax.annotate(text = text,
                    xy = (1.1, 1), xycoords = ax.transAxes,
                    ha = 'left', va = 'top', fontsize = infofntsize, fontfamily = infofntfamily)
    else :
        text = '{}\nn_uniques = {}\nvalue counts of top 10\n{}' \
                .format(i, n_uni,create_string_for_plot(norm_count.round(2)[0:max_val_counts]))
        ax.annotate(text = text,
                    xy = (1.1, 1), xycoords = ax.transAxes,
                    ha = 'left', va = 'top', fontsize = infofntsize, fontfamily = infofntfamily)


    plt.gcf().tight_layout()

```

## Overview of all the categorical variables 


```python
from eda import eda_overview
eda_overview.UVA_category(data, data.columns,
                          rowheight = 3, normalize = False);
```

    Categorical features : Index(['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'], dtype='object') 
    



![png](Car%20Evaluation_files/Car%20Evaluation_31_1.png)



```python
from eda import eda_overview
eda_overview.UVA_category(data, data.columns,
                          rowheight = 3, normalize = True);
```

    Categorical features : Index(['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'], dtype='object') 
    



![png](Car%20Evaluation_files/Car%20Evaluation_32_1.png)



```python
from eda import composite_plots
composite_plots.bar_counts(data, data.columns)
```


![png](Car%20Evaluation_files/Car%20Evaluation_33_0.png)


For the car to be acceptable, it has to be low atleast in one of the pricing parameters - maintenance or buying price


```python
dv = "buying"
df = pd.crosstab([data[dv]],[data['class']])
df.head()
df.plot(kind='bar', stacked = True, title=dv )
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('Class Frequency')
```




    Text(0, 0.5, 'Class Frequency')




![png](Car%20Evaluation_files/Car%20Evaluation_35_1.png)



```python
data.columns
```




    Index(['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'], dtype='object')




```python
from eda.composite_plots import features_plots
features_plots(data, ['buying', 'maint'], 'class')
```


![png](Car%20Evaluation_files/Car%20Evaluation_37_0.png)



```python
pd.crosstab(index = [data['buying'],data['maint']], columns = data['class'], margins = False).transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>buying</th>
      <th colspan="4" halign="left">low</th>
      <th colspan="4" halign="left">med</th>
      <th colspan="4" halign="left">high</th>
      <th colspan="4" halign="left">vhigh</th>
    </tr>
    <tr>
      <th>maint</th>
      <th>low</th>
      <th>med</th>
      <th>high</th>
      <th>vhigh</th>
      <th>low</th>
      <th>med</th>
      <th>high</th>
      <th>vhigh</th>
      <th>low</th>
      <th>med</th>
      <th>high</th>
      <th>vhigh</th>
      <th>low</th>
      <th>med</th>
      <th>high</th>
      <th>vhigh</th>
    </tr>
    <tr>
      <th>class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>unacc</th>
      <td>62</td>
      <td>62</td>
      <td>62</td>
      <td>72</td>
      <td>62</td>
      <td>62</td>
      <td>72</td>
      <td>72</td>
      <td>72</td>
      <td>72</td>
      <td>72</td>
      <td>108</td>
      <td>72</td>
      <td>72</td>
      <td>108</td>
      <td>108</td>
    </tr>
    <tr>
      <th>acc</th>
      <td>10</td>
      <td>10</td>
      <td>33</td>
      <td>36</td>
      <td>10</td>
      <td>33</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>0</td>
      <td>36</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>good</th>
      <td>23</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.pivot_table(index=['buying','maint'], aggfunc='size')
```




    buying  maint
    low     low      108
            med      108
            high     108
            vhigh    108
    med     low      108
            med      108
            high     108
            vhigh    108
    high    low      108
            med      108
            high     108
            vhigh    108
    vhigh   low      108
            med      108
            high     108
            vhigh    108
    dtype: int64




```python
df3 = data.groupby(["buying", "maint"]).size().reset_index(name="value_count")
sns.barplot(x = 'buying', y = 'value_count', hue = 'maint', data = df3)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
```




    <matplotlib.legend.Legend at 0x7fac93ffa100>




![png](Car%20Evaluation_files/Car%20Evaluation_40_1.png)

CAR                      car acceptability
   . PRICE                  overall price
   . . buying               buying price
   . . maint                price of the maintenance
   . TECH                   technical characteristics
   . . COMFORT              comfort
   . . . doors              number of doors
   . . . persons            capacity in terms of persons to carry
   . . . lug_boot           the size of luggage boot
   . . safety               estimated safety of the car

```python
pd.set_option('display.max_rows', 20)
df4 = pd.DataFrame({'count' : data.groupby(by = ['buying','maint','class']).size()}).reset_index()
df4.style.background_gradient(cmap='Blues')
#https://stackoverflow.com/questions/12286607/making-heatmap-from-pandas-dataframe
```




<style  type="text/css" >
#T_dd57d_row0_col3,#T_dd57d_row4_col3,#T_dd57d_row8_col3,#T_dd57d_row16_col3,#T_dd57d_row20_col3{
            background-color:  #539ecd;
            color:  #000000;
        }#T_dd57d_row1_col3,#T_dd57d_row5_col3,#T_dd57d_row17_col3{
            background-color:  #e5eff9;
            color:  #000000;
        }#T_dd57d_row2_col3,#T_dd57d_row6_col3,#T_dd57d_row18_col3{
            background-color:  #cde0f1;
            color:  #000000;
        }#T_dd57d_row3_col3,#T_dd57d_row7_col3,#T_dd57d_row10_col3,#T_dd57d_row11_col3,#T_dd57d_row14_col3,#T_dd57d_row15_col3,#T_dd57d_row19_col3,#T_dd57d_row22_col3,#T_dd57d_row23_col3,#T_dd57d_row26_col3,#T_dd57d_row27_col3,#T_dd57d_row30_col3,#T_dd57d_row31_col3,#T_dd57d_row34_col3,#T_dd57d_row35_col3,#T_dd57d_row38_col3,#T_dd57d_row39_col3,#T_dd57d_row42_col3,#T_dd57d_row43_col3,#T_dd57d_row45_col3,#T_dd57d_row46_col3,#T_dd57d_row47_col3,#T_dd57d_row50_col3,#T_dd57d_row51_col3,#T_dd57d_row54_col3,#T_dd57d_row55_col3,#T_dd57d_row57_col3,#T_dd57d_row58_col3,#T_dd57d_row59_col3,#T_dd57d_row61_col3,#T_dd57d_row62_col3,#T_dd57d_row63_col3{
            background-color:  #f7fbff;
            color:  #000000;
        }#T_dd57d_row9_col3,#T_dd57d_row21_col3{
            background-color:  #b4d3e9;
            color:  #000000;
        }#T_dd57d_row12_col3,#T_dd57d_row24_col3,#T_dd57d_row28_col3,#T_dd57d_row32_col3,#T_dd57d_row36_col3,#T_dd57d_row40_col3,#T_dd57d_row48_col3,#T_dd57d_row52_col3{
            background-color:  #3787c0;
            color:  #000000;
        }#T_dd57d_row13_col3,#T_dd57d_row25_col3,#T_dd57d_row29_col3,#T_dd57d_row33_col3,#T_dd57d_row37_col3,#T_dd57d_row41_col3,#T_dd57d_row49_col3,#T_dd57d_row53_col3{
            background-color:  #abd0e6;
            color:  #000000;
        }#T_dd57d_row44_col3,#T_dd57d_row56_col3,#T_dd57d_row60_col3{
            background-color:  #08306b;
            color:  #f1f1f1;
        }</style><table id="T_dd57d_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >buying</th>        <th class="col_heading level0 col1" >maint</th>        <th class="col_heading level0 col2" >class</th>        <th class="col_heading level0 col3" >count</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_dd57d_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_dd57d_row0_col0" class="data row0 col0" >low</td>
                        <td id="T_dd57d_row0_col1" class="data row0 col1" >low</td>
                        <td id="T_dd57d_row0_col2" class="data row0 col2" >unacc</td>
                        <td id="T_dd57d_row0_col3" class="data row0 col3" >62</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_dd57d_row1_col0" class="data row1 col0" >low</td>
                        <td id="T_dd57d_row1_col1" class="data row1 col1" >low</td>
                        <td id="T_dd57d_row1_col2" class="data row1 col2" >acc</td>
                        <td id="T_dd57d_row1_col3" class="data row1 col3" >10</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_dd57d_row2_col0" class="data row2 col0" >low</td>
                        <td id="T_dd57d_row2_col1" class="data row2 col1" >low</td>
                        <td id="T_dd57d_row2_col2" class="data row2 col2" >good</td>
                        <td id="T_dd57d_row2_col3" class="data row2 col3" >23</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_dd57d_row3_col0" class="data row3 col0" >low</td>
                        <td id="T_dd57d_row3_col1" class="data row3 col1" >low</td>
                        <td id="T_dd57d_row3_col2" class="data row3 col2" >v-good</td>
                        <td id="T_dd57d_row3_col3" class="data row3 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_dd57d_row4_col0" class="data row4 col0" >low</td>
                        <td id="T_dd57d_row4_col1" class="data row4 col1" >med</td>
                        <td id="T_dd57d_row4_col2" class="data row4 col2" >unacc</td>
                        <td id="T_dd57d_row4_col3" class="data row4 col3" >62</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_dd57d_row5_col0" class="data row5 col0" >low</td>
                        <td id="T_dd57d_row5_col1" class="data row5 col1" >med</td>
                        <td id="T_dd57d_row5_col2" class="data row5 col2" >acc</td>
                        <td id="T_dd57d_row5_col3" class="data row5 col3" >10</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_dd57d_row6_col0" class="data row6 col0" >low</td>
                        <td id="T_dd57d_row6_col1" class="data row6 col1" >med</td>
                        <td id="T_dd57d_row6_col2" class="data row6 col2" >good</td>
                        <td id="T_dd57d_row6_col3" class="data row6 col3" >23</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_dd57d_row7_col0" class="data row7 col0" >low</td>
                        <td id="T_dd57d_row7_col1" class="data row7 col1" >med</td>
                        <td id="T_dd57d_row7_col2" class="data row7 col2" >v-good</td>
                        <td id="T_dd57d_row7_col3" class="data row7 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_dd57d_row8_col0" class="data row8 col0" >low</td>
                        <td id="T_dd57d_row8_col1" class="data row8 col1" >high</td>
                        <td id="T_dd57d_row8_col2" class="data row8 col2" >unacc</td>
                        <td id="T_dd57d_row8_col3" class="data row8 col3" >62</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_dd57d_row9_col0" class="data row9 col0" >low</td>
                        <td id="T_dd57d_row9_col1" class="data row9 col1" >high</td>
                        <td id="T_dd57d_row9_col2" class="data row9 col2" >acc</td>
                        <td id="T_dd57d_row9_col3" class="data row9 col3" >33</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_dd57d_row10_col0" class="data row10 col0" >low</td>
                        <td id="T_dd57d_row10_col1" class="data row10 col1" >high</td>
                        <td id="T_dd57d_row10_col2" class="data row10 col2" >good</td>
                        <td id="T_dd57d_row10_col3" class="data row10 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_dd57d_row11_col0" class="data row11 col0" >low</td>
                        <td id="T_dd57d_row11_col1" class="data row11 col1" >high</td>
                        <td id="T_dd57d_row11_col2" class="data row11 col2" >v-good</td>
                        <td id="T_dd57d_row11_col3" class="data row11 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_dd57d_row12_col0" class="data row12 col0" >low</td>
                        <td id="T_dd57d_row12_col1" class="data row12 col1" >vhigh</td>
                        <td id="T_dd57d_row12_col2" class="data row12 col2" >unacc</td>
                        <td id="T_dd57d_row12_col3" class="data row12 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_dd57d_row13_col0" class="data row13 col0" >low</td>
                        <td id="T_dd57d_row13_col1" class="data row13 col1" >vhigh</td>
                        <td id="T_dd57d_row13_col2" class="data row13 col2" >acc</td>
                        <td id="T_dd57d_row13_col3" class="data row13 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_dd57d_row14_col0" class="data row14 col0" >low</td>
                        <td id="T_dd57d_row14_col1" class="data row14 col1" >vhigh</td>
                        <td id="T_dd57d_row14_col2" class="data row14 col2" >good</td>
                        <td id="T_dd57d_row14_col3" class="data row14 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_dd57d_row15_col0" class="data row15 col0" >low</td>
                        <td id="T_dd57d_row15_col1" class="data row15 col1" >vhigh</td>
                        <td id="T_dd57d_row15_col2" class="data row15 col2" >v-good</td>
                        <td id="T_dd57d_row15_col3" class="data row15 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_dd57d_row16_col0" class="data row16 col0" >med</td>
                        <td id="T_dd57d_row16_col1" class="data row16 col1" >low</td>
                        <td id="T_dd57d_row16_col2" class="data row16 col2" >unacc</td>
                        <td id="T_dd57d_row16_col3" class="data row16 col3" >62</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_dd57d_row17_col0" class="data row17 col0" >med</td>
                        <td id="T_dd57d_row17_col1" class="data row17 col1" >low</td>
                        <td id="T_dd57d_row17_col2" class="data row17 col2" >acc</td>
                        <td id="T_dd57d_row17_col3" class="data row17 col3" >10</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_dd57d_row18_col0" class="data row18 col0" >med</td>
                        <td id="T_dd57d_row18_col1" class="data row18 col1" >low</td>
                        <td id="T_dd57d_row18_col2" class="data row18 col2" >good</td>
                        <td id="T_dd57d_row18_col3" class="data row18 col3" >23</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_dd57d_row19_col0" class="data row19 col0" >med</td>
                        <td id="T_dd57d_row19_col1" class="data row19 col1" >low</td>
                        <td id="T_dd57d_row19_col2" class="data row19 col2" >v-good</td>
                        <td id="T_dd57d_row19_col3" class="data row19 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_dd57d_row20_col0" class="data row20 col0" >med</td>
                        <td id="T_dd57d_row20_col1" class="data row20 col1" >med</td>
                        <td id="T_dd57d_row20_col2" class="data row20 col2" >unacc</td>
                        <td id="T_dd57d_row20_col3" class="data row20 col3" >62</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_dd57d_row21_col0" class="data row21 col0" >med</td>
                        <td id="T_dd57d_row21_col1" class="data row21 col1" >med</td>
                        <td id="T_dd57d_row21_col2" class="data row21 col2" >acc</td>
                        <td id="T_dd57d_row21_col3" class="data row21 col3" >33</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_dd57d_row22_col0" class="data row22 col0" >med</td>
                        <td id="T_dd57d_row22_col1" class="data row22 col1" >med</td>
                        <td id="T_dd57d_row22_col2" class="data row22 col2" >good</td>
                        <td id="T_dd57d_row22_col3" class="data row22 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_dd57d_row23_col0" class="data row23 col0" >med</td>
                        <td id="T_dd57d_row23_col1" class="data row23 col1" >med</td>
                        <td id="T_dd57d_row23_col2" class="data row23 col2" >v-good</td>
                        <td id="T_dd57d_row23_col3" class="data row23 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_dd57d_row24_col0" class="data row24 col0" >med</td>
                        <td id="T_dd57d_row24_col1" class="data row24 col1" >high</td>
                        <td id="T_dd57d_row24_col2" class="data row24 col2" >unacc</td>
                        <td id="T_dd57d_row24_col3" class="data row24 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_dd57d_row25_col0" class="data row25 col0" >med</td>
                        <td id="T_dd57d_row25_col1" class="data row25 col1" >high</td>
                        <td id="T_dd57d_row25_col2" class="data row25 col2" >acc</td>
                        <td id="T_dd57d_row25_col3" class="data row25 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_dd57d_row26_col0" class="data row26 col0" >med</td>
                        <td id="T_dd57d_row26_col1" class="data row26 col1" >high</td>
                        <td id="T_dd57d_row26_col2" class="data row26 col2" >good</td>
                        <td id="T_dd57d_row26_col3" class="data row26 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_dd57d_row27_col0" class="data row27 col0" >med</td>
                        <td id="T_dd57d_row27_col1" class="data row27 col1" >high</td>
                        <td id="T_dd57d_row27_col2" class="data row27 col2" >v-good</td>
                        <td id="T_dd57d_row27_col3" class="data row27 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_dd57d_row28_col0" class="data row28 col0" >med</td>
                        <td id="T_dd57d_row28_col1" class="data row28 col1" >vhigh</td>
                        <td id="T_dd57d_row28_col2" class="data row28 col2" >unacc</td>
                        <td id="T_dd57d_row28_col3" class="data row28 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_dd57d_row29_col0" class="data row29 col0" >med</td>
                        <td id="T_dd57d_row29_col1" class="data row29 col1" >vhigh</td>
                        <td id="T_dd57d_row29_col2" class="data row29 col2" >acc</td>
                        <td id="T_dd57d_row29_col3" class="data row29 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_dd57d_row30_col0" class="data row30 col0" >med</td>
                        <td id="T_dd57d_row30_col1" class="data row30 col1" >vhigh</td>
                        <td id="T_dd57d_row30_col2" class="data row30 col2" >good</td>
                        <td id="T_dd57d_row30_col3" class="data row30 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_dd57d_row31_col0" class="data row31 col0" >med</td>
                        <td id="T_dd57d_row31_col1" class="data row31 col1" >vhigh</td>
                        <td id="T_dd57d_row31_col2" class="data row31 col2" >v-good</td>
                        <td id="T_dd57d_row31_col3" class="data row31 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_dd57d_row32_col0" class="data row32 col0" >high</td>
                        <td id="T_dd57d_row32_col1" class="data row32 col1" >low</td>
                        <td id="T_dd57d_row32_col2" class="data row32 col2" >unacc</td>
                        <td id="T_dd57d_row32_col3" class="data row32 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_dd57d_row33_col0" class="data row33 col0" >high</td>
                        <td id="T_dd57d_row33_col1" class="data row33 col1" >low</td>
                        <td id="T_dd57d_row33_col2" class="data row33 col2" >acc</td>
                        <td id="T_dd57d_row33_col3" class="data row33 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_dd57d_row34_col0" class="data row34 col0" >high</td>
                        <td id="T_dd57d_row34_col1" class="data row34 col1" >low</td>
                        <td id="T_dd57d_row34_col2" class="data row34 col2" >good</td>
                        <td id="T_dd57d_row34_col3" class="data row34 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_dd57d_row35_col0" class="data row35 col0" >high</td>
                        <td id="T_dd57d_row35_col1" class="data row35 col1" >low</td>
                        <td id="T_dd57d_row35_col2" class="data row35 col2" >v-good</td>
                        <td id="T_dd57d_row35_col3" class="data row35 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_dd57d_row36_col0" class="data row36 col0" >high</td>
                        <td id="T_dd57d_row36_col1" class="data row36 col1" >med</td>
                        <td id="T_dd57d_row36_col2" class="data row36 col2" >unacc</td>
                        <td id="T_dd57d_row36_col3" class="data row36 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_dd57d_row37_col0" class="data row37 col0" >high</td>
                        <td id="T_dd57d_row37_col1" class="data row37 col1" >med</td>
                        <td id="T_dd57d_row37_col2" class="data row37 col2" >acc</td>
                        <td id="T_dd57d_row37_col3" class="data row37 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_dd57d_row38_col0" class="data row38 col0" >high</td>
                        <td id="T_dd57d_row38_col1" class="data row38 col1" >med</td>
                        <td id="T_dd57d_row38_col2" class="data row38 col2" >good</td>
                        <td id="T_dd57d_row38_col3" class="data row38 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_dd57d_row39_col0" class="data row39 col0" >high</td>
                        <td id="T_dd57d_row39_col1" class="data row39 col1" >med</td>
                        <td id="T_dd57d_row39_col2" class="data row39 col2" >v-good</td>
                        <td id="T_dd57d_row39_col3" class="data row39 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_dd57d_row40_col0" class="data row40 col0" >high</td>
                        <td id="T_dd57d_row40_col1" class="data row40 col1" >high</td>
                        <td id="T_dd57d_row40_col2" class="data row40 col2" >unacc</td>
                        <td id="T_dd57d_row40_col3" class="data row40 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_dd57d_row41_col0" class="data row41 col0" >high</td>
                        <td id="T_dd57d_row41_col1" class="data row41 col1" >high</td>
                        <td id="T_dd57d_row41_col2" class="data row41 col2" >acc</td>
                        <td id="T_dd57d_row41_col3" class="data row41 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row42" class="row_heading level0 row42" >42</th>
                        <td id="T_dd57d_row42_col0" class="data row42 col0" >high</td>
                        <td id="T_dd57d_row42_col1" class="data row42 col1" >high</td>
                        <td id="T_dd57d_row42_col2" class="data row42 col2" >good</td>
                        <td id="T_dd57d_row42_col3" class="data row42 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row43" class="row_heading level0 row43" >43</th>
                        <td id="T_dd57d_row43_col0" class="data row43 col0" >high</td>
                        <td id="T_dd57d_row43_col1" class="data row43 col1" >high</td>
                        <td id="T_dd57d_row43_col2" class="data row43 col2" >v-good</td>
                        <td id="T_dd57d_row43_col3" class="data row43 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row44" class="row_heading level0 row44" >44</th>
                        <td id="T_dd57d_row44_col0" class="data row44 col0" >high</td>
                        <td id="T_dd57d_row44_col1" class="data row44 col1" >vhigh</td>
                        <td id="T_dd57d_row44_col2" class="data row44 col2" >unacc</td>
                        <td id="T_dd57d_row44_col3" class="data row44 col3" >108</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row45" class="row_heading level0 row45" >45</th>
                        <td id="T_dd57d_row45_col0" class="data row45 col0" >high</td>
                        <td id="T_dd57d_row45_col1" class="data row45 col1" >vhigh</td>
                        <td id="T_dd57d_row45_col2" class="data row45 col2" >acc</td>
                        <td id="T_dd57d_row45_col3" class="data row45 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row46" class="row_heading level0 row46" >46</th>
                        <td id="T_dd57d_row46_col0" class="data row46 col0" >high</td>
                        <td id="T_dd57d_row46_col1" class="data row46 col1" >vhigh</td>
                        <td id="T_dd57d_row46_col2" class="data row46 col2" >good</td>
                        <td id="T_dd57d_row46_col3" class="data row46 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row47" class="row_heading level0 row47" >47</th>
                        <td id="T_dd57d_row47_col0" class="data row47 col0" >high</td>
                        <td id="T_dd57d_row47_col1" class="data row47 col1" >vhigh</td>
                        <td id="T_dd57d_row47_col2" class="data row47 col2" >v-good</td>
                        <td id="T_dd57d_row47_col3" class="data row47 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row48" class="row_heading level0 row48" >48</th>
                        <td id="T_dd57d_row48_col0" class="data row48 col0" >vhigh</td>
                        <td id="T_dd57d_row48_col1" class="data row48 col1" >low</td>
                        <td id="T_dd57d_row48_col2" class="data row48 col2" >unacc</td>
                        <td id="T_dd57d_row48_col3" class="data row48 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row49" class="row_heading level0 row49" >49</th>
                        <td id="T_dd57d_row49_col0" class="data row49 col0" >vhigh</td>
                        <td id="T_dd57d_row49_col1" class="data row49 col1" >low</td>
                        <td id="T_dd57d_row49_col2" class="data row49 col2" >acc</td>
                        <td id="T_dd57d_row49_col3" class="data row49 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row50" class="row_heading level0 row50" >50</th>
                        <td id="T_dd57d_row50_col0" class="data row50 col0" >vhigh</td>
                        <td id="T_dd57d_row50_col1" class="data row50 col1" >low</td>
                        <td id="T_dd57d_row50_col2" class="data row50 col2" >good</td>
                        <td id="T_dd57d_row50_col3" class="data row50 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row51" class="row_heading level0 row51" >51</th>
                        <td id="T_dd57d_row51_col0" class="data row51 col0" >vhigh</td>
                        <td id="T_dd57d_row51_col1" class="data row51 col1" >low</td>
                        <td id="T_dd57d_row51_col2" class="data row51 col2" >v-good</td>
                        <td id="T_dd57d_row51_col3" class="data row51 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row52" class="row_heading level0 row52" >52</th>
                        <td id="T_dd57d_row52_col0" class="data row52 col0" >vhigh</td>
                        <td id="T_dd57d_row52_col1" class="data row52 col1" >med</td>
                        <td id="T_dd57d_row52_col2" class="data row52 col2" >unacc</td>
                        <td id="T_dd57d_row52_col3" class="data row52 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row53" class="row_heading level0 row53" >53</th>
                        <td id="T_dd57d_row53_col0" class="data row53 col0" >vhigh</td>
                        <td id="T_dd57d_row53_col1" class="data row53 col1" >med</td>
                        <td id="T_dd57d_row53_col2" class="data row53 col2" >acc</td>
                        <td id="T_dd57d_row53_col3" class="data row53 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row54" class="row_heading level0 row54" >54</th>
                        <td id="T_dd57d_row54_col0" class="data row54 col0" >vhigh</td>
                        <td id="T_dd57d_row54_col1" class="data row54 col1" >med</td>
                        <td id="T_dd57d_row54_col2" class="data row54 col2" >good</td>
                        <td id="T_dd57d_row54_col3" class="data row54 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row55" class="row_heading level0 row55" >55</th>
                        <td id="T_dd57d_row55_col0" class="data row55 col0" >vhigh</td>
                        <td id="T_dd57d_row55_col1" class="data row55 col1" >med</td>
                        <td id="T_dd57d_row55_col2" class="data row55 col2" >v-good</td>
                        <td id="T_dd57d_row55_col3" class="data row55 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row56" class="row_heading level0 row56" >56</th>
                        <td id="T_dd57d_row56_col0" class="data row56 col0" >vhigh</td>
                        <td id="T_dd57d_row56_col1" class="data row56 col1" >high</td>
                        <td id="T_dd57d_row56_col2" class="data row56 col2" >unacc</td>
                        <td id="T_dd57d_row56_col3" class="data row56 col3" >108</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row57" class="row_heading level0 row57" >57</th>
                        <td id="T_dd57d_row57_col0" class="data row57 col0" >vhigh</td>
                        <td id="T_dd57d_row57_col1" class="data row57 col1" >high</td>
                        <td id="T_dd57d_row57_col2" class="data row57 col2" >acc</td>
                        <td id="T_dd57d_row57_col3" class="data row57 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row58" class="row_heading level0 row58" >58</th>
                        <td id="T_dd57d_row58_col0" class="data row58 col0" >vhigh</td>
                        <td id="T_dd57d_row58_col1" class="data row58 col1" >high</td>
                        <td id="T_dd57d_row58_col2" class="data row58 col2" >good</td>
                        <td id="T_dd57d_row58_col3" class="data row58 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row59" class="row_heading level0 row59" >59</th>
                        <td id="T_dd57d_row59_col0" class="data row59 col0" >vhigh</td>
                        <td id="T_dd57d_row59_col1" class="data row59 col1" >high</td>
                        <td id="T_dd57d_row59_col2" class="data row59 col2" >v-good</td>
                        <td id="T_dd57d_row59_col3" class="data row59 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row60" class="row_heading level0 row60" >60</th>
                        <td id="T_dd57d_row60_col0" class="data row60 col0" >vhigh</td>
                        <td id="T_dd57d_row60_col1" class="data row60 col1" >vhigh</td>
                        <td id="T_dd57d_row60_col2" class="data row60 col2" >unacc</td>
                        <td id="T_dd57d_row60_col3" class="data row60 col3" >108</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row61" class="row_heading level0 row61" >61</th>
                        <td id="T_dd57d_row61_col0" class="data row61 col0" >vhigh</td>
                        <td id="T_dd57d_row61_col1" class="data row61 col1" >vhigh</td>
                        <td id="T_dd57d_row61_col2" class="data row61 col2" >acc</td>
                        <td id="T_dd57d_row61_col3" class="data row61 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row62" class="row_heading level0 row62" >62</th>
                        <td id="T_dd57d_row62_col0" class="data row62 col0" >vhigh</td>
                        <td id="T_dd57d_row62_col1" class="data row62 col1" >vhigh</td>
                        <td id="T_dd57d_row62_col2" class="data row62 col2" >good</td>
                        <td id="T_dd57d_row62_col3" class="data row62 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_dd57d_level0_row63" class="row_heading level0 row63" >63</th>
                        <td id="T_dd57d_row63_col0" class="data row63 col0" >vhigh</td>
                        <td id="T_dd57d_row63_col1" class="data row63 col1" >vhigh</td>
                        <td id="T_dd57d_row63_col2" class="data row63 col2" >v-good</td>
                        <td id="T_dd57d_row63_col3" class="data row63 col3" >0</td>
            </tr>
    </tbody></table>




```python
df4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying</th>
      <th>maint</th>
      <th>class</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>low</td>
      <td>low</td>
      <td>unacc</td>
      <td>62</td>
    </tr>
    <tr>
      <th>1</th>
      <td>low</td>
      <td>low</td>
      <td>acc</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>low</td>
      <td>good</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>low</td>
      <td>low</td>
      <td>v-good</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>low</td>
      <td>med</td>
      <td>unacc</td>
      <td>62</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59</th>
      <td>vhigh</td>
      <td>high</td>
      <td>v-good</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>unacc</td>
      <td>108</td>
    </tr>
    <tr>
      <th>61</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>acc</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>good</td>
      <td>0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>v-good</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>64 rows × 4 columns</p>
</div>




```python
#pd.DataFrame(data[['buying','maint','class']].value_counts())
```


```python
df4
df4.style.background_gradient(cmap='Blues')
#https://stackoverflow.com/questions/12286607/making-heatmap-from-pandas-dataframe
```




<style  type="text/css" >
#T_ec1d0_row0_col3,#T_ec1d0_row4_col3,#T_ec1d0_row8_col3,#T_ec1d0_row16_col3,#T_ec1d0_row20_col3{
            background-color:  #539ecd;
            color:  #000000;
        }#T_ec1d0_row1_col3,#T_ec1d0_row5_col3,#T_ec1d0_row17_col3{
            background-color:  #e5eff9;
            color:  #000000;
        }#T_ec1d0_row2_col3,#T_ec1d0_row6_col3,#T_ec1d0_row18_col3{
            background-color:  #cde0f1;
            color:  #000000;
        }#T_ec1d0_row3_col3,#T_ec1d0_row7_col3,#T_ec1d0_row10_col3,#T_ec1d0_row11_col3,#T_ec1d0_row14_col3,#T_ec1d0_row15_col3,#T_ec1d0_row19_col3,#T_ec1d0_row22_col3,#T_ec1d0_row23_col3,#T_ec1d0_row26_col3,#T_ec1d0_row27_col3,#T_ec1d0_row30_col3,#T_ec1d0_row31_col3,#T_ec1d0_row34_col3,#T_ec1d0_row35_col3,#T_ec1d0_row38_col3,#T_ec1d0_row39_col3,#T_ec1d0_row42_col3,#T_ec1d0_row43_col3,#T_ec1d0_row45_col3,#T_ec1d0_row46_col3,#T_ec1d0_row47_col3,#T_ec1d0_row50_col3,#T_ec1d0_row51_col3,#T_ec1d0_row54_col3,#T_ec1d0_row55_col3,#T_ec1d0_row57_col3,#T_ec1d0_row58_col3,#T_ec1d0_row59_col3,#T_ec1d0_row61_col3,#T_ec1d0_row62_col3,#T_ec1d0_row63_col3{
            background-color:  #f7fbff;
            color:  #000000;
        }#T_ec1d0_row9_col3,#T_ec1d0_row21_col3{
            background-color:  #b4d3e9;
            color:  #000000;
        }#T_ec1d0_row12_col3,#T_ec1d0_row24_col3,#T_ec1d0_row28_col3,#T_ec1d0_row32_col3,#T_ec1d0_row36_col3,#T_ec1d0_row40_col3,#T_ec1d0_row48_col3,#T_ec1d0_row52_col3{
            background-color:  #3787c0;
            color:  #000000;
        }#T_ec1d0_row13_col3,#T_ec1d0_row25_col3,#T_ec1d0_row29_col3,#T_ec1d0_row33_col3,#T_ec1d0_row37_col3,#T_ec1d0_row41_col3,#T_ec1d0_row49_col3,#T_ec1d0_row53_col3{
            background-color:  #abd0e6;
            color:  #000000;
        }#T_ec1d0_row44_col3,#T_ec1d0_row56_col3,#T_ec1d0_row60_col3{
            background-color:  #08306b;
            color:  #f1f1f1;
        }</style><table id="T_ec1d0_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >buying</th>        <th class="col_heading level0 col1" >maint</th>        <th class="col_heading level0 col2" >class</th>        <th class="col_heading level0 col3" >count</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_ec1d0_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_ec1d0_row0_col0" class="data row0 col0" >low</td>
                        <td id="T_ec1d0_row0_col1" class="data row0 col1" >low</td>
                        <td id="T_ec1d0_row0_col2" class="data row0 col2" >unacc</td>
                        <td id="T_ec1d0_row0_col3" class="data row0 col3" >62</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_ec1d0_row1_col0" class="data row1 col0" >low</td>
                        <td id="T_ec1d0_row1_col1" class="data row1 col1" >low</td>
                        <td id="T_ec1d0_row1_col2" class="data row1 col2" >acc</td>
                        <td id="T_ec1d0_row1_col3" class="data row1 col3" >10</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_ec1d0_row2_col0" class="data row2 col0" >low</td>
                        <td id="T_ec1d0_row2_col1" class="data row2 col1" >low</td>
                        <td id="T_ec1d0_row2_col2" class="data row2 col2" >good</td>
                        <td id="T_ec1d0_row2_col3" class="data row2 col3" >23</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_ec1d0_row3_col0" class="data row3 col0" >low</td>
                        <td id="T_ec1d0_row3_col1" class="data row3 col1" >low</td>
                        <td id="T_ec1d0_row3_col2" class="data row3 col2" >v-good</td>
                        <td id="T_ec1d0_row3_col3" class="data row3 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_ec1d0_row4_col0" class="data row4 col0" >low</td>
                        <td id="T_ec1d0_row4_col1" class="data row4 col1" >med</td>
                        <td id="T_ec1d0_row4_col2" class="data row4 col2" >unacc</td>
                        <td id="T_ec1d0_row4_col3" class="data row4 col3" >62</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_ec1d0_row5_col0" class="data row5 col0" >low</td>
                        <td id="T_ec1d0_row5_col1" class="data row5 col1" >med</td>
                        <td id="T_ec1d0_row5_col2" class="data row5 col2" >acc</td>
                        <td id="T_ec1d0_row5_col3" class="data row5 col3" >10</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_ec1d0_row6_col0" class="data row6 col0" >low</td>
                        <td id="T_ec1d0_row6_col1" class="data row6 col1" >med</td>
                        <td id="T_ec1d0_row6_col2" class="data row6 col2" >good</td>
                        <td id="T_ec1d0_row6_col3" class="data row6 col3" >23</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_ec1d0_row7_col0" class="data row7 col0" >low</td>
                        <td id="T_ec1d0_row7_col1" class="data row7 col1" >med</td>
                        <td id="T_ec1d0_row7_col2" class="data row7 col2" >v-good</td>
                        <td id="T_ec1d0_row7_col3" class="data row7 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_ec1d0_row8_col0" class="data row8 col0" >low</td>
                        <td id="T_ec1d0_row8_col1" class="data row8 col1" >high</td>
                        <td id="T_ec1d0_row8_col2" class="data row8 col2" >unacc</td>
                        <td id="T_ec1d0_row8_col3" class="data row8 col3" >62</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_ec1d0_row9_col0" class="data row9 col0" >low</td>
                        <td id="T_ec1d0_row9_col1" class="data row9 col1" >high</td>
                        <td id="T_ec1d0_row9_col2" class="data row9 col2" >acc</td>
                        <td id="T_ec1d0_row9_col3" class="data row9 col3" >33</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_ec1d0_row10_col0" class="data row10 col0" >low</td>
                        <td id="T_ec1d0_row10_col1" class="data row10 col1" >high</td>
                        <td id="T_ec1d0_row10_col2" class="data row10 col2" >good</td>
                        <td id="T_ec1d0_row10_col3" class="data row10 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_ec1d0_row11_col0" class="data row11 col0" >low</td>
                        <td id="T_ec1d0_row11_col1" class="data row11 col1" >high</td>
                        <td id="T_ec1d0_row11_col2" class="data row11 col2" >v-good</td>
                        <td id="T_ec1d0_row11_col3" class="data row11 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_ec1d0_row12_col0" class="data row12 col0" >low</td>
                        <td id="T_ec1d0_row12_col1" class="data row12 col1" >vhigh</td>
                        <td id="T_ec1d0_row12_col2" class="data row12 col2" >unacc</td>
                        <td id="T_ec1d0_row12_col3" class="data row12 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_ec1d0_row13_col0" class="data row13 col0" >low</td>
                        <td id="T_ec1d0_row13_col1" class="data row13 col1" >vhigh</td>
                        <td id="T_ec1d0_row13_col2" class="data row13 col2" >acc</td>
                        <td id="T_ec1d0_row13_col3" class="data row13 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_ec1d0_row14_col0" class="data row14 col0" >low</td>
                        <td id="T_ec1d0_row14_col1" class="data row14 col1" >vhigh</td>
                        <td id="T_ec1d0_row14_col2" class="data row14 col2" >good</td>
                        <td id="T_ec1d0_row14_col3" class="data row14 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_ec1d0_row15_col0" class="data row15 col0" >low</td>
                        <td id="T_ec1d0_row15_col1" class="data row15 col1" >vhigh</td>
                        <td id="T_ec1d0_row15_col2" class="data row15 col2" >v-good</td>
                        <td id="T_ec1d0_row15_col3" class="data row15 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_ec1d0_row16_col0" class="data row16 col0" >med</td>
                        <td id="T_ec1d0_row16_col1" class="data row16 col1" >low</td>
                        <td id="T_ec1d0_row16_col2" class="data row16 col2" >unacc</td>
                        <td id="T_ec1d0_row16_col3" class="data row16 col3" >62</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_ec1d0_row17_col0" class="data row17 col0" >med</td>
                        <td id="T_ec1d0_row17_col1" class="data row17 col1" >low</td>
                        <td id="T_ec1d0_row17_col2" class="data row17 col2" >acc</td>
                        <td id="T_ec1d0_row17_col3" class="data row17 col3" >10</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_ec1d0_row18_col0" class="data row18 col0" >med</td>
                        <td id="T_ec1d0_row18_col1" class="data row18 col1" >low</td>
                        <td id="T_ec1d0_row18_col2" class="data row18 col2" >good</td>
                        <td id="T_ec1d0_row18_col3" class="data row18 col3" >23</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_ec1d0_row19_col0" class="data row19 col0" >med</td>
                        <td id="T_ec1d0_row19_col1" class="data row19 col1" >low</td>
                        <td id="T_ec1d0_row19_col2" class="data row19 col2" >v-good</td>
                        <td id="T_ec1d0_row19_col3" class="data row19 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_ec1d0_row20_col0" class="data row20 col0" >med</td>
                        <td id="T_ec1d0_row20_col1" class="data row20 col1" >med</td>
                        <td id="T_ec1d0_row20_col2" class="data row20 col2" >unacc</td>
                        <td id="T_ec1d0_row20_col3" class="data row20 col3" >62</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_ec1d0_row21_col0" class="data row21 col0" >med</td>
                        <td id="T_ec1d0_row21_col1" class="data row21 col1" >med</td>
                        <td id="T_ec1d0_row21_col2" class="data row21 col2" >acc</td>
                        <td id="T_ec1d0_row21_col3" class="data row21 col3" >33</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_ec1d0_row22_col0" class="data row22 col0" >med</td>
                        <td id="T_ec1d0_row22_col1" class="data row22 col1" >med</td>
                        <td id="T_ec1d0_row22_col2" class="data row22 col2" >good</td>
                        <td id="T_ec1d0_row22_col3" class="data row22 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_ec1d0_row23_col0" class="data row23 col0" >med</td>
                        <td id="T_ec1d0_row23_col1" class="data row23 col1" >med</td>
                        <td id="T_ec1d0_row23_col2" class="data row23 col2" >v-good</td>
                        <td id="T_ec1d0_row23_col3" class="data row23 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_ec1d0_row24_col0" class="data row24 col0" >med</td>
                        <td id="T_ec1d0_row24_col1" class="data row24 col1" >high</td>
                        <td id="T_ec1d0_row24_col2" class="data row24 col2" >unacc</td>
                        <td id="T_ec1d0_row24_col3" class="data row24 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_ec1d0_row25_col0" class="data row25 col0" >med</td>
                        <td id="T_ec1d0_row25_col1" class="data row25 col1" >high</td>
                        <td id="T_ec1d0_row25_col2" class="data row25 col2" >acc</td>
                        <td id="T_ec1d0_row25_col3" class="data row25 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_ec1d0_row26_col0" class="data row26 col0" >med</td>
                        <td id="T_ec1d0_row26_col1" class="data row26 col1" >high</td>
                        <td id="T_ec1d0_row26_col2" class="data row26 col2" >good</td>
                        <td id="T_ec1d0_row26_col3" class="data row26 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_ec1d0_row27_col0" class="data row27 col0" >med</td>
                        <td id="T_ec1d0_row27_col1" class="data row27 col1" >high</td>
                        <td id="T_ec1d0_row27_col2" class="data row27 col2" >v-good</td>
                        <td id="T_ec1d0_row27_col3" class="data row27 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_ec1d0_row28_col0" class="data row28 col0" >med</td>
                        <td id="T_ec1d0_row28_col1" class="data row28 col1" >vhigh</td>
                        <td id="T_ec1d0_row28_col2" class="data row28 col2" >unacc</td>
                        <td id="T_ec1d0_row28_col3" class="data row28 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_ec1d0_row29_col0" class="data row29 col0" >med</td>
                        <td id="T_ec1d0_row29_col1" class="data row29 col1" >vhigh</td>
                        <td id="T_ec1d0_row29_col2" class="data row29 col2" >acc</td>
                        <td id="T_ec1d0_row29_col3" class="data row29 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_ec1d0_row30_col0" class="data row30 col0" >med</td>
                        <td id="T_ec1d0_row30_col1" class="data row30 col1" >vhigh</td>
                        <td id="T_ec1d0_row30_col2" class="data row30 col2" >good</td>
                        <td id="T_ec1d0_row30_col3" class="data row30 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_ec1d0_row31_col0" class="data row31 col0" >med</td>
                        <td id="T_ec1d0_row31_col1" class="data row31 col1" >vhigh</td>
                        <td id="T_ec1d0_row31_col2" class="data row31 col2" >v-good</td>
                        <td id="T_ec1d0_row31_col3" class="data row31 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_ec1d0_row32_col0" class="data row32 col0" >high</td>
                        <td id="T_ec1d0_row32_col1" class="data row32 col1" >low</td>
                        <td id="T_ec1d0_row32_col2" class="data row32 col2" >unacc</td>
                        <td id="T_ec1d0_row32_col3" class="data row32 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_ec1d0_row33_col0" class="data row33 col0" >high</td>
                        <td id="T_ec1d0_row33_col1" class="data row33 col1" >low</td>
                        <td id="T_ec1d0_row33_col2" class="data row33 col2" >acc</td>
                        <td id="T_ec1d0_row33_col3" class="data row33 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_ec1d0_row34_col0" class="data row34 col0" >high</td>
                        <td id="T_ec1d0_row34_col1" class="data row34 col1" >low</td>
                        <td id="T_ec1d0_row34_col2" class="data row34 col2" >good</td>
                        <td id="T_ec1d0_row34_col3" class="data row34 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_ec1d0_row35_col0" class="data row35 col0" >high</td>
                        <td id="T_ec1d0_row35_col1" class="data row35 col1" >low</td>
                        <td id="T_ec1d0_row35_col2" class="data row35 col2" >v-good</td>
                        <td id="T_ec1d0_row35_col3" class="data row35 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_ec1d0_row36_col0" class="data row36 col0" >high</td>
                        <td id="T_ec1d0_row36_col1" class="data row36 col1" >med</td>
                        <td id="T_ec1d0_row36_col2" class="data row36 col2" >unacc</td>
                        <td id="T_ec1d0_row36_col3" class="data row36 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_ec1d0_row37_col0" class="data row37 col0" >high</td>
                        <td id="T_ec1d0_row37_col1" class="data row37 col1" >med</td>
                        <td id="T_ec1d0_row37_col2" class="data row37 col2" >acc</td>
                        <td id="T_ec1d0_row37_col3" class="data row37 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_ec1d0_row38_col0" class="data row38 col0" >high</td>
                        <td id="T_ec1d0_row38_col1" class="data row38 col1" >med</td>
                        <td id="T_ec1d0_row38_col2" class="data row38 col2" >good</td>
                        <td id="T_ec1d0_row38_col3" class="data row38 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_ec1d0_row39_col0" class="data row39 col0" >high</td>
                        <td id="T_ec1d0_row39_col1" class="data row39 col1" >med</td>
                        <td id="T_ec1d0_row39_col2" class="data row39 col2" >v-good</td>
                        <td id="T_ec1d0_row39_col3" class="data row39 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_ec1d0_row40_col0" class="data row40 col0" >high</td>
                        <td id="T_ec1d0_row40_col1" class="data row40 col1" >high</td>
                        <td id="T_ec1d0_row40_col2" class="data row40 col2" >unacc</td>
                        <td id="T_ec1d0_row40_col3" class="data row40 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_ec1d0_row41_col0" class="data row41 col0" >high</td>
                        <td id="T_ec1d0_row41_col1" class="data row41 col1" >high</td>
                        <td id="T_ec1d0_row41_col2" class="data row41 col2" >acc</td>
                        <td id="T_ec1d0_row41_col3" class="data row41 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row42" class="row_heading level0 row42" >42</th>
                        <td id="T_ec1d0_row42_col0" class="data row42 col0" >high</td>
                        <td id="T_ec1d0_row42_col1" class="data row42 col1" >high</td>
                        <td id="T_ec1d0_row42_col2" class="data row42 col2" >good</td>
                        <td id="T_ec1d0_row42_col3" class="data row42 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row43" class="row_heading level0 row43" >43</th>
                        <td id="T_ec1d0_row43_col0" class="data row43 col0" >high</td>
                        <td id="T_ec1d0_row43_col1" class="data row43 col1" >high</td>
                        <td id="T_ec1d0_row43_col2" class="data row43 col2" >v-good</td>
                        <td id="T_ec1d0_row43_col3" class="data row43 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row44" class="row_heading level0 row44" >44</th>
                        <td id="T_ec1d0_row44_col0" class="data row44 col0" >high</td>
                        <td id="T_ec1d0_row44_col1" class="data row44 col1" >vhigh</td>
                        <td id="T_ec1d0_row44_col2" class="data row44 col2" >unacc</td>
                        <td id="T_ec1d0_row44_col3" class="data row44 col3" >108</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row45" class="row_heading level0 row45" >45</th>
                        <td id="T_ec1d0_row45_col0" class="data row45 col0" >high</td>
                        <td id="T_ec1d0_row45_col1" class="data row45 col1" >vhigh</td>
                        <td id="T_ec1d0_row45_col2" class="data row45 col2" >acc</td>
                        <td id="T_ec1d0_row45_col3" class="data row45 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row46" class="row_heading level0 row46" >46</th>
                        <td id="T_ec1d0_row46_col0" class="data row46 col0" >high</td>
                        <td id="T_ec1d0_row46_col1" class="data row46 col1" >vhigh</td>
                        <td id="T_ec1d0_row46_col2" class="data row46 col2" >good</td>
                        <td id="T_ec1d0_row46_col3" class="data row46 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row47" class="row_heading level0 row47" >47</th>
                        <td id="T_ec1d0_row47_col0" class="data row47 col0" >high</td>
                        <td id="T_ec1d0_row47_col1" class="data row47 col1" >vhigh</td>
                        <td id="T_ec1d0_row47_col2" class="data row47 col2" >v-good</td>
                        <td id="T_ec1d0_row47_col3" class="data row47 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row48" class="row_heading level0 row48" >48</th>
                        <td id="T_ec1d0_row48_col0" class="data row48 col0" >vhigh</td>
                        <td id="T_ec1d0_row48_col1" class="data row48 col1" >low</td>
                        <td id="T_ec1d0_row48_col2" class="data row48 col2" >unacc</td>
                        <td id="T_ec1d0_row48_col3" class="data row48 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row49" class="row_heading level0 row49" >49</th>
                        <td id="T_ec1d0_row49_col0" class="data row49 col0" >vhigh</td>
                        <td id="T_ec1d0_row49_col1" class="data row49 col1" >low</td>
                        <td id="T_ec1d0_row49_col2" class="data row49 col2" >acc</td>
                        <td id="T_ec1d0_row49_col3" class="data row49 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row50" class="row_heading level0 row50" >50</th>
                        <td id="T_ec1d0_row50_col0" class="data row50 col0" >vhigh</td>
                        <td id="T_ec1d0_row50_col1" class="data row50 col1" >low</td>
                        <td id="T_ec1d0_row50_col2" class="data row50 col2" >good</td>
                        <td id="T_ec1d0_row50_col3" class="data row50 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row51" class="row_heading level0 row51" >51</th>
                        <td id="T_ec1d0_row51_col0" class="data row51 col0" >vhigh</td>
                        <td id="T_ec1d0_row51_col1" class="data row51 col1" >low</td>
                        <td id="T_ec1d0_row51_col2" class="data row51 col2" >v-good</td>
                        <td id="T_ec1d0_row51_col3" class="data row51 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row52" class="row_heading level0 row52" >52</th>
                        <td id="T_ec1d0_row52_col0" class="data row52 col0" >vhigh</td>
                        <td id="T_ec1d0_row52_col1" class="data row52 col1" >med</td>
                        <td id="T_ec1d0_row52_col2" class="data row52 col2" >unacc</td>
                        <td id="T_ec1d0_row52_col3" class="data row52 col3" >72</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row53" class="row_heading level0 row53" >53</th>
                        <td id="T_ec1d0_row53_col0" class="data row53 col0" >vhigh</td>
                        <td id="T_ec1d0_row53_col1" class="data row53 col1" >med</td>
                        <td id="T_ec1d0_row53_col2" class="data row53 col2" >acc</td>
                        <td id="T_ec1d0_row53_col3" class="data row53 col3" >36</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row54" class="row_heading level0 row54" >54</th>
                        <td id="T_ec1d0_row54_col0" class="data row54 col0" >vhigh</td>
                        <td id="T_ec1d0_row54_col1" class="data row54 col1" >med</td>
                        <td id="T_ec1d0_row54_col2" class="data row54 col2" >good</td>
                        <td id="T_ec1d0_row54_col3" class="data row54 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row55" class="row_heading level0 row55" >55</th>
                        <td id="T_ec1d0_row55_col0" class="data row55 col0" >vhigh</td>
                        <td id="T_ec1d0_row55_col1" class="data row55 col1" >med</td>
                        <td id="T_ec1d0_row55_col2" class="data row55 col2" >v-good</td>
                        <td id="T_ec1d0_row55_col3" class="data row55 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row56" class="row_heading level0 row56" >56</th>
                        <td id="T_ec1d0_row56_col0" class="data row56 col0" >vhigh</td>
                        <td id="T_ec1d0_row56_col1" class="data row56 col1" >high</td>
                        <td id="T_ec1d0_row56_col2" class="data row56 col2" >unacc</td>
                        <td id="T_ec1d0_row56_col3" class="data row56 col3" >108</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row57" class="row_heading level0 row57" >57</th>
                        <td id="T_ec1d0_row57_col0" class="data row57 col0" >vhigh</td>
                        <td id="T_ec1d0_row57_col1" class="data row57 col1" >high</td>
                        <td id="T_ec1d0_row57_col2" class="data row57 col2" >acc</td>
                        <td id="T_ec1d0_row57_col3" class="data row57 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row58" class="row_heading level0 row58" >58</th>
                        <td id="T_ec1d0_row58_col0" class="data row58 col0" >vhigh</td>
                        <td id="T_ec1d0_row58_col1" class="data row58 col1" >high</td>
                        <td id="T_ec1d0_row58_col2" class="data row58 col2" >good</td>
                        <td id="T_ec1d0_row58_col3" class="data row58 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row59" class="row_heading level0 row59" >59</th>
                        <td id="T_ec1d0_row59_col0" class="data row59 col0" >vhigh</td>
                        <td id="T_ec1d0_row59_col1" class="data row59 col1" >high</td>
                        <td id="T_ec1d0_row59_col2" class="data row59 col2" >v-good</td>
                        <td id="T_ec1d0_row59_col3" class="data row59 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row60" class="row_heading level0 row60" >60</th>
                        <td id="T_ec1d0_row60_col0" class="data row60 col0" >vhigh</td>
                        <td id="T_ec1d0_row60_col1" class="data row60 col1" >vhigh</td>
                        <td id="T_ec1d0_row60_col2" class="data row60 col2" >unacc</td>
                        <td id="T_ec1d0_row60_col3" class="data row60 col3" >108</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row61" class="row_heading level0 row61" >61</th>
                        <td id="T_ec1d0_row61_col0" class="data row61 col0" >vhigh</td>
                        <td id="T_ec1d0_row61_col1" class="data row61 col1" >vhigh</td>
                        <td id="T_ec1d0_row61_col2" class="data row61 col2" >acc</td>
                        <td id="T_ec1d0_row61_col3" class="data row61 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row62" class="row_heading level0 row62" >62</th>
                        <td id="T_ec1d0_row62_col0" class="data row62 col0" >vhigh</td>
                        <td id="T_ec1d0_row62_col1" class="data row62 col1" >vhigh</td>
                        <td id="T_ec1d0_row62_col2" class="data row62 col2" >good</td>
                        <td id="T_ec1d0_row62_col3" class="data row62 col3" >0</td>
            </tr>
            <tr>
                        <th id="T_ec1d0_level0_row63" class="row_heading level0 row63" >63</th>
                        <td id="T_ec1d0_row63_col0" class="data row63 col0" >vhigh</td>
                        <td id="T_ec1d0_row63_col1" class="data row63 col1" >vhigh</td>
                        <td id="T_ec1d0_row63_col2" class="data row63 col2" >v-good</td>
                        <td id="T_ec1d0_row63_col3" class="data row63 col3" >0</td>
            </tr>
    </tbody></table>



## Price vs Class heatmap
Price has two components - buying price and maintenance.

- Data is equally distributed for buying price (4 levels) and maintenance(4 levels) combinations.
- Class instances of good and very good have buying price and maintenance ranging from **low and medium**.

### Buying Price vs Class


```python
pd.crosstab(index = [data['buying']], columns = data['class'], margins = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>class</th>
      <th>unacc</th>
      <th>acc</th>
      <th>good</th>
    </tr>
    <tr>
      <th>buying</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>low</th>
      <td>258</td>
      <td>89</td>
      <td>46</td>
    </tr>
    <tr>
      <th>med</th>
      <td>268</td>
      <td>115</td>
      <td>23</td>
    </tr>
    <tr>
      <th>high</th>
      <td>324</td>
      <td>108</td>
      <td>0</td>
    </tr>
    <tr>
      <th>vhigh</th>
      <td>360</td>
      <td>72</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from eda.composite_plots import heatmap_plot
heatmap_plot(data, index = ['buying'], column = ['class'])
```


![png](Car%20Evaluation_files/Car%20Evaluation_49_0.png)


### Maintenance vs Class


```python
pd.crosstab(index = [data['maint']], columns = data['class'], margins = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>class</th>
      <th>unacc</th>
      <th>acc</th>
      <th>good</th>
    </tr>
    <tr>
      <th>maint</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>low</th>
      <td>268</td>
      <td>92</td>
      <td>46</td>
    </tr>
    <tr>
      <th>med</th>
      <td>268</td>
      <td>115</td>
      <td>23</td>
    </tr>
    <tr>
      <th>high</th>
      <td>314</td>
      <td>105</td>
      <td>0</td>
    </tr>
    <tr>
      <th>vhigh</th>
      <td>360</td>
      <td>72</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(4,3))
#font_kwds = dict(fontsize = 12)

df_MainVsClass = pd.crosstab(index = [data['maint']], columns = data['class'], margins = False)

sns.heatmap(df_MainVsClass)
ax.tick_params(axis = 'both', labelcolor = 'black', labelsize = 12) 
plt.xlabel(ax.get_xlabel(), fontsize = 15, fontweight = 'heavy')
plt.ylabel(ax.get_ylabel(), fontsize = 15, fontweight = 'heavy')
```




    Text(20.72222222222222, 0.5, 'maint')




![png](Car%20Evaluation_files/Car%20Evaluation_52_1.png)


### Buying vs Maintenance


```python
df_BuyingVsMaintenance = pd.crosstab(index = [data['buying']], columns = data['maint'], margins = False)

df_BuyingVsMaintenance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>maint</th>
      <th>low</th>
      <th>med</th>
      <th>high</th>
      <th>vhigh</th>
    </tr>
    <tr>
      <th>buying</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>low</th>
      <td>108</td>
      <td>108</td>
      <td>108</td>
      <td>108</td>
    </tr>
    <tr>
      <th>med</th>
      <td>108</td>
      <td>108</td>
      <td>108</td>
      <td>108</td>
    </tr>
    <tr>
      <th>high</th>
      <td>108</td>
      <td>108</td>
      <td>108</td>
      <td>108</td>
    </tr>
    <tr>
      <th>vhigh</th>
      <td>108</td>
      <td>108</td>
      <td>108</td>
      <td>108</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(4,3))
#font_kwds = dict(fontsize = 12)
                       
sns.heatmap(df_BuyingVsMaintenance)
ax.tick_params(axis = 'both', labelcolor = 'black', labelsize = 12) 
plt.xlabel(ax.get_xlabel(), fontsize = 15, fontweight = 'heavy')
plt.ylabel(ax.get_ylabel(), fontsize = 15, fontweight = 'heavy')
```




    Text(20.72222222222222, 0.5, 'buying')




![png](Car%20Evaluation_files/Car%20Evaluation_55_1.png)


### Price (Buying Price and Maintenance) vs Class


```python
df_PricingVsClass = pd.crosstab(index = [data['buying'],data['maint']], columns = data['class'], margins = False)

df_PricingVsClass
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>unacc</th>
      <th>acc</th>
      <th>good</th>
    </tr>
    <tr>
      <th>buying</th>
      <th>maint</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">low</th>
      <th>low</th>
      <td>62</td>
      <td>10</td>
      <td>23</td>
    </tr>
    <tr>
      <th>med</th>
      <td>62</td>
      <td>10</td>
      <td>23</td>
    </tr>
    <tr>
      <th>high</th>
      <td>62</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>vhigh</th>
      <td>72</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">med</th>
      <th>low</th>
      <td>62</td>
      <td>10</td>
      <td>23</td>
    </tr>
    <tr>
      <th>med</th>
      <td>62</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>high</th>
      <td>72</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>vhigh</th>
      <td>72</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">high</th>
      <th>low</th>
      <td>72</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>med</th>
      <td>72</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>high</th>
      <td>72</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>vhigh</th>
      <td>108</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">vhigh</th>
      <th>low</th>
      <td>72</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>med</th>
      <td>72</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>high</th>
      <td>108</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>vhigh</th>
      <td>108</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(5,4))
#font_kwds = dict(fontsize = 12)
sns.heatmap(df_PricingVsClass)
ax.tick_params(axis = 'both', labelcolor = 'black', labelsize = 10) 
plt.xlabel(ax.get_xlabel(), fontsize = 8, fontweight = 'heavy')
plt.ylabel(ax.get_ylabel(), fontsize = 8, fontweight = 'heavy')
```




    Text(33.222222222222214, 0.5, 'buying-maint')




![png](Car%20Evaluation_files/Car%20Evaluation_58_1.png)


## Comfort vs Class Heatmaps

Comfort has three aspects - doors, luggage boot size and persons

- All two persons car are classified as Unacceptable.
- No specific pattern emerging from EDA of Comfort vs Class.


```python
from eda.composite_plots import heatmap_plot
heatmap_plot(data, ['lug_boot'], ['class'])
```


![png](Car%20Evaluation_files/Car%20Evaluation_60_0.png)



```python
pd.crosstab([data['doors'], data['lug_boot']], [data['class']])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>unacc</th>
      <th>acc</th>
      <th>good</th>
    </tr>
    <tr>
      <th>doors</th>
      <th>lug_boot</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>small</th>
      <td>126</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>med</th>
      <td>108</td>
      <td>30</td>
      <td>6</td>
    </tr>
    <tr>
      <th>big</th>
      <td>92</td>
      <td>36</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">3</th>
      <th>small</th>
      <td>108</td>
      <td>30</td>
      <td>6</td>
    </tr>
    <tr>
      <th>med</th>
      <td>100</td>
      <td>33</td>
      <td>6</td>
    </tr>
    <tr>
      <th>big</th>
      <td>92</td>
      <td>36</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">4</th>
      <th>small</th>
      <td>108</td>
      <td>30</td>
      <td>6</td>
    </tr>
    <tr>
      <th>med</th>
      <td>92</td>
      <td>36</td>
      <td>6</td>
    </tr>
    <tr>
      <th>big</th>
      <td>92</td>
      <td>36</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">5more</th>
      <th>small</th>
      <td>108</td>
      <td>30</td>
      <td>6</td>
    </tr>
    <tr>
      <th>med</th>
      <td>92</td>
      <td>36</td>
      <td>6</td>
    </tr>
    <tr>
      <th>big</th>
      <td>92</td>
      <td>36</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
from eda.composite_plots import heatmap_plot
heatmap_plot(data, ['doors','lug_boot'], ['class'])
```


![png](Car%20Evaluation_files/Car%20Evaluation_62_0.png)



```python
data.columns
```




    Index(['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'], dtype='object')



# Defining the Problem

This is a multiclass classification problem.  
Data is imbalanced as all the classes are not equally represented.  
Accuracy is not the metric to use when working with an imbalanced dataset. We have seen that it is misleading.  
Instead, we can use below performance measures :  
- Confusion Matrix: A breakdown of predictions into a table showing correct predictions (the diagonal) and the types of incorrect predictions made (what classes incorrect predictions were assigned).  
- Precision: A measure of a classifiers exactness.  
- Recall: A measure of a classifiers completeness  
- F1 Score (or F-score): A weighted average of precision and recall.  


Appropriateness of the performance measures :  
- Accuracy : Appropriate for Balanced datasets  
- Precision : Appropriate when minimising false positive is the focus  
- Recall : Appropriate when minimising false negatives is the focus.  
- F-measure provides a way to combine both precision and recall into a single measure that captures both properties.  
  
Note : It is common practice to measure efficacy of binary classification models using the **Area under the Curve** (AUC) of the ROC curve. Multi class classicfication will require some tweaks in the ROC AUC approach. We are not pursuing that in this notebook. Read this for more info :https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

#http://www.svds.com/learning-imbalanced-classes/





```python
from eda.axes_utils import Add_valuecountsinfo, Add_data_labels

ax = data['class'].value_counts().plot(kind = 'bar', figsize=(3,3))
ax.yaxis.grid(True, alpha = 0.3)
ax.set(axisbelow = True)
Add_data_labels(ax.patches)
Add_valuecountsinfo(ax, 'class', data)
```


![png](Car%20Evaluation_files/Car%20Evaluation_66_0.png)



```python
from eda import axes_utils
print(list(dir(axes_utils)))
```

    ['Add_data_labels', 'Add_valuecountsinfo', 'Change_barWidth', 'Highlight_Top_n_values', 'Set_axes_labels_titles', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'blended_transform_factory', 'create_string_for_plot', 'get_hist', 'gridspec', 'mdates', 'np', 'paddingString', 'pd', 'plt', 'sns', 'stylize_axes', 'time']


# Approach for imbalanced datasets

Techniques to correctly distinguish the minority class can be categorized into *four main groups*, depending on how they deal with the problem. The four groups are :  
  
***  
1. **Algorithm level approaches** (also called internal), try to adapt existing classifier learning algorithms to bias the learning toward the minority class. In order to perform the adaptation a special knowledge of both the corresponding classifier and the application domain is required so as to comprehend why the classifier fails when the class distribution is uneven. More details about these types of methods are given in Chap. 6.  
***    
    
2. **Data level (or external) approaches** aim at rebalancing the class distribution by resampling the data space . This way, the modification of the learning algorithm is avoided since the effect caused by imbalance is decreased with a preprocessing step. These methods are discussed in depth in Chap. 5.  
***  
3. **Cost-sensitive learning** framework falls between data and algorithm level approaches. Both data level transformations (by adding costs to instances) and algorithm level modifications (by modifying the learning process to accept costs) [13, 48, 86] are incorporated. The classifier is biased toward the minority class by assuming higher misclassification costs for this class and seeking to minimize the total cost errors of both classes. An overview of cost-sensitive approaches for the class imbalance problem is presented Chap. 4.  
***  
4. **Ensemble-based methods** usually consist of a combination between an ensemble learning algorithm [59] and one of the techniques above, specifically, data level and cost-sensitive ones [27]. Adding a data level approach to the ensemble learning algorithm, the new hybrid method usually preprocesses the data before training each classifier, whereas cost-sensitive ensembles instead of modifying the base classifier in order to accept costs in the learning process, guide the cost minimization via the ensemble learning algorithm. Ensemble-based models are thoroughly described in Chap. 7.  
***  
*Learning from imbalanced datasets* by Alberto Fernandez et al

# Applying Machine Learning Algorithms for classification

## Training and testing the data


```python
X_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
Y_cols = ['class']

X = data[X_cols]
Y = data[Y_cols]
```


```python
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
```


```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying</th>
      <th>maint</th>
      <th>doors</th>
      <th>persons</th>
      <th>lug_boot</th>
      <th>safety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>142</th>
      <td>vhigh</td>
      <td>high</td>
      <td>3</td>
      <td>2</td>
      <td>big</td>
      <td>med</td>
    </tr>
    <tr>
      <th>1026</th>
      <td>med</td>
      <td>high</td>
      <td>4</td>
      <td>2</td>
      <td>small</td>
      <td>low</td>
    </tr>
    <tr>
      <th>537</th>
      <td>high</td>
      <td>vhigh</td>
      <td>5more</td>
      <td>more</td>
      <td>big</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1298</th>
      <td>low</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>high</td>
    </tr>
    <tr>
      <th>1296</th>
      <td>low</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>low</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>715</th>
      <td>high</td>
      <td>med</td>
      <td>4</td>
      <td>4</td>
      <td>med</td>
      <td>med</td>
    </tr>
    <tr>
      <th>905</th>
      <td>med</td>
      <td>vhigh</td>
      <td>3</td>
      <td>4</td>
      <td>med</td>
      <td>high</td>
    </tr>
    <tr>
      <th>1096</th>
      <td>med</td>
      <td>med</td>
      <td>2</td>
      <td>4</td>
      <td>big</td>
      <td>med</td>
    </tr>
    <tr>
      <th>235</th>
      <td>vhigh</td>
      <td>med</td>
      <td>2</td>
      <td>more</td>
      <td>small</td>
      <td>med</td>
    </tr>
    <tr>
      <th>1061</th>
      <td>med</td>
      <td>high</td>
      <td>5more</td>
      <td>2</td>
      <td>big</td>
      <td>high</td>
    </tr>
  </tbody>
</table>
<p>1296 rows × 6 columns</p>
</div>



## Preparing the data


```python
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)
```

    Train (1296, 6) (1296, 1)
    Test (432, 6) (432, 1)



```python
#Verifying order of columns
#cat = ["buying","maint","doors","persons","lug_boot","safety"]
print(X_train.columns)
print(X_test.columns)
```

    Index(['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'], dtype='object')
    Index(['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'], dtype='object')



```python
from sklearn.preprocessing import OrdinalEncoder

#https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#sphx-glr-auto-examples-applications-plot-cyclical-feature-engineering-py

# prepare input data
def prepare_inputs(X_train, X_test): 
    oe = OrdinalEncoder(categories = [['low','med','high', 'vhigh' ],
                         ['low','med','high', 'vhigh' ],
                         ['2', '3', '4', '5more'],
                         ['2', '4', 'more'],
                         ['small', 'med', 'big'],
                         ['low', 'med', 'high']]
                       )
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train) 
    X_test_enc = oe.transform(X_test) 
    
    return X_train_enc, X_test_enc
```


```python
from sklearn.preprocessing import LabelEncoder

# prepare input data
def prepare_targets(y_train, y_test): 
    le = LabelEncoder()
    le.fit(np.ravel(y_train))
    y_train_enc = le.transform(np.ravel(y_train)) 
    y_test_enc = le.transform(np.ravel(y_test)) 
    
    return y_train_enc, y_test_enc
```


```python
from sklearn.preprocessing import OrdinalEncoder
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)

# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
```

## Feature Selection (Optional)

Comparison of below three scenarios do not justify feature selection methods.
- Accuracy when all features are considered : 81.02%
- Accuracy when 4 features selected through "information gain" considered
- Accuracy when 4 features selected through "Chi Squared" considered

Note we have applied Logistic regression classifier on all three scenarios to check any significant movement in the accuracy scores.


```python
# fit the model using all the features
model = LogisticRegression(solver='lbfgs', max_iter = 1000)
model.fit(X_train_enc, y_train_enc)
# evaluate the model
yhat = model.predict(X_test_enc)
# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %.2f' % (accuracy*100))
```

    Accuracy: 81.02


### example of mutual information feature selection for categorical data


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
```


```python
# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k=4) 
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
```


```python
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
```


```python
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
```

    Feature 0: 0.094965
    Feature 1: 0.059104
    Feature 2: 0.000000
    Feature 3: 0.127436
    Feature 4: 0.030315
    Feature 5: 0.195964



```python
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_) 
plt.show()

```


![png](Car%20Evaluation_files/Car%20Evaluation_87_0.png)



```python
# fit the model
model = LogisticRegression(solver='lbfgs',max_iter= 1000)
model.fit(X_train_fs, y_train_enc)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %.2f' % (accuracy*100))
```

    Accuracy: 81.25


### example of chi squared feature selection for categorical data


```python
from sklearn.feature_selection import chi2
# feature selection
def select_features2(X_train, y_train, X_test): 
    fs = SelectKBest(score_func=chi2, k=4)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
```


```python
# feature selection
X_train_fs2, X_test_fs2, fs2 = select_features2(X_train_enc, y_train_enc, X_test_enc)

# what are scores for the features
for i in range(len(fs2.scores_)):
    print('Feature %d: %f' % (i, fs2.scores_[i]))

# plot the scores
plt.bar([i for i in range(len(fs2.scores_))], fs2.scores_) 
plt.show()
```

    Feature 0: 107.312113
    Feature 1: 71.158014
    Feature 2: 5.761583
    Feature 3: 136.535729
    Feature 4: 27.666582
    Feature 5: 197.004190



![png](Car%20Evaluation_files/Car%20Evaluation_91_1.png)



```python
# fit the model
model = LogisticRegression(solver='lbfgs',max_iter= 1000)
model.fit(X_train_fs2, y_train_enc)
# evaluate the model
yhat = model.predict(X_test_fs2)
# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %.2f' % (accuracy*100))
```

    Accuracy: 81.25



```python
## Decision Tree incase of categorical variable as input
```

## Popular algorithms for multi-class classification include:
  
- k-Nearest Neighbors  
- Decision Trees  
- Naive Bayes  
- Random Forest  
- Gradient Boosting 


    
- Note :
    - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
    - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn-ensemble-gradientboostingclassifier)

A learning curve shows the validation and training score of an estimator for varying numbers of training samples. It is a tool to find out how much we benefit from adding more training data and whether the estimator suffers more from a variance error or a bias error.  

- <https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve>  
- <https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py>
# Sample code for calculating time

st_time = time.time()

#code.......
#code.......

en_time = time.time()
print('Total time: {:.2f}s'.format(en_time-st_time))

```python
# Refer to section 8 for the evaluation metrics

def evaluation_parametrics(y_train,yp_train,y_test,yp_test):
  print("--------------------------------------------------------------------------")
  print("Classification Report for Train Data")
  print(classification_report(y_train, yp_train))
  print("Classification Report for Test Data")
  print(classification_report(y_test, yp_test))
  print("--------------------------------------------------------------------------")
  # Accuracy
  print("Accuracy on Train Data is: {}".format(round(accuracy_score(y_train,yp_train),2)))
  print("Accuracy on Test Data is: {}".format(round(accuracy_score(y_test,yp_test),2)))
  print("--------------------------------------------------------------------------")
  # Precision
  print("Precision on Train Data is: {}".format(round(precision_score(y_train,yp_train,average = "weighted"),2)))
  print("Precision on Test Data is: {}".format(round(precision_score(y_test,yp_test,average = "weighted"),2)))
  print("--------------------------------------------------------------------------")
  # Recall 
  print("Recall on Train Data is: {}".format(round(recall_score(y_train,yp_train,average = "weighted"),2)))
  print("Recall on Test Data is: {}".format(round(recall_score(y_test,yp_test,average = "weighted"),2)))
  print("--------------------------------------------------------------------------")
  # F1 Score
  print("F1 Score on Train Data is: {}".format(round(f1_score(y_train,yp_train,average = "weighted"),2)))
  print("F1 Score on Test Data is: {}".format(round(f1_score(y_test,yp_test,average = "weighted"),2)))
  print("--------------------------------------------------------------------------")
```


```python
def create_dict(model, modelname, y_train, yp_train, y_test, yp_test):
    dict1 = {modelname :  {"F1" : {"Train": float(np.round(f1_score(y_train,yp_train,average = "weighted"),2)),
                                  "Test": float(np.round(f1_score(y_test,yp_test,average = "weighted"),2))},
                            "Recall": {"Train": float(np.round(recall_score(y_train,yp_train,average = "weighted"),2)),
                                       "Test": float(np.round(recall_score(y_test,yp_test,average = "weighted"),2))},
                            "Precision" :{"Train": float(np.round(precision_score(y_train,yp_train,average = "weighted"),2)),
                                        "Test": float(np.round(precision_score(y_test,yp_test,average = "weighted"),2))
                                       }}
                          
            }
    return dict1

dict = {}
```


```python
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
```

## Logistic Regression Classifier


```python
lr = LogisticRegression(max_iter = 1000,random_state = 48, multi_class = 'multinomial')

st_time = time.time()
lr.fit(X_train_enc,y_train_enc)

yp_train_enc = lr.predict(X_train_enc)
yp_test_enc = lr.predict(X_test_enc)

en_time = time.time()
print('Total time: {:.2f}s'.format(en_time-st_time))

evaluation_parametrics(y_train_enc,yp_train_enc,y_test_enc,yp_test_enc)
plot_confusion_matrix(lr,X_test_enc, y_test_enc)

dict1 = create_dict(lr, "Logistic Regression Classifier", y_train_enc, yp_train_enc, y_test_enc, yp_test_enc)
dict.update(dict1)

plot_learning_curve(lr, X = X_train_enc, y = y_train_enc, 
                    title =  "Learning Curves (Logistic Regression Classifier)", 
                    train_sizes=np.linspace(0.1, 1.0, 5))

```

    Total time: 0.06s
    --------------------------------------------------------------------------
    Classification Report for Train Data
                  precision    recall  f1-score   support
    
               0       0.68      0.63      0.66       296
               1       0.60      0.41      0.49        51
               2       0.89      0.93      0.91       900
               3       0.80      0.71      0.75        49
    
        accuracy                           0.83      1296
       macro avg       0.74      0.67      0.70      1296
    weighted avg       0.83      0.83      0.83      1296
    
    Classification Report for Test Data
                  precision    recall  f1-score   support
    
               0       0.60      0.53      0.57        88
               1       0.42      0.28      0.33        18
               2       0.88      0.93      0.90       310
               3       0.79      0.69      0.73        16
    
        accuracy                           0.81       432
       macro avg       0.67      0.61      0.63       432
    weighted avg       0.80      0.81      0.80       432
    
    --------------------------------------------------------------------------
    Accuracy on Train Data is: 0.83
    Accuracy on Test Data is: 0.81
    --------------------------------------------------------------------------
    Precision on Train Data is: 0.83
    Precision on Test Data is: 0.8
    --------------------------------------------------------------------------
    Recall on Train Data is: 0.83
    Recall on Test Data is: 0.81
    --------------------------------------------------------------------------
    F1 Score on Train Data is: 0.83
    F1 Score on Test Data is: 0.8
    --------------------------------------------------------------------------


    /Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <module 'matplotlib.pyplot' from '/Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/matplotlib/pyplot.py'>




![png](Car%20Evaluation_files/Car%20Evaluation_101_3.png)



![png](Car%20Evaluation_files/Car%20Evaluation_101_4.png)


## Decision Tree Classifier


```python
dt = DecisionTreeClassifier(max_depth = 7,random_state = 48) # Keeping max_depth = 7 to avoid overfitting
dt.fit(X_train_enc,y_train_enc)

yp_train_enc = dt.predict(X_train_enc)
yp_test_enc = dt.predict(X_test_enc)

evaluation_parametrics(y_train_enc,yp_train_enc,y_test_enc,yp_test_enc)
plot_confusion_matrix(dt,X_test_enc, y_test_enc)

dict1 = create_dict(dt, "Decision Tree Classifier", y_train_enc, yp_train_enc, y_test_enc, yp_test_enc)
dict.update(dict1)

plot_learning_curve(dt, X = X_train_enc, y = y_train_enc, 
                    title =  "Learning Curves (Decision Tree Classifier)", 
                    train_sizes=np.linspace(0.1, 1.0, 5))

```

    --------------------------------------------------------------------------
    Classification Report for Train Data
                  precision    recall  f1-score   support
    
               0       0.84      0.96      0.89       296
               1       0.83      0.59      0.69        51
               2       0.99      0.96      0.98       900
               3       0.79      0.78      0.78        49
    
        accuracy                           0.94      1296
       macro avg       0.86      0.82      0.84      1296
    weighted avg       0.94      0.94      0.94      1296
    
    Classification Report for Test Data
                  precision    recall  f1-score   support
    
               0       0.78      0.94      0.85        88
               1       0.83      0.56      0.67        18
               2       0.99      0.95      0.97       310
               3       0.88      0.88      0.88        16
    
        accuracy                           0.93       432
       macro avg       0.87      0.83      0.84       432
    weighted avg       0.94      0.93      0.93       432
    
    --------------------------------------------------------------------------
    Accuracy on Train Data is: 0.94
    Accuracy on Test Data is: 0.93
    --------------------------------------------------------------------------
    Precision on Train Data is: 0.94
    Precision on Test Data is: 0.94
    --------------------------------------------------------------------------
    Recall on Train Data is: 0.94
    Recall on Test Data is: 0.93
    --------------------------------------------------------------------------
    F1 Score on Train Data is: 0.94
    F1 Score on Test Data is: 0.93
    --------------------------------------------------------------------------


    /Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <module 'matplotlib.pyplot' from '/Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/matplotlib/pyplot.py'>




![png](Car%20Evaluation_files/Car%20Evaluation_103_3.png)



![png](Car%20Evaluation_files/Car%20Evaluation_103_4.png)


## K Nearest Neighbors Classifier


```python
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)

st_time = time.time()

knn.fit(X_train_enc,y_train_enc)

yp_train_enc = knn.predict(X_train_enc)
yp_test_enc = knn.predict(X_test_enc)

en_time = time.time()
print('Total time: {:.2f}s'.format(en_time-st_time))

evaluation_parametrics(y_train_enc,yp_train_enc,y_test_enc,yp_test_enc)
plot_confusion_matrix(knn,X_test_enc, y_test_enc)

dict1 = create_dict(knn, "K Nearest Neighbor Classifier", y_train_enc, yp_train_enc, y_test_enc, yp_test_enc)
dict.update(dict1)

plot_learning_curve(knn, X = X_train_enc, y = y_train_enc, 
                    title =  "Learning Curves (Knn Classifier)", 
                    train_sizes=np.linspace(0.1, 1.0, 5))
```

    Total time: 0.06s
    --------------------------------------------------------------------------
    Classification Report for Train Data
                  precision    recall  f1-score   support
    
               0       0.94      0.97      0.96       296
               1       0.96      0.84      0.90        51
               2       0.99      0.99      0.99       900
               3       1.00      0.90      0.95        49
    
        accuracy                           0.98      1296
       macro avg       0.97      0.93      0.95      1296
    weighted avg       0.98      0.98      0.98      1296
    
    Classification Report for Test Data
                  precision    recall  f1-score   support
    
               0       0.86      0.94      0.90        88
               1       0.92      0.67      0.77        18
               2       0.98      0.98      0.98       310
               3       1.00      0.81      0.90        16
    
        accuracy                           0.95       432
       macro avg       0.94      0.85      0.89       432
    weighted avg       0.95      0.95      0.95       432
    
    --------------------------------------------------------------------------
    Accuracy on Train Data is: 0.98
    Accuracy on Test Data is: 0.95
    --------------------------------------------------------------------------
    Precision on Train Data is: 0.98
    Precision on Test Data is: 0.95
    --------------------------------------------------------------------------
    Recall on Train Data is: 0.98
    Recall on Test Data is: 0.95
    --------------------------------------------------------------------------
    F1 Score on Train Data is: 0.98
    F1 Score on Test Data is: 0.95
    --------------------------------------------------------------------------


    /Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <module 'matplotlib.pyplot' from '/Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/matplotlib/pyplot.py'>




![png](Car%20Evaluation_files/Car%20Evaluation_105_3.png)



![png](Car%20Evaluation_files/Car%20Evaluation_105_4.png)


## Naive Bayes Classifier


```python
# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

st_time = time.time()

gnb.fit(X_train_enc,y_train_enc)

yp_train_enc = gnb.predict(X_train_enc)
yp_test_enc = gnb.predict(X_test_enc)

en_time = time.time()
print('Total time: {:.2f}s'.format(en_time-st_time))

evaluation_parametrics(y_train_enc,yp_train_enc,y_test_enc,yp_test_enc)
plot_confusion_matrix(gnb,X_test_enc, y_test_enc)

dict1 = create_dict(gnb, "Naive Bayes Classifier", y_train_enc, yp_train_enc, y_test_enc, yp_test_enc)
dict.update(dict1)

plot_learning_curve(gnb, X = X_train_enc, y = y_train_enc, 
                    title =  "Learning Curves (Naive Bayes Classifier)", 
                    train_sizes=np.linspace(0.1, 1.0, 5))
```

    Total time: 0.00s
    --------------------------------------------------------------------------
    Classification Report for Train Data
                  precision    recall  f1-score   support
    
               0       0.67      0.23      0.35       296
               1       0.55      0.24      0.33        51
               2       0.87      0.87      0.87       900
               3       0.18      1.00      0.30        49
    
        accuracy                           0.70      1296
       macro avg       0.57      0.58      0.46      1296
    weighted avg       0.78      0.70      0.71      1296
    
    Classification Report for Test Data
                  precision    recall  f1-score   support
    
               0       0.58      0.20      0.30        88
               1       0.50      0.17      0.25        18
               2       0.87      0.87      0.87       310
               3       0.19      1.00      0.32        16
    
        accuracy                           0.71       432
       macro avg       0.54      0.56      0.44       432
    weighted avg       0.77      0.71      0.71       432
    
    --------------------------------------------------------------------------
    Accuracy on Train Data is: 0.7
    Accuracy on Test Data is: 0.71
    --------------------------------------------------------------------------
    Precision on Train Data is: 0.78
    Precision on Test Data is: 0.77
    --------------------------------------------------------------------------
    Recall on Train Data is: 0.7
    Recall on Test Data is: 0.71
    --------------------------------------------------------------------------
    F1 Score on Train Data is: 0.71
    F1 Score on Test Data is: 0.71
    --------------------------------------------------------------------------


    /Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <module 'matplotlib.pyplot' from '/Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/matplotlib/pyplot.py'>




![png](Car%20Evaluation_files/Car%20Evaluation_107_3.png)



![png](Car%20Evaluation_files/Car%20Evaluation_107_4.png)


## Random Forest Classifier


```python
rf = RandomForestClassifier(max_depth = 7,random_state = 48) # Keeping max_depth = 7 same as DT

st_time = time.time()

rf.fit(X_train_enc,y_train_enc)

yp_train_enc = rf.predict(X_train_enc)
yp_test_enc = rf.predict(X_test_enc)

en_time = time.time()
print('Total time: {:.2f}s'.format(en_time-st_time))

evaluation_parametrics(y_train_enc,yp_train_enc,y_test_enc,yp_test_enc)
plot_confusion_matrix(rf,X_test_enc, y_test_enc)

dict1 = create_dict(rf, "Random Forest Classifier", y_train_enc, yp_train_enc, y_test_enc, yp_test_enc)
dict.update(dict1)

plot_learning_curve(rf, X = X_train_enc, y = y_train_enc, 
                    title =  "Learning Curves (Random Forest Classifier)", 
                    train_sizes=np.linspace(0.1, 1.0, 5))
```

    Total time: 0.18s
    --------------------------------------------------------------------------
    Classification Report for Train Data
                  precision    recall  f1-score   support
    
               0       0.92      0.99      0.95       296
               1       0.93      0.84      0.89        51
               2       1.00      0.99      0.99       900
               3       0.93      0.76      0.83        49
    
        accuracy                           0.97      1296
       macro avg       0.94      0.89      0.92      1296
    weighted avg       0.98      0.97      0.97      1296
    
    Classification Report for Test Data
                  precision    recall  f1-score   support
    
               0       0.81      0.93      0.87        88
               1       0.86      0.67      0.75        18
               2       0.99      0.96      0.97       310
               3       0.93      0.88      0.90        16
    
        accuracy                           0.94       432
       macro avg       0.90      0.86      0.87       432
    weighted avg       0.94      0.94      0.94       432
    
    --------------------------------------------------------------------------
    Accuracy on Train Data is: 0.97
    Accuracy on Test Data is: 0.94
    --------------------------------------------------------------------------
    Precision on Train Data is: 0.98
    Precision on Test Data is: 0.94
    --------------------------------------------------------------------------
    Recall on Train Data is: 0.97
    Recall on Test Data is: 0.94
    --------------------------------------------------------------------------
    F1 Score on Train Data is: 0.97
    F1 Score on Test Data is: 0.94
    --------------------------------------------------------------------------


    /Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <module 'matplotlib.pyplot' from '/Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/matplotlib/pyplot.py'>




![png](Car%20Evaluation_files/Car%20Evaluation_109_3.png)



![png](Car%20Evaluation_files/Car%20Evaluation_109_4.png)


## Linear SVC Classifier


```python
svm = LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-4, C=0.1)

st_time = time.time()
svm.fit(X_train_enc,y_train_enc)

yp_train_enc = svm.predict(X_train_enc)
yp_test_enc = svm.predict(X_test_enc)

en_time = time.time()
print('Total time: {:.2f}s'.format(en_time-st_time))

evaluation_parametrics(y_train_enc,yp_train_enc,y_test_enc,yp_test_enc)
plot_confusion_matrix(svm,X_test_enc, y_test_enc)

dict1 = create_dict(svm, "Linear SVC", y_train_enc, yp_train_enc, y_test_enc, yp_test_enc)
dict.update(dict1)

plot_learning_curve(svm, X = X_train_enc, y = y_train_enc, 
                    title =  "Learning Curves (Linear SVC Classifier)", 
                    train_sizes=np.linspace(0.1, 1.0, 5))

```

    Total time: 0.01s
    --------------------------------------------------------------------------
    Classification Report for Train Data
                  precision    recall  f1-score   support
    
               0       0.78      0.66      0.71       296
               1       0.48      0.94      0.63        51
               2       0.91      0.89      0.90       900
               3       0.61      0.76      0.67        49
    
        accuracy                           0.84      1296
       macro avg       0.69      0.81      0.73      1296
    weighted avg       0.85      0.84      0.84      1296
    
    Classification Report for Test Data
                  precision    recall  f1-score   support
    
               0       0.74      0.64      0.68        88
               1       0.50      1.00      0.67        18
               2       0.90      0.88      0.89       310
               3       0.59      0.62      0.61        16
    
        accuracy                           0.83       432
       macro avg       0.68      0.79      0.71       432
    weighted avg       0.84      0.83      0.83       432
    
    --------------------------------------------------------------------------
    Accuracy on Train Data is: 0.84
    Accuracy on Test Data is: 0.83
    --------------------------------------------------------------------------
    Precision on Train Data is: 0.85
    Precision on Test Data is: 0.84
    --------------------------------------------------------------------------
    Recall on Train Data is: 0.84
    Recall on Test Data is: 0.83
    --------------------------------------------------------------------------
    F1 Score on Train Data is: 0.84
    F1 Score on Test Data is: 0.83
    --------------------------------------------------------------------------


    /Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <module 'matplotlib.pyplot' from '/Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/matplotlib/pyplot.py'>




![png](Car%20Evaluation_files/Car%20Evaluation_111_3.png)



![png](Car%20Evaluation_files/Car%20Evaluation_111_4.png)


## Gradient Boosting


```python
gb_model = GradientBoostingClassifier(n_estimators=50, max_depth=10)

st_time = time.time()
gb_model.fit(X_train_enc,y_train_enc)

yp_train_enc = gb_model.predict(X_train_enc)
yp_test_enc = gb_model.predict(X_test_enc)

en_time = time.time()
print('Total time: {:.2f}s'.format(en_time-st_time))

evaluation_parametrics(y_train_enc,yp_train_enc,y_test_enc,yp_test_enc)
plot_confusion_matrix(svm,X_test_enc, y_test_enc)

dict1 = create_dict(gb_model, "Gradient Boosting", y_train_enc, yp_train_enc, y_test_enc, yp_test_enc)
dict.update(dict1)

plot_learning_curve(gb_model, X = X_train_enc, y = y_train_enc, 
                    title =  "Learning Curves (Gradient Boosting Classifier)", 
                    train_sizes=np.linspace(0.1, 1.0, 5))
```

    Total time: 1.03s
    --------------------------------------------------------------------------
    Classification Report for Train Data
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       296
               1       1.00      1.00      1.00        51
               2       1.00      1.00      1.00       900
               3       1.00      1.00      1.00        49
    
        accuracy                           1.00      1296
       macro avg       1.00      1.00      1.00      1296
    weighted avg       1.00      1.00      1.00      1296
    
    Classification Report for Test Data
                  precision    recall  f1-score   support
    
               0       0.92      0.98      0.95        88
               1       0.93      0.78      0.85        18
               2       0.99      0.99      0.99       310
               3       1.00      0.94      0.97        16
    
        accuracy                           0.97       432
       macro avg       0.96      0.92      0.94       432
    weighted avg       0.97      0.97      0.97       432
    
    --------------------------------------------------------------------------
    Accuracy on Train Data is: 1.0
    Accuracy on Test Data is: 0.97
    --------------------------------------------------------------------------
    Precision on Train Data is: 1.0
    Precision on Test Data is: 0.97
    --------------------------------------------------------------------------
    Recall on Train Data is: 1.0
    Recall on Test Data is: 0.97
    --------------------------------------------------------------------------
    F1 Score on Train Data is: 1.0
    F1 Score on Test Data is: 0.97
    --------------------------------------------------------------------------


    /Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <module 'matplotlib.pyplot' from '/Users/bhaskarroy/opt/anaconda3/lib/python3.8/site-packages/matplotlib/pyplot.py'>




![png](Car%20Evaluation_files/Car%20Evaluation_113_3.png)



![png](Car%20Evaluation_files/Car%20Evaluation_113_4.png)



```python
st_time = time.time()
en_time = time.time()
print('Total time: {:.2f}s'.format(en_time-st_time))
```

    Total time: 0.00s


## Listing the performance from all the models
# Accuracy
  print("Accuracy on Train Data is: {}".format(round(accuracy_score(y_train,yp_train),2)))
  print("Accuracy on Test Data is: {}".format(round(accuracy_score(y_test,yp_test),2)))
  print("--------------------------------------------------------------------------")
  # Precision
  print("Precision on Train Data is: {}".format(round(precision_score(y_train,yp_train,average = "weighted"),2)))
  print("Precision on Test Data is: {}".format(round(precision_score(y_test,yp_test,average = "weighted"),2)))
  print("--------------------------------------------------------------------------")
  # Recall 
  print("Recall on Train Data is: {}".format(round(recall_score(y_train,yp_train,average = "weighted"),2)))
  print("Recall on Test Data is: {}".format(round(recall_score(y_test,yp_test,average = "weighted"),2)))
  print("--------------------------------------------------------------------------")
  # F1 Score
  print("F1 Score on Train Data is: {}".format(round(f1_score(y_train,yp_train,average = "weighted"),2)))
  print("F1 Score on Test Data is: {}".format(round(f1_score(y_test,yp_test,average = "weighted"),2)))
  print("--------------------------------------------------------------------------")

```python
pd.DataFrame.from_dict({(i,j): dict[i][j] 
                           for i in dict.keys() 
                           for j in dict[i].keys()},
                       orient='index')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Train</th>
      <th>Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Logistic Regression Classifier</th>
      <th>F1</th>
      <td>0.83</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>Recall</th>
      <td>0.83</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>Precision</th>
      <td>0.83</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Decision Tree Classifier</th>
      <th>F1</th>
      <td>0.94</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>Recall</th>
      <td>0.94</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Linear SVC</th>
      <th>Recall</th>
      <td>0.84</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>Precision</th>
      <td>0.85</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Gradient Boosting</th>
      <th>F1</th>
      <td>1.00</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>Recall</th>
      <td>1.00</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>Precision</th>
      <td>1.00</td>
      <td>0.97</td>
    </tr>
  </tbody>
</table>
<p>21 rows × 2 columns</p>
</div>




```python
user_ids = []
frames = []

for user_id, d in dict.items():
    user_ids.append(user_id)
    frames.append(pd.DataFrame.from_dict(d, orient='columns'))

df = pd.concat(frames, keys=user_ids)
df.unstack(level = -1).style.background_gradient(cmap='Blues')
```




<style  type="text/css" >
#T_84f8d_row0_col0{
            background-color:  #8fc2de;
            color:  #000000;
        }#T_84f8d_row0_col1{
            background-color:  #a8cee4;
            color:  #000000;
        }#T_84f8d_row0_col2{
            background-color:  #87bddc;
            color:  #000000;
        }#T_84f8d_row0_col3{
            background-color:  #9ac8e0;
            color:  #000000;
        }#T_84f8d_row0_col4{
            background-color:  #cadef0;
            color:  #000000;
        }#T_84f8d_row0_col5{
            background-color:  #d9e8f5;
            color:  #000000;
        }#T_84f8d_row1_col0{
            background-color:  #1865ac;
            color:  #f1f1f1;
        }#T_84f8d_row1_col1,#T_84f8d_row1_col3{
            background-color:  #0e58a2;
            color:  #f1f1f1;
        }#T_84f8d_row1_col2{
            background-color:  #1764ab;
            color:  #f1f1f1;
        }#T_84f8d_row1_col4{
            background-color:  #2676b8;
            color:  #000000;
        }#T_84f8d_row1_col5,#T_84f8d_row4_col5{
            background-color:  #0d57a1;
            color:  #f1f1f1;
        }#T_84f8d_row2_col0,#T_84f8d_row2_col2{
            background-color:  #084285;
            color:  #f1f1f1;
        }#T_84f8d_row2_col1,#T_84f8d_row2_col3{
            background-color:  #084488;
            color:  #f1f1f1;
        }#T_84f8d_row2_col4,#T_84f8d_row4_col4{
            background-color:  #08488e;
            color:  #f1f1f1;
        }#T_84f8d_row2_col5,#T_84f8d_row4_col2{
            background-color:  #084a91;
            color:  #f1f1f1;
        }#T_84f8d_row3_col0,#T_84f8d_row3_col1,#T_84f8d_row3_col2,#T_84f8d_row3_col3,#T_84f8d_row3_col4,#T_84f8d_row3_col5{
            background-color:  #f7fbff;
            color:  #000000;
        }#T_84f8d_row4_col0{
            background-color:  #084b93;
            color:  #f1f1f1;
        }#T_84f8d_row4_col1,#T_84f8d_row4_col3{
            background-color:  #084e98;
            color:  #f1f1f1;
        }#T_84f8d_row5_col0{
            background-color:  #81badb;
            color:  #000000;
        }#T_84f8d_row5_col1,#T_84f8d_row5_col3{
            background-color:  #7ab6d9;
            color:  #000000;
        }#T_84f8d_row5_col2{
            background-color:  #79b5d9;
            color:  #000000;
        }#T_84f8d_row5_col4{
            background-color:  #b0d2e7;
            color:  #000000;
        }#T_84f8d_row5_col5{
            background-color:  #a6cee4;
            color:  #000000;
        }#T_84f8d_row6_col0,#T_84f8d_row6_col1,#T_84f8d_row6_col2,#T_84f8d_row6_col3,#T_84f8d_row6_col4,#T_84f8d_row6_col5{
            background-color:  #08306b;
            color:  #f1f1f1;
        }</style><table id="T_84f8d_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" colspan="2">F1</th>        <th class="col_heading level0 col2" colspan="2">Recall</th>        <th class="col_heading level0 col4" colspan="2">Precision</th>    </tr>    <tr>        <th class="blank level1" ></th>        <th class="col_heading level1 col0" >Train</th>        <th class="col_heading level1 col1" >Test</th>        <th class="col_heading level1 col2" >Train</th>        <th class="col_heading level1 col3" >Test</th>        <th class="col_heading level1 col4" >Train</th>        <th class="col_heading level1 col5" >Test</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_84f8d_level0_row0" class="row_heading level0 row0" >Logistic Regression Classifier</th>
                        <td id="T_84f8d_row0_col0" class="data row0 col0" >0.830000</td>
                        <td id="T_84f8d_row0_col1" class="data row0 col1" >0.800000</td>
                        <td id="T_84f8d_row0_col2" class="data row0 col2" >0.830000</td>
                        <td id="T_84f8d_row0_col3" class="data row0 col3" >0.810000</td>
                        <td id="T_84f8d_row0_col4" class="data row0 col4" >0.830000</td>
                        <td id="T_84f8d_row0_col5" class="data row0 col5" >0.800000</td>
            </tr>
            <tr>
                        <th id="T_84f8d_level0_row1" class="row_heading level0 row1" >Decision Tree Classifier</th>
                        <td id="T_84f8d_row1_col0" class="data row1 col0" >0.940000</td>
                        <td id="T_84f8d_row1_col1" class="data row1 col1" >0.930000</td>
                        <td id="T_84f8d_row1_col2" class="data row1 col2" >0.940000</td>
                        <td id="T_84f8d_row1_col3" class="data row1 col3" >0.930000</td>
                        <td id="T_84f8d_row1_col4" class="data row1 col4" >0.940000</td>
                        <td id="T_84f8d_row1_col5" class="data row1 col5" >0.940000</td>
            </tr>
            <tr>
                        <th id="T_84f8d_level0_row2" class="row_heading level0 row2" >K Nearest Neighbor Classifier</th>
                        <td id="T_84f8d_row2_col0" class="data row2 col0" >0.980000</td>
                        <td id="T_84f8d_row2_col1" class="data row2 col1" >0.950000</td>
                        <td id="T_84f8d_row2_col2" class="data row2 col2" >0.980000</td>
                        <td id="T_84f8d_row2_col3" class="data row2 col3" >0.950000</td>
                        <td id="T_84f8d_row2_col4" class="data row2 col4" >0.980000</td>
                        <td id="T_84f8d_row2_col5" class="data row2 col5" >0.950000</td>
            </tr>
            <tr>
                        <th id="T_84f8d_level0_row3" class="row_heading level0 row3" >Naive Bayes Classifier</th>
                        <td id="T_84f8d_row3_col0" class="data row3 col0" >0.710000</td>
                        <td id="T_84f8d_row3_col1" class="data row3 col1" >0.710000</td>
                        <td id="T_84f8d_row3_col2" class="data row3 col2" >0.700000</td>
                        <td id="T_84f8d_row3_col3" class="data row3 col3" >0.710000</td>
                        <td id="T_84f8d_row3_col4" class="data row3 col4" >0.780000</td>
                        <td id="T_84f8d_row3_col5" class="data row3 col5" >0.770000</td>
            </tr>
            <tr>
                        <th id="T_84f8d_level0_row4" class="row_heading level0 row4" >Random Forest Classifier</th>
                        <td id="T_84f8d_row4_col0" class="data row4 col0" >0.970000</td>
                        <td id="T_84f8d_row4_col1" class="data row4 col1" >0.940000</td>
                        <td id="T_84f8d_row4_col2" class="data row4 col2" >0.970000</td>
                        <td id="T_84f8d_row4_col3" class="data row4 col3" >0.940000</td>
                        <td id="T_84f8d_row4_col4" class="data row4 col4" >0.980000</td>
                        <td id="T_84f8d_row4_col5" class="data row4 col5" >0.940000</td>
            </tr>
            <tr>
                        <th id="T_84f8d_level0_row5" class="row_heading level0 row5" >Linear SVC</th>
                        <td id="T_84f8d_row5_col0" class="data row5 col0" >0.840000</td>
                        <td id="T_84f8d_row5_col1" class="data row5 col1" >0.830000</td>
                        <td id="T_84f8d_row5_col2" class="data row5 col2" >0.840000</td>
                        <td id="T_84f8d_row5_col3" class="data row5 col3" >0.830000</td>
                        <td id="T_84f8d_row5_col4" class="data row5 col4" >0.850000</td>
                        <td id="T_84f8d_row5_col5" class="data row5 col5" >0.840000</td>
            </tr>
            <tr>
                        <th id="T_84f8d_level0_row6" class="row_heading level0 row6" >Gradient Boosting</th>
                        <td id="T_84f8d_row6_col0" class="data row6 col0" >1.000000</td>
                        <td id="T_84f8d_row6_col1" class="data row6 col1" >0.970000</td>
                        <td id="T_84f8d_row6_col2" class="data row6 col2" >1.000000</td>
                        <td id="T_84f8d_row6_col3" class="data row6 col3" >0.970000</td>
                        <td id="T_84f8d_row6_col4" class="data row6 col4" >1.000000</td>
                        <td id="T_84f8d_row6_col5" class="data row6 col5" >0.970000</td>
            </tr>
    </tbody></table>



# Conclusion

- Decision Tree Classifier, K Nearest Neighbours, Random Forest Classifiers, Gradient Boosting have performed high in F1, Recall and Precision measures for both test and train methods.
- Gradient Boosting has given the best performance. However, note that this is accompanied with lack of interpretability and explainability.
- Logistic Regression, Linear SVC followed by Naive Bayes have low scores on the evaluation metrics. 

### Resources to follow for imbalanced learning 

- http://www.svds.com/learning-imbalanced-classes/ (The Applied Data Science Workshop - Second Edition Get started with the applications of data science and techniques to explore and assess data effectively by Alex Galea)
- Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido
- Machine Learning Engineering by Andriy Burkov
- https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
- Comparing Oversampling Techniques to Handle the Class Imbalance Problem: A Customer Churn Prediction Case Study - ADNAN AMIN1, SAJID ANWAR1, AWAIS ADNAN1, MUHAMMAD NAWAZ1, NEWTON HOWARD2, JUNAID QADIR3, (Senior Member, IEEE), AHMAD HAWALAH4, AND AMIR HUSSAIN5, (Senior Member, IEEE)
- Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research, 18 (17), 1-5.
- Learning from imbalanced data: open challenges and future directions by Bartosz Krawczyk1
- Evaluating machine learning models - a beginner's guide to key concepts and pitfalls by Zheng, Alice
- Fawcett, Tom. 2006. An introduction to ROC analysis. Pattern Recognition Letters 27 (8): 861–874.
- https://www.codespeedy.com/multiclass-classification-using-scikit-learn/
- The Essentials of Machine Learning in Finance and Accounting, Chapter 11, Handling class imbalance data in business domain - Authored by Md. Shajalal, Mohammad Zoynul Abedin and Mohammed Mohi Uddin
- Learning from imbalanced datasets* by Alberto Fernandez et al
- https://win-vector.com/2015/02/27/does-balancing-classes-improve-classifier-performance/
- https://datascientistdiary.com/index.php/2021/09/02/how-to-handle-imbalanced-data-example-in-r/



#### Datasets 
- Breast Cancer Wisconsin dataset
- Credit Card Fraud detection Kaggle dataset

#### Real life scenarios with data imbalance
Learning from imbalanced data: open challenges and future directions by Bartosz Krawczyk

| Application Area                    | Problem Description                                          |
|:---|:---|
|Activity Recognition | Detection of rare or less-frequent activities (multi-class problem)|
|Behavior Analysis| Recognition of dangerous behavior (binary problem)|
|Cancer Malignancy grading|Analyzing the cancer severity (binary and multi-class problem)|
|Hyperspectral data analysis |Classification of varying areas in multi-dimensional images (multi-class problem)|
|Industrial Systems monitoring|Fault detection in industrial machinery (binary problem)|
|Sentiment analysis|Emotion and temper recognition in text (binary and multi-class problem)|
|Software defect prediction|Recognition of errors in code blocks (binary problem)|
|Target detection |Classification of specified targets appearing with varied frequency (multi-class problem)|
|Text mining |Detecting relations in literature (binary problem)|
|Video mining|Recognizing objects and actions in video sequences (binary and multi-class problem)|






