---
layout : single
title : ML Projects
toc: true
toc_sticky: true
author_profile: true
---


{:.no_toc}  


* TOC
{:toc}  

## Credit screening (UCI)
This dataset has been downloaded from UC Irvine Machine Learning Repository. [Link](https://archive.ics.uci.edu/ml/datasets/Credit+Approval)

This dataset is regarding credit card applications.
The target variable/label is whether the application has been granted credit or not.
All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.

The objective is here to build a classifier model to give **binary output** based on the input attributes.The dataset is balanced.

We have spot-checked multiple classifier algorithms using both simple train/test split as well as k fold cross validation.
Accuracy was chosen as the evaluation metric as the dataset is balanced. Two algorithms were shortlisted and taken forward for hyper-parameter tuning.

Check the [notebook](/Pages/ML Projects/Credit Screening/Credit Screening_01.md)  
Check [classic jupyter version](/Pages/ML Projects/Credit Screening.html)


## Sales Transaction Weekly of 800 products (UCI)

This dataset has been downloaded from UC Irvine Machine Learning Repository. [Link1](https://archive.ics.uci.edu/ml/machine-learning-databases/00396/) [Link2](https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly)  

We have the sales quantity of 811 products over a period of 52 weeks.  
The available information can be used for two kinds of problems.

- Firstly, we can create a time series forecasting model to predict the sales qty for next week.
- Secondly, we can build a clustering model to identify natural groupings of the product based on the weekly sales pattern.

Check the [clustering notebook](/Pages/ML Projects/Sales_Transactions_Dataset_Weekly_Clustering/Sales_Transactions_Dataset_Weekly_Clustering.md)  
Check the [regression notebook](/assets/scripts/Sales_Transactions_Dataset_Weekly_Clustering/Sales_Transactions_Dataset_Weekly_Clustering)

## Car Evaluation

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

Check the [notebook](Pages/ML Projects/Car Evaluation/Car Evaluation.md)


## Online Purchasing Intention Dataset




## LasVegas Strip Dataset




## Telecom Churn






## Flight Fare Prediction
