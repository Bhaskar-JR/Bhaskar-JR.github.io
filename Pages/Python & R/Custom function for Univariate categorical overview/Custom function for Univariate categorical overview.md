---
layout : single
title : "Custom function for Univariate categorical overview"
toc: true
toc_label: "Custom function for Univariate categorical overview"
toc_sticky: true
author_profile: true
comments: true
---  

# Introduction  

[jupyter notebook file](Custom function for EDA categorical overview.ipynb)  
[jupyter notebook html](Custom function for EDA categorical overview.html)

Quite often we start out with a dataframe and  want a quick overview of categorical features. We ideally want to check the number of categories/levels of each categorical attribute. We want to check the count/proportion for each of the discrete labels/categories of a categorical feature.  

This is such a common usecase that I have created a [**custom module**](UVA_category.py) in Python.  
The output will be barplots of all the categories and annotating the axes with count/proportion of the categorical levels. This will become amply clear on seeing the output.  

While calling the function, I have kept the provision to pass a number of keyword arguments
that would allow immense flexibility. This would also take into account multiple scenarios.  

The keyword arguments fall under following categories :
- plot layout parameters
- barwidth parameters  
- font family  
- font size parameter for xtick-labels, y-tick labels, axis labels
- Maximum Number of categorical levels per attribute to display  
- Configuring the text annotation of categorical levelwise count/proportion

The function requires atleast a dataframe to retun a output.


# Custom function for easy and efficient analysis of categorical univariate


```python
# Importing the libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
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
  - spine_linewidth : width of spine

  - change_ratio : default value is 0.7.
  If the width of each bar exceeds the barwidth_threshold,then the change_ratio
  is the proportion by which to change the bar width.
  - barwidth_threshold : default value is 0.2.
  Refers to proportion of the total height of the axes.
  Will be used to compare whether the bar width exceeds the barwidth_threshold.

  - axlabel_fntsize : Fontsize of x axis and y axis labels
  - infofntsize : Fontsize in the info of unique value counts
  - ax_xticklabel_fntsize : fontsize of x-axis tick labels
  - ax_yticklabel_fntsize : fontsize of y-axis tick labels
  - infofntfamily : Font family of info of unique value counts.
  Choose font family belonging to Monospace for multiline alignment.
  Some choices are : 'Consolas', 'Courier','Courier New', 'Lucida Sans Typewriter','Lucidatypewriter','Andale Mono'
  https://www.tutorialbrain.com/css_tutorial/css_font_family_list/
  - max_val_counts : Number of unique values for which count should be displayed
  - nspaces : Length of each line for the multiline strings in the info area for value_counts
  - ncountspaces : Length allocated to the count value for the unique values in the info area
  - show_percentage : Whether to show percentage of total for each unique value count
  Also check link for formatting syntax :
      https://docs.python.org/3/library/string.html#formatspec
      <Format Specification for minilanguage>

      https://mkaz.blog/code/python-string-format-cookbook/#f-strings
      https://pyformat.info/#number
      https://stackoverflow.com/questions/3228865/
  '''

  # *args and **kwargs are special keyword which allows function to take variable length argument.
  # *args passes variable number of non-keyworded arguments and on which operation of the tuple can be performed.
  # **kwargs passes variable number of keyword arguments dictionary to function on which operation of a dictionary can be performed.
  # *args and **kwargs make the function flexible.

  import textwrap
  data = data_frame.copy(deep = True)
  # Using dictionary with default values of keywrod arguments
  params_plot = dict(colcount = 2, colwidth = 7, rowheight = 4, \
                     spine_linewidth = 1, normalize = False, sort_by = "Values")
  params_bar= dict(change_ratio = 1, barwidth_threshold = 0.2)
  params_fontsize =  dict(axlabel_fntsize = 10,
                          ax_xticklabel_fntsize = 8,
                          ax_yticklabel_fntsize = 8,
                          infofntsize = 10)
  params_fontfamily = dict(infofntfamily = 'Andale Mono')
  params_max_val_counts = dict(max_val_counts = 10)
  params_infospaces = dict(nspaces = 10, ncountspaces = 4)
  params_show_percentage = dict(show_percentage = True)



  # Updating the dictionary with parameter values passed while calling the function
  params_plot.update((k, v) for k, v in kargs.items() if k in params_plot)
  params_bar.update((k, v) for k, v in kargs.items() if k in params_bar)
  params_fontsize.update((k, v) for k, v in kargs.items() if k in params_fontsize)
  params_fontfamily.update((k, v) for k, v in kargs.items() if k in params_fontfamily)
  params_max_val_counts.update((k, v) for k, v in kargs.items() if k in params_max_val_counts)
  params_infospaces.update((k, v) for k, v in kargs.items() if k in params_infospaces)
  params_show_percentage.update((k, v) for k, v in kargs.items() if k in params_show_percentage)

  #params = dict(**params_plot, **params_fontsize)

  # Initialising all the possible keyword arguments of doc string with updated values
  colcount = params_plot['colcount']
  colwidth = params_plot['colwidth']
  rowheight = params_plot['rowheight']
  normalize = params_plot['normalize']
  sort_by = params_plot['sort_by']
  spine_linewidth  =  params_plot['spine_linewidth']

  change_ratio = params_bar['change_ratio']
  barwidth_threshold = params_bar['barwidth_threshold']

  axlabel_fntsize = params_fontsize['axlabel_fntsize']
  ax_xticklabel_fntsize = params_fontsize['ax_xticklabel_fntsize']
  ax_yticklabel_fntsize = params_fontsize['ax_yticklabel_fntsize']
  infofntsize = params_fontsize['infofntsize']
  infofntfamily = params_fontfamily['infofntfamily']
  max_val_counts =  params_max_val_counts['max_val_counts']
  nspaces = params_infospaces['nspaces']
  ncountspaces = params_infospaces['ncountspaces']
  show_percentage = params_show_percentage['show_percentage']


  if len(var_group) == 0:
        var_group = data.select_dtypes(exclude = ['number']).columns.to_list()
  print(f'Categorical features : {var_group} \n')

  import matplotlib.pyplot as plt
  plt.rcdefaults()
  # setting figure_size
  size = len(var_group)
  #rowcount = 1
  #colcount = size//rowcount+(size%rowcount != 0)*1


  colcount = colcount
  #print(colcount)
  rowcount = size//colcount+(size%colcount != 0)*1

  fig = plt.figure(figsize = (colwidth*colcount,rowheight*rowcount), dpi = 150)


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
        norm_count = data[i].value_counts(normalize = normalize). \
        sort_values(ascending = False)

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
    ax.set_yticklabels([textwrap.fill(str(e), 20) for e in norm_count.index],
                       fontsize = ax_yticklabel_fntsize)

    [labels.set(size = ax_xticklabel_fntsize) for labels in ax.get_xticklabels()]

    for key, _  in ax.spines._dict.items():
        ax.spines._dict[key].set_linewidth(spine_linewidth)

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
        Also check link for formatting syntax :
            https://docs.python.org/3/library/string.html#formatspec
            <Format Specification for minilanguage>

            https://mkaz.blog/code/python-string-format-cookbook/#f-strings
            https://pyformat.info/#number
            https://stackoverflow.com/questions/3228865/
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

    # Change Bar height if each bar height exceeds barwidth_threshold (default = 20%) of the axes y length
    from eda.axes_utils import Change_barWidth
    if ax.patches[1].get_height() >= barwidth_threshold*(ax.get_ylim()[1]-ax.get_ylim()[0]):
        Change_barWidth(ax.patches, change_ratio= change_ratio, orient = 'h')

  fig.tight_layout()
  return fig
```

# Implementing the custom function for overview of categorical features

We will showcase the use of custom function on following three datasets :  
- [simulated retail store dataset](#simulated_retail_store_dataset)
- [simulated customer subscription dataset](#simulated_customer_subscription_dataset)  
- [online shoppers purchasing intention dataset](#Online_Shoppers_Purchasing_Intention_Dataset)

<a id='simulated_retail_store_dataset'></a>
# Simulated retail store dataset  

The simulation code for [retail store dataset](retail_store_dataset.py) has been taken from 'Python for Marketing Research and Analytics' authored by Jason S. Schwarz, Chris Chapman, Elea McDonnell Feit. The book has well explained sections on simulating data for various usecases. It also helps building both intuition and practical acumen in using different probability distributions depending on attribute type.  

Another recommended reading for simulating datasets in python is 'Practical Time Series Analysis-Prediction with Statistics and Machine Learning' by Aileen Nelson.

## Creating simulated retail store dataset


```python
def retail_store_data():
    '''
    This dataset represents observations of total sales by week
    for two competing products at a chain of stores.

    We create a data structure that will hold the data,
    a simulation of sales for the two products in 20 stores over
    2 years, with price and promotion status.

    # Constants
        N_STORES = 20
        N_WEEKS = 104

    The code has been taken from the book :
        • 'Python for Marketing Research and Analytics'
        by Jason S. Schwarz,Chris Chapman, Elea McDonnell Feit

    Additional links :
        • An information website: https://python-marketing-research.github.io
        • A Github repository: https://github.com/python-marketing-research/python-marketing-research-1ed
        • The Colab Github browser: https://colab.sandbox.google.com/github/python-marketing-research/python-marketing-research-1ed



    '''
    import pandas as pd
    import numpy as np
    # Constants
    N_STORES = 20
    N_WEEKS = 104

    # create a dataframe of initially missing values to hold the data
    columns = ('store_num', 'year', 'week', 'p1_sales', 'p2_sales',
               'p1_price', 'p2_price', 'p1_promo', 'p2_promo', 'country')
    n_rows = N_STORES * N_WEEKS

    store_sales = pd.DataFrame(np.empty(shape=(n_rows, 10)),
                               columns=columns)

    # Create store Ids
    store_numbers = range(101, 101 + N_STORES)

    # assign each store a country
    store_country = dict(zip(store_numbers,
                                     ['USA', 'USA', 'USA', 'DEU', 'DEU', 'DEU',
                                      'DEU', 'DEU', 'GBR', 'GBR', 'GBR', 'BRA',
                                      'BRA', 'JPN', 'JPN', 'JPN', 'JPN', 'AUS',
                                      'CHN', 'CHN']))

    # filling in the store_sales dataframe:
    i = 0
    for store_num in store_numbers:
        for year in [1, 2]:
            for week in range(1, 53):
                store_sales.loc[i, 'store_num'] = store_num
                store_sales.loc[i, 'year'] = year
                store_sales.loc[i, 'week'] = week
                store_sales.loc[i, 'country'] = store_country[store_num]
                i += 1

    # setting the variable types correctly using the astype method
    store_sales.loc[:,'country'] = store_sales['country'].astype( pd.CategoricalDtype())
    store_sales.loc[:,'store_num'] = store_sales['store_num'].astype(pd.CategoricalDtype())
    #print(store_sales['store_num'].head())
    #print(store_sales['country'].head())


    # For each store in each week,
    # we want to randomly determine whether each product was promoted or not.
    # We randomly assign 10% likelihood of promotion for product 1, and
    # 15% likelihood for product 2.

    # setting the random seed
    np.random.seed(37204)

    # 10% promoted
    store_sales.p1_promo = np.random.binomial(n=1, p=0.1, size=n_rows)

    # 15% promoted
    store_sales.p2_promo = np.random.binomial(n=1, p=0.15, size=n_rows)


    # we set a price for each product in each row of the data.
    # We suppose that each product is sold at one of five distinct \
    # price points ranging from $2.19 to $3.19 overall. We randomly \
    # draw a price for each week by defining a vector with the five \
    # price points and using np.random.choice(a, size, replace) to \
    # draw from it as many times as we have rows of data (size=n_rows). \
    # The five prices are sampled many times, so we sample with replacement.

    store_sales.p1_price = np.random.choice([2.19, 2.29, 2.49, 2.79, 2.99],
                                            size=n_rows)
    store_sales.p2_price = np.random.choice([2.29, 2.49, 2.59, 2.99,3.19],
                                            size=n_rows)
    #store_sales.sample(5) # now how does it look?


    # simulate the sales figures for each week. We calculate sales as a \
    # function of the relative prices of the two products along with the  \
    # promotional status of each.

    # Item sales are in unit counts, so we use the Poisson distribution \
    # to generate count data

    # sales data, using poisson (counts) distribution, np.random.poisson()
    # first, the default sales in the absence of promotion
    sales_p1 = np.random.poisson(lam=120, size=n_rows)
    sales_p2 = np.random.poisson(lam=100, size=n_rows)


    # Price effects often follow a logarithmic function rather than a \
    # linear function (Rao 2009)
    # scale sales according to the ratio of log(price)

    log_p1_price = np.log(store_sales.p1_price)
    log_p2_price = np.log(store_sales.p2_price)
    sales_p1 = sales_p1 * log_p2_price/log_p1_price
    sales_p2 = sales_p2 * log_p1_price/log_p2_price

    # We have assumed that sales vary as the inverse ratio of prices.  \
    # That is, sales of Product 1 go up to the degree that  \
    # the log(price) of Product 1 is lower than the log(price) of Product 2.


    # we assume that sales get a 30 or 40% lift when each product is promoted  \
    # in store. We simply multiply the promotional status vector (which comprises all {0, 1} values) \
    # by 0.3 or 0.4 respectively, and then multiply the sales vector by that.


    # final sales get a 30% or 40% lift when promoted
    store_sales.p1_sales = np.floor(sales_p1 *(1 + store_sales.p1_promo * 0.3))
    store_sales.p2_sales = np.floor(sales_p2 *(1 + store_sales.p2_promo * 0.4))

    return store_sales

```


```python
df1 = retail_store_data()
```

## Inspecting the dataframe


```python
df1.head()
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
      <th>store_num</th>
      <th>year</th>
      <th>week</th>
      <th>p1_sales</th>
      <th>p2_sales</th>
      <th>p1_price</th>
      <th>p2_price</th>
      <th>p1_promo</th>
      <th>p2_promo</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>101.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>115.0</td>
      <td>114.0</td>
      <td>2.79</td>
      <td>2.59</td>
      <td>0</td>
      <td>0</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>101.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>131.0</td>
      <td>87.0</td>
      <td>2.49</td>
      <td>2.49</td>
      <td>0</td>
      <td>0</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>101.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>176.0</td>
      <td>74.0</td>
      <td>2.29</td>
      <td>3.19</td>
      <td>0</td>
      <td>0</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>101.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>125.0</td>
      <td>115.0</td>
      <td>2.19</td>
      <td>2.29</td>
      <td>0</td>
      <td>0</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>114.0</td>
      <td>120.0</td>
      <td>2.99</td>
      <td>2.49</td>
      <td>0</td>
      <td>0</td>
      <td>USA</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2080 entries, 0 to 2079
    Data columns (total 10 columns):
     #   Column     Non-Null Count  Dtype   
    ---  ------     --------------  -----   
     0   store_num  2080 non-null   category
     1   year       2080 non-null   float64
     2   week       2080 non-null   float64
     3   p1_sales   2080 non-null   float64
     4   p2_sales   2080 non-null   float64
     5   p1_price   2080 non-null   float64
     6   p2_price   2080 non-null   float64
     7   p1_promo   2080 non-null   int64   
     8   p2_promo   2080 non-null   int64   
     9   country    2080 non-null   category
    dtypes: category(2), float64(6), int64(2)
    memory usage: 135.2 KB



```python
df1.describe()
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
      <th>year</th>
      <th>week</th>
      <th>p1_sales</th>
      <th>p2_sales</th>
      <th>p1_price</th>
      <th>p2_price</th>
      <th>p1_promo</th>
      <th>p2_promo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2080.00000</td>
      <td>2080.00000</td>
      <td>2080.000000</td>
      <td>2080.000000</td>
      <td>2080.000000</td>
      <td>2080.000000</td>
      <td>2080.000000</td>
      <td>2080.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.50000</td>
      <td>26.50000</td>
      <td>132.658173</td>
      <td>100.928365</td>
      <td>2.555721</td>
      <td>2.705769</td>
      <td>0.096154</td>
      <td>0.153846</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.50012</td>
      <td>15.01194</td>
      <td>28.632719</td>
      <td>25.084833</td>
      <td>0.299788</td>
      <td>0.330379</td>
      <td>0.294873</td>
      <td>0.360888</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>72.000000</td>
      <td>50.000000</td>
      <td>2.190000</td>
      <td>2.290000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.00000</td>
      <td>13.75000</td>
      <td>112.000000</td>
      <td>83.000000</td>
      <td>2.290000</td>
      <td>2.490000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.50000</td>
      <td>26.50000</td>
      <td>128.000000</td>
      <td>97.000000</td>
      <td>2.490000</td>
      <td>2.590000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.00000</td>
      <td>39.25000</td>
      <td>150.000000</td>
      <td>116.000000</td>
      <td>2.790000</td>
      <td>2.990000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.00000</td>
      <td>52.00000</td>
      <td>263.000000</td>
      <td>223.000000</td>
      <td>2.990000</td>
      <td>3.190000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Checking null values


```python
df1.isnull().sum()
```




    store_num    0
    year         0
    week         0
    p1_sales     0
    p2_sales     0
    p1_price     0
    p2_price     0
    p1_promo     0
    p2_promo     0
    country      0
    dtype: int64



### Number of unique values for each feature


```python
print(df1.nunique(axis=0))
```

    store_num     20
    year           2
    week          52
    p1_sales     153
    p2_sales     139
    p1_price       5
    p2_price       5
    p1_promo       2
    p2_promo       2
    country        7
    dtype: int64


## EDA overview of categorical features


```python
UVA_category(df1)
```

    Categorical features : ['store_num', 'country']




![png](output_20_1.png)


<a id='simulated_customer_subscription_dataset'></a>
# Simulated customer subscription dataset  

The simulation code for [customer subscription](customer_subscription.py) dataset has been taken from 'Python for Marketing Research and Analytics' authored by Jason S. Schwarz, Chris Chapman, Elea McDonnell Feit. The book has well explained sections on simulating data for various usecases. It also helps building both intuition and practical acumen in using different probability distributions depending on attribute type.

Another recommended reading for simulating datasets in python is 'Practical Time Series Analysis-Prediction with Statistics and Machine Learning' by Aileen Nelson.


## Creating simulated customer subscription dataset


```python
def customer_subscription(debug = False):
    '''Customer subscription data

    This dataset exemplifies a consumer segmentation project.
    We are offering a subscription-based service (such as cable television or membership in a warehouse club)
    and have collected data from N = 300 respondents on age, gender, income, number of children, whether they own or rent their homes, and whether they currently subscribe to the offered service or not.

    The code has been taken from the book :
        • 'Python for Marketing Research and Analytics'
        by Jason S. Schwarz,Chris Chapman, Elea McDonnell Feit
        Chapter 5 - Comparing Groups: Tables and Visualizations

    Additional links :
        • An information website: https://python-marketing-research.github.io
        • A Github repository: https://github.com/python-marketing-research/python-marketing-research-1ed
        • The Colab Github browser: https://colab.sandbox.google.com/github/python-marketing-research/python-marketing-research-1ed

    '''

    # Defining the six variables
    segment_variables = ['age', 'gender', 'income', 'kids', 'own_home',
                         'subscribe']

    # segment_variables_distribution defines what kind of data will be
    # present in each of those variables: normal data (continuous), binomial (yes/no), or Poisson (counts).
    # segment_variables_ distribution is a dictionary keyed by the variable name.

    segment_variables_distribution = dict(zip(segment_variables,
                                              ['normal', 'binomial',
                                               'normal','poisson',
                                               'binomial', 'binomial']))

    # segment_variables_distribution['age']


    # defining the statistics for each variable in each segment:
    segment_means = {'suburb_mix': [40, 0.5, 55000, 2, 0.5, 0.1],
                     'urban_hip': [24, 0.7, 21000, 1, 0.2, 0.2],
                     'travelers': [58, 0.5, 64000, 0, 0.7, 0.05],
                     'moving_up': [36, 0.3, 52000, 2, 0.3, 0.2]}


    # standard deviations for each segment
    # None = not applicable for the variable)
    segment_stddev = {'suburb_mix': [5, None, 12000, None, None, None],
                      'urban_hip': [2, None, 5000, None, None, None],
                      'travelers': [8, None, 21000, None, None, None],
                      'moving_up': [4, None, 10000, None, None, None]}


    segment_names = ['suburb_mix', 'urban_hip', 'travelers', 'moving_up']
    segment_sizes = dict(zip(segment_names,[100, 50, 80, 70]))

    # iterate through all the segments and all the variables and create a
    # dictionary to hold everything:
    segment_statistics = {}
    for name in segment_names:
        segment_statistics[name] = {'size': segment_sizes[name]}
        for i, variable in enumerate(segment_variables):
            segment_statistics[name][variable] = {
                'mean': segment_means[name][i],
                'stddev': segment_stddev[name][i]}

    if debug == True :
        print('segment_statistics : {}'.format(segment_statistics.keys()))
        print('segment_names : {}'.format(segment_statistics))

    # Final Segment Data Generation

    #Set up dictionary "segment_constructor" and pseudorandom number sequence
    #For each SEGMENT i in "segment_names" {
      # Set up a temporary dictionary "segment_data_subset" for this SEGMENT’s data

      # For each VARIABLE in "seg_variables" {
        # Check "segment_variable_distribution[variable]" to find distribution type for VARIABLE

        # Look up the segment size and variable mean and standard deviation in segment_statistics
        # for that SEGMENT and VARIABLE to
        # ... Draw random data for VARIABLE (within SEGMENT) with
        # ... "size" observations
    # }
      # Add this SEGMENT’s data ("segment_data_subset") to the overall data ("segment_constructor")
      # Create a DataFrame "segment_data" from "segment_constructor"
    # }


    import numpy as np
    import pandas as pd
    np.random.seed(seed=2554)
    segment_constructor = {}
    # Iterate over segments to create data for each
    for name in segment_names:
        segment_data_subset = {}
        if debug == True :
            print('segment: {0}'.format(name))
        # Within each segment, iterate over the variables and generate data
        for variable in segment_variables:
            if debug == True :
                print('\tvariable: {0}'.format(variable))
            if segment_variables_distribution[variable] == 'normal':
                # Draw random normals
                segment_data_subset[variable] = np.random.normal(
                    loc=segment_statistics[name][variable]['mean'],
                    scale=segment_statistics[name][variable]['stddev'],
                    size=segment_statistics[name]['size']
                )
            elif segment_variables_distribution[variable] == 'poisson':
                # Draw counts
                segment_data_subset[variable] = np.random.poisson(
                    lam=segment_statistics[name][variable]['mean'],
                    size=segment_statistics[name]['size']
                )
            elif segment_variables_distribution[variable] == 'binomial':
                # Draw binomials
                segment_data_subset[variable] = np.random.binomial(
                    n=1,
                    p=segment_statistics[name][variable]['mean'],
                    size=segment_statistics[name]['size']
                )
            else:
                # Data type unknown
                if debug == True :
                    print('Bad segment data type: {0}'.format(
                        segment_variables_distribution[j])
                        )
                raise StopIteration
            segment_data_subset['Segment'] = np.repeat(
                name,
                repeats=segment_statistics[name]['size']
            )
            segment_constructor[name] = pd.DataFrame(segment_data_subset)

    segment_data = pd.concat(segment_constructor.values())


    # perform a few housekeeping tasks,
    # converting each binomial variable to clearer values, booleans or strings:
    segment_data['gender'] = (segment_data['gender'] \
                              .apply( lambda x: 'male' if x else 'female'))
    segment_data['own_home'] = (segment_data['own_home'] \
                                .apply(lambda x: True if x else False ))
    segment_data['subscribe'] = (segment_data['subscribe'] \
                                 .apply( lambda x: True if x else False))
    return segment_data
```


```python
df2 = customer_subscription()
```

## Inspecting the dataframe


```python
df2.head()
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
      <th>age</th>
      <th>Segment</th>
      <th>gender</th>
      <th>income</th>
      <th>kids</th>
      <th>own_home</th>
      <th>subscribe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>44.057078</td>
      <td>suburb_mix</td>
      <td>female</td>
      <td>54312.575694</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34.284213</td>
      <td>suburb_mix</td>
      <td>female</td>
      <td>67057.192182</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45.159484</td>
      <td>suburb_mix</td>
      <td>female</td>
      <td>56306.492991</td>
      <td>3</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41.032557</td>
      <td>suburb_mix</td>
      <td>male</td>
      <td>66329.337521</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41.781819</td>
      <td>suburb_mix</td>
      <td>female</td>
      <td>56500.410372</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Checking null values


```python
df2.isnull().sum()
```




    age          0
    Segment      0
    gender       0
    income       0
    kids         0
    own_home     0
    subscribe    0
    dtype: int64


### Number of unique values for each feature


```python
print(df2.nunique(axis=0))
```

    age          300
    Segment        4
    gender         2
    income       300
    kids           8
    own_home       2
    subscribe      2
    dtype: int64


## EDA overview of categorical features


```python
UVA_category(df2)
```

    ['Segment', 'gender', 'own_home', 'subscribe']



![png](output_32_1.png)


<a id='Online_Shoppers_Purchasing_Intention_Dataset'></a>
# Online Shoppers Purchasing Intention Dataset

## Loading the dataset

- We have downloaded dataset from the [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) and locally stored it.  
- We load the dataset into a dataframe using [pandas.read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) function.


```python
path = "/Users/bhaskarroy/BHASKAR FILES/BHASKAR CAREER/Data Science/Practise/\
Python/UCI Machine Learning Repository/Online Shoppers Purchasing Intention Dataset Data Set/"

path1 = path + "online_shoppers_intention.csv"
```


```python
df3 = pd.read_csv(path1)
```

## About the dataset
The dataset consists of feature vectors belonging to 12,330 sessions.
The dataset was formed so that each session
would belong to a different user in a 1-year period to avoid
any tendency to a specific campaign, special day, user
profile, or period.

## Column Descriptions  

- **Administrative**: This is the number of pages of this type (administrative) that the user visited.  
- **Administrative_Duration**: This is the amount of time spent in this category of pages.
- **Informational**: This is the number of pages of this type (informational) that the user visited.
- **Informational_Duration**: This is the amount of time spent in this category of pages.
- **ProductRelated**: This is the number of pages of this type (product related) that the user visited.  
- **ProductRelated_Duration**: This is the amount of time spent in this category of pages.
- **BounceRates**: The percentage of visitors who enter the website through that page and exit without triggering any additional tasks. (https://support.google.com/analytics/answer/1009409?)
- **ExitRates**: The percentage of pageviews on the website that end at that specific page.  
(https://support.google.com/analytics/answer/2525491?)
- **PageValues**: The average value of the page averaged over the value of the target page and/or the completion of an eCommerce transaction. (https://support.google.com/analytics/answer/2695658?hl=en)
- **SpecialDay**: This value represents the closeness of the browsing date to special days or holidays (eg Mother's Day or Valentine's day) in which the transaction is more likely to be finalized. More information about how this value is calculated below.
- **Month**: Contains the month the pageview occurred, in string form.
- **OperatingSystems**: An integer value representing the operating system that the user was on when viewing the page.
- **Browser**: An integer value representing the browser that the user was using to view the page.
- **Region**: An integer value representing which region the user is located in.
- **TrafficType**: An integer value representing what type of traffic the user is categorized into. (https://www.practicalecommerce.com/Understanding-Traffic-Sources-in-Google-Analytics)
- **VisitorType**: A string representing whether a visitor is New Visitor, Returning Visitor, or Other.
- **Weekend**: A boolean representing whether the session is on a weekend.
- **Revenue**: A boolean representing whether or not the user completed the purchase.

## Inspecting the dataframe


```python
df3.head()
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
      <th>Administrative</th>
      <th>Administrative_Duration</th>
      <th>Informational</th>
      <th>Informational_Duration</th>
      <th>ProductRelated</th>
      <th>ProductRelated_Duration</th>
      <th>BounceRates</th>
      <th>ExitRates</th>
      <th>PageValues</th>
      <th>SpecialDay</th>
      <th>Month</th>
      <th>OperatingSystems</th>
      <th>Browser</th>
      <th>Region</th>
      <th>TrafficType</th>
      <th>VisitorType</th>
      <th>Weekend</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.20</td>
      <td>0.20</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Feb</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Returning_Visitor</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>64.000000</td>
      <td>0.00</td>
      <td>0.10</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Feb</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>Returning_Visitor</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.20</td>
      <td>0.20</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Feb</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>Returning_Visitor</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>2.666667</td>
      <td>0.05</td>
      <td>0.14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Feb</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>Returning_Visitor</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>10</td>
      <td>627.500000</td>
      <td>0.02</td>
      <td>0.05</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Feb</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>Returning_Visitor</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.describe()
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
      <th>Administrative</th>
      <th>Administrative_Duration</th>
      <th>Informational</th>
      <th>Informational_Duration</th>
      <th>ProductRelated</th>
      <th>ProductRelated_Duration</th>
      <th>BounceRates</th>
      <th>ExitRates</th>
      <th>PageValues</th>
      <th>SpecialDay</th>
      <th>OperatingSystems</th>
      <th>Browser</th>
      <th>Region</th>
      <th>TrafficType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.315166</td>
      <td>80.818611</td>
      <td>0.503569</td>
      <td>34.472398</td>
      <td>31.731468</td>
      <td>1194.746220</td>
      <td>0.022191</td>
      <td>0.043073</td>
      <td>5.889258</td>
      <td>0.061427</td>
      <td>2.124006</td>
      <td>2.357097</td>
      <td>3.147364</td>
      <td>4.069586</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.321784</td>
      <td>176.779107</td>
      <td>1.270156</td>
      <td>140.749294</td>
      <td>44.475503</td>
      <td>1913.669288</td>
      <td>0.048488</td>
      <td>0.048597</td>
      <td>18.568437</td>
      <td>0.198917</td>
      <td>0.911325</td>
      <td>1.717277</td>
      <td>2.401591</td>
      <td>4.025169</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>184.137500</td>
      <td>0.000000</td>
      <td>0.014286</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>7.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>598.936905</td>
      <td>0.003112</td>
      <td>0.025156</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>93.256250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>38.000000</td>
      <td>1464.157214</td>
      <td>0.016813</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>27.000000</td>
      <td>3398.750000</td>
      <td>24.000000</td>
      <td>2549.375000</td>
      <td>705.000000</td>
      <td>63973.522230</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>361.763742</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>13.000000</td>
      <td>9.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Checking null values


```python
df3.isnull().sum()
```




    Administrative             0
    Administrative_Duration    0
    Informational              0
    Informational_Duration     0
    ProductRelated             0
    ProductRelated_Duration    0
    BounceRates                0
    ExitRates                  0
    PageValues                 0
    SpecialDay                 0
    Month                      0
    OperatingSystems           0
    Browser                    0
    Region                     0
    TrafficType                0
    VisitorType                0
    Weekend                    0
    Revenue                    0
    dtype: int64



### Number of unique values for each feature


```python
uniques = df3.nunique(axis=0)
print(uniques)
```

    Administrative               27
    Administrative_Duration    3335
    Informational                17
    Informational_Duration     1258
    ProductRelated              311
    ProductRelated_Duration    9551
    BounceRates                1872
    ExitRates                  4777
    PageValues                 2704
    SpecialDay                    6
    Month                        10
    OperatingSystems              8
    Browser                      13
    Region                        9
    TrafficType                  20
    VisitorType                   3
    Weekend                       2
    Revenue                       2
    dtype: int64



```python
print(list(df3.columns))
```

    ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Revenue']


## EDA overview of categorical features


```python
list2 = ['Month', 'OperatingSystems', 'Browser','Region', 'TrafficType', 'VisitorType','Revenue']
UVA_category(df3, list2,normalize = False,
                               axlabel_fntsize= 10,
                               ax_xticklabel_fntsize = 9, ax_yticklabel_fntsize = 9,
                               infofntsize= 12)
```

    ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Revenue']



![png](output_49_1.png)  

# Useful links  
- [Set opacity of background colour in matplotlib](https://stackoverflow.com/questions/4581504/how-to-set-opacity-of-background-colour-of-graph-with-matplotlib)  
- [PEP style guide for python code](https://peps.python.org/pep-0008/)  
- [Python for marketing research and analytics Notebooks](https://github.com/python-marketing-research/python-marketing-research-1ed)
