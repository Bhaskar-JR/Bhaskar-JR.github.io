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
