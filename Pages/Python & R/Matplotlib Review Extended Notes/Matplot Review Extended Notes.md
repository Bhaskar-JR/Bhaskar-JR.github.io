---
layout : single
author_profile : true
toc : true
toc_sticky : true
---

# Epilogue to the Matplotlib Review Files

We are at the end of this review of matplotlib.  
We have covered lots of ground. Indeed, the learning curve is steep, As there is a very large code base teeming with both stateless (pyplot) and stateful (OO style) interfaces.  

## Nuggets of distilled wisdom  
However, certain fundamentals remain the same. That being, Every element in the plot is an artist that knows how to render itself on the figure canvas. Every Artist instance has properties, parameters and possibly attributes. Axes and Figure are the most important Artist/Containers in the matplotlib universe. It is through the the Axes instance, one can access almost all the artists in the figure. Once having accessed the specific artist instance, the properties can be modified using artist.set method. At any moment one can chose to inspect the artist objects.  

## Deeper Understanding of Matplotlib
A deeper understanding of matplotlib can emerge by understanding “Anatomy of a Figure” and the hierarchy of artists. This should be followed by sections - [“the Lifecycle of a Plot”](https://matplotlib.org/stable/tutorials/introductory/lifecycle.html) and [“the Artist Tutorial”](https://matplotlib.org/3.5.0/tutorials/intermediate/artists.html) in [Matplotlib Release, 3.4.2](https://matplotlib.org/3.4.2/Matplotlib.pdf). Without the understanding of these two foundational concepts, we would be limiting our capability and fail to leverage the full range of matplotlib functions and different class modules. Further, most efforts in debugging and coding would materialize into suboptimal coding practices.  

In terms of the resources, the latest official documentation “[Matplotlib Release, 3.4.2](https://matplotlib.org/3.4.2/Matplotlib.pdf)” pdf is a **veritable treasure trove** along with the examples gallery. Though, it may take some time to establish a pattern to search the required information. It also requires a threshold knowledge on the part of the user. One section to look out for is – “WHAT'S NEW IN MATPLOTLIB 3.4.0” for the latest developments and feature rollouts. Further, I also would suggest to check [Matplotlib Release, 2.0.2 version](https://matplotlib.org/2.0.2/Matplotlib.pdf) especially for the matplotlib examples section. Apart from the matplotlib official site, [**StackOverflow**](https://stackoverflow.com/questions/tagged/matplotlib) is the go to site for all queries with an active community.  

As we have limited our focus here on becoming comfortable on OO style, we have not covered numerous topic such as color bars, Tight Layout guide, Constraint layout guide, Different Patches and Advanced tutorials such as image tutorials, path tutorials. The official documentation has a comprehensive coverage on those.  

## Seaborn, Pandas and Matplotlib  

If you are majorly a [Seaborn](https://seaborn.pydata.org/) User, this review will let you finetune the seaborn plots. As Seaborn is a high-level wrapper for Matplotlib, in essence, the OO interface will work there as well. While Seaborn plots take very less code and is intuitive from the moment go, this comes at the cost of lesser flexibility unless complemented by matplotlib. For instance, in a [facetgrid object](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html), modifying the row titles, column titles or the placement of figure legends, subplot legends becomes fairly easy with understanding of OO style. Same goes with [**Pandas**](https://pandas.pydata.org/docs/user_guide/index.html) plotting. Pandas has well developed charting capabilities. [Pandas](https://pandas.pydata.org/docs/user_guide/index.html) has powerful functionalities to deal with time series data.  
>Using the three in conjunction particularly in multiple subplots will let you extract the best from each of the modules. Yes, this is the way.  

Depending on specific use case, we shall go for the best combination. For sophisticated plots like [heat maps](https://seaborn.pydata.org/examples/spreadsheet_heatmap.html?highlight=heatmaps) or even [faceted grids]((https://seaborn.pydata.org/generated/seaborn.FacetGrid.html)), Seaborn churns out beautiful charts with minimal code. For time series, [Pandas](https://pandas.pydata.org/docs/user_guide/index.html) has a definitive edge. Use **matplotlib axes instance methods** to customize them further. In that sense, the whole is indeed greater than the sum of parts.  

## From R, ggplot2 to Matplotlib and Python  

For R Users, matplotlib can be analogous to R base graphics. And very likely, one used to the elegant graphics of ggplot2 can find the default matplotlib graphics code verbose and plot outputs very basic or “bare bones” similar to plots in R base graphics. But, it’s only a matter of time before one gets comfortable to the extent of customizing the plots even to the minutest level of setting orientation, thickness, number, padding of ticks. There is **immense flexibility and clarity that is on offer within a clear OO framework of matplotlib**. While I would rather not choose sides between ggplot2 and Python, instead am in favor of viewing them as tools for analysis to be used depending on particular use contexts. And while ggplot2 is easy to get started with but there is **learning curve** to gain control there in as well.  

>Having advocated the matplotlib skills for Python work environment, a suitable learning journey for R user can be becoming comfortable with Python basics, the basic data structures, built in functions, Numpy, Pandas followed by the pyplot interface and Seaborn. Pyplot interface, Seaborn and Pandas would be good entry points to be culminated with matplotlib.  

## A Note on Excel  

And for Excel users, a reason for learning matplotlib and even ggplot2 can be the **vast library of visualizations** that is simply on offer. Excel spreadsheet shall always remain relevant and has its own place in analysis. But it is no match for advanced data analysis workflow with tidyverse packages in R; and numpy, Pandas in Python. For quick Exploratory data analysis, you have ggplot2 in R and matplotlib/Seaborn/pandas in Python. Also, there are some basic limitations in the charts you can plot in Excel. **Trellis chart** is not possible unless some tricks in terms of arranging the data in separate columns with blanks is employed. But this is only a trick/ manoeuvre around an inherent limitation. To give another limitation, there is **no fill and span feature**. If daily sales data is plotted across months and I want to highlight particular days and weekends, it’s a hard ask. This and lot of other limitations are easily handled in R and Python.  

One definitive edge of R and Python over excel is the **reproducibility of the plots**. While excel is mainly drag and drop, point and click type GUI, so every time you are repeating an analysis, there are number of tasks that have to be redone. Data cleaning and aggregation followed by pivoting and then copying/pasting the relevant outputs. And then doing some EDA and then some graphs if found useful. This entire flow can of course be automated by VBA but that would boil down to individual expertise and additional time of code development and debugging. And if the data is huge, excel is no longer conducive even for pivoting. In comparison, R and Python being code based have a much simpler workflow, less error prone, shorter throughput and exact reproducibility. For an Excel user to get started on matplotlib, the learning journey would be same as suggested previously for R users. To get a headstart, check out below [link](https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_spreadsheets.html) for comparison of Pandas data structures and spreadsheet.  

## Back to this Review  

Coming back to matplotlib, I would also emphasize being thorough with the basics of the color module, **transformation framework** and **legend guide**. Also, few basics like modifying tick lines, tick labels, spines, gridlines, fig patch, axes patch can elevate the visual quality of plots. You also have **rcparams** and style sheets for global changes. Adding subplots using with and without gridspec also requires due consideration. For modifying font properties, using font dictionary is useful.  

There are also **args and kwargs** that you will frequently encounter with every artist instance. It is convenient to create a dictionary containing keyword argument and value pairings in the code script. And later use **dictionary style unpacking** (**dict) to pass those arguments for setting the artist property. This helps in keeping code organized and avoid repetition of keyword arguments.  

**Certain properties are common to all artists**. And there are certain artist specific properties. In this review, I have included the artist methods and instances with high likelihood of use. However, I would strongly recommend being familiar with the official documentation and website for the complete list and final verdict.  

It is also inevitable that we will be dealing with date and time. There are multiple classes to handle the various concepts of date and time – either a single moment in time or a duration of time. To capture a moment in time, we have the [datetime object] (https://docs.python.org/3/library/time.html)in Native Python, [datetime64](https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-and-timedelta-arithmetic) in Numpy, [timestamp](https://pandas.pydata.org/docs/user_guide/timeseries.html) in Pandas, [matplotlib dates](https://matplotlib.org/stable/api/dates_api.html). For parsing strings and ISO 8601 date time formats, we have dateutil parser; datetime strft, strfp; Pandas pd.to_pydatetime to name a few. TimeDelta objects are very handy and will be frequently used in conjunction with the datetime equivalent objects. Eventually, we shall develop liking to a method/module/class and stick with it.  

Further, this review would be incomplete without coverage of **basic data structures** in Python, **Numpy Array** Attributes and Methods, Pandas Objects Attributes and Methods, **Pseudo Random number generation in** [Python](https://docs.python.org/3/library/random.html) and [Numpy](https://numpy.org/doc/stable/reference/random/generator.html?highlight=generator), [Scipy](https://scipy-lectures.org) for Statistics with python. I have included short sections on them to make this review self contained. However, I would recommend to go to official documentation of Pandas and Numpy for details.  

## The Curtain Falls ! But The End is only the beginning !!  

On a final note, I have included an appendix containing multiple examples of plots and code implementations with different use cases. The intent here is to cover a gamut of plotting scenarios that can be always be referred to and specific code snippets be adapted for specific contexts. The references/source links for code implementations wherever applicable has been mentioned. There is also a references list at the very end presenting useful links related to matplotlib, python and data viz in general.
Feel free to go through it.  

To reaffirm, Consider this review as a compilation of **distilled personal learnings and a handy reference**. With this, I wrap up this first edition and wish you the very best !  

Bhaskar  
Aug’2021
