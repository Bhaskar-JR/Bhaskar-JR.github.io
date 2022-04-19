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
  params_plot = dict(colcount = 2, colwidth = 7, rowheight = 4,
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

    plt.gcf().tight_layout()
