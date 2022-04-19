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

``
