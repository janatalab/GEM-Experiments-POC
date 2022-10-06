# Functions used across multiple GEM experiments and analyses
# Associated with Fink, Alexander, Janata (2022)
# Lauren Fink
# Contact: lkfink@ucdavis.edu
# or lauren.fink@ae.mpg.de

# Load required packages
import os
import sys
import seaborn as sns
import pandas as pd
import numpy as np
from numpy import *
import re
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
import pingouin as pg
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# --------------------------------------------------------------------------------------------------------------#
# Basic function to normalize data within a pandas series

# Input:
# - pandas series containing raw data
# Output:
# - pandas series containing normalized data

def zscore(series):
    return (series - series.mean()) / series.std()

# --------------------------------------------------------------------------------------------------------------#
# Function to calculate repeated measures ANOVA from input variables of interest

# Input:
# - Dataframe containing all experiment data
# - Flag whether this data is from group experiment (1) or not (0)
# - Variable of interest to average over (e.g. 'avg_std_async'). Make sure this matches the column label in the data frame
# - Flag whether to zscore data (1) or not (0) in var column of interest. Note that ratings have already been zscored
#     within each participant and item. Tapping data have not been zscored, as raw values are also of interest.

# Output:
# - table containing AOV results
# - table containing t-test results

def rmaov_results(df, group, var, to_zscore):
    if group:
        col_label = 'group_num'
    else:
        col_label = 'subID'

    # z-score variable of interest, if user requested
    if to_zscore:
        zscores = df.groupby(col_label)[var].transform(zscore)
        df.insert(0, "zscores", zscores, True)
        var = 'zscores'

    means = df.groupby([col_label, 'alpha'], as_index=False)[var].mean()
    print('\n\n')

    # Compute the repeated measures ANOVA
    print('ANOVA table\n')
    print(pg.rm_anova(dv=var, within='alpha', subject=col_label, data=means, detailed=True))

    # Comparisons between all means
    print('\n\nComparisons between all means\n')
    return pg.pairwise_ttests(dv=var, within='alpha', subject=col_label, data=means, alpha = .05, tail = 'one-sided', padjust = 'holm', effsize = 'cohen')


# --------------------------------------------------------------------------------------------------------------#
# Function to return means and sems for any variable of interest against alpha condition (for a single participant or group)

# Input:
# - Dataframe containing all experiment data
# - Flag whether this data is from group experiment (1) or not (0)
# - Variable of interest to average over (e.g. 'avg_std_async'). Make sure this matches the column label in the data frame
# - Flag whether to zscore data (1) or not (0) in var column of interest. Note that ratings have already been zscored
#     within each participant and item. Tapping data have not been zscored, as raw values are also of interest.

# Output:
# - data frame containing means and sems for variable of interest, in each alpha condition

def return_mean_sem(df, group, var, to_zscore):
    if group:
        col_label = 'group_num'
    else:
        col_label = 'subID'

    # z-score variable of interest, if user requested
    if to_zscore:
        zscores = df.groupby(col_label)[var].transform(zscore)
        df.insert(0, "zscores", zscores, True)
        var = 'zscores'

    # get mean for each sub/group and alpha condition
    means = df.groupby([col_label, 'alpha'], as_index=False)[var].mean()

    # get mean across groups for each alpha condition
    means_all = means.groupby(['alpha'])[var].mean()

    # do the same for the standared error of the mean
    sems = df.groupby([col_label, 'alpha'], as_index=True)[var].sem().reset_index()
    sems_all = means.groupby(['alpha'])[var].sem()

    # subtract baseline mean from each condition mean
    subtract_base = means_all - means_all[0]

    # return means and sems for user
    final_avg_sem_by_cond =  pd.concat([means_all, sems_all, subtract_base], join = "inner", axis = 1)
    final_avg_sem_by_cond.columns = ['mean', 'sem', 'mean-baseline']
    return final_avg_sem_by_cond


# --------------------------------------------------------------------------------------------------------------#
# Function to plot any variable of interest against alpha condition for each participant or group in the experiment
# Input:
# - experiment data frame
# - flag whether it is a group exp or not
# - specify the variable of interest to plot (e.g. std async, groove rating, etc.)

# Output:
# - Figure containing multiple subplots (one for each participant or group)


def plot_all(df, group, varcol): #, to_zscore)

    fig = plt.figure(figsize=(20,20))

    if group: # group tapping
        IDcol = 'group_num'
        group = 1
    else:
        group = 0
        IDcol = 'subID'

    # Calculate means and sems
    means = df.groupby([IDcol, 'alpha'], as_index=False)[varcol].mean()
    sems = df.groupby([IDcol, 'alpha'], as_index=True)[varcol].sem().reset_index()

    # Figure out number of subplots we need in figure
    groups = unique(df[IDcol])
    plotrows = ceil(round(len(groups))/5)
    plotcols = ceil(round(len(groups))/plotrows)

    # Loop through each participant or group and plot
    for idx, val in enumerate(groups):

        # Subset data
        submeans = means.loc[means[IDcol] == groups[idx]]
        #print(submeans)
        subsems = sems.loc[sems[IDcol] == groups[idx]]

        # Create new axis for every subplot
        axstr = str('ax' + str(idx))
        axstr = fig.add_subplot(plotrows, plotcols, idx+1)
        plt.errorbar(submeans['alpha'], submeans[varcol], yerr = subsems[varcol], fmt='-ko')
        axstr.set_xticks(submeans['alpha'])
        axstr.set_xticklabels(submeans['alpha'])
        plt.title(val)

    # Add meta-labels to figure
    fig.text(0.5, 0.00, 'Adaptivity', ha='center', va='center', fontsize=16)
    fig.text(0.0, 0.5, varcol, ha='center', va='center', rotation='vertical', fontsize=16)
    fig.tight_layout(h_pad=3, w_pad=3)



# --------------------------------------------------------------------------------------------------------------#
# Function to to perform Bartletts's and KMO test to check if data are suitable for FA
# Input:
# -  data frame of ratings

# Output:
# - print statements about statistical tests and suitability of future FA
# - flag about whether to proceed

def check_fa(df):
    flag = 0
    chi_square_value,p_value=calculate_bartlett_sphericity(df)
    chi_square_value, p_value
    print('Bartlett’s test of sphericity: ', chi_square_value, '\np value: ', p_value, '\ndf:', df.shape[0]-1)
    if p_value > .05:
        print('\nProbably should not do factor analysis')
        flag = 1
    else:
        print('\nResults are fine. Ok to proceed.\n\n')

    kmo_all,kmo_model=calculate_kmo(df)
    print('Kaiser-Meyer-Olkin (KMO) Test: ', kmo_model)
    if kmo_model < .6:
        print('\nProbably should not do factor analysis')
        flag = 1
    else:
        print('\nResults are fine. Ok to proceed.\n\n')
    return flag



# --------------------------------------------------------------------------------------------------------------#
# Function to to perform factor analyses and transform raw ratings into factor scores
# Input:
# - data frame of ratings

# Output:
# - print statements about eigen values and factor loadings
# - tranformed df of factor scores

def factor_df(df):
    # Create factor analysis object and perform factor analysis
    fa = FactorAnalyzer(rotation='varimax') #try diff rotations
    fa.fit(df)

    # Check Eigenvalues
    ev, v = fa.get_eigenvalues()
    print('Eigenvalues for factors\n', ev, '\n')
    #print(v)
    nfactors = sum(ev > 1.5)
    print(nfactors, 'eigenvalue(s) are > 1. Therefore, proceed with', nfactors, 'factor(s).')

    # Plot scree if we want
    plt.scatter(range(1,df.shape[1]+1),ev)
    plt.plot(range(1,df.shape[1]+1),ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')

    # Now do fa again and specify number of factors
    fa = FactorAnalyzer(n_factors=nfactors, method='ml', rotation=None) #no rotation when only one factor
    fa.fit(df)
    loadings = fa.loadings_
    communalities = fa.get_communalities()
    variance = fa.get_factor_variance()

    # create new column labels based on number of factors
    factorLabels = list()
    factorLabels.append('Item')
    for i in range(nfactors):
        factorLabels.append('F' + str(i+1))

    loading_DF = pd.DataFrame(columns = factorLabels)
    loading_DF['Item'] = df.columns
    for i, c in enumerate(df.columns):
        loading_DF.iloc[i, 1:nfactors+1] = loadings[i]
    loading_DF['communalities'] = communalities
    print('\nFactor Loadings:\n', loading_DF)

    print('\nvariance:\n', 'Factor variances, proportional variance, cumulative variances\n', variance)

    # transform data
    return fa.transform(df)
