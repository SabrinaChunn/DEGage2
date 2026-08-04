import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod import families

import warnings
warnings.simplefilter("ignore", category=sm.tools.sm_exceptions.HessianInversionWarning)
warnings.simplefilter("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)

def calculate_r(data):
    """
    Calculates the r parameter based on a negative binomial distribution with GLM

    Parameters 
    ----------
    data: 2D np array 
        Stores the normalized count expression data for one group

    Returns
    -------
    r_list: float64 list
        Stores all calculated r parameters 
    """

    #define empty list to store all of the r values in 
    r_list = []
    #iterate across each gene to get r value 
    for row in data:
        #fit negative binomial distribution to the counts 
        model = sm.NegativeBinomial(row, np.ones(len(row)))
        #set r to large number if there was high variance [prevents issues]
        if np.var(row) < 1e-6:
            r = 1E6
        #get the dispersion parameter to determine r 
        else:
            result = model.fit(disp=0, method='nm')
            alpha = result.params[-1]
            r = 1 / alpha
        r_list.append(r)
    return r_list
    
def calculate_k_subsample(data, group1_size):
    """
    Calculates the test statsistic with subsampling the larger group 

    Parameters 
    ----------
    data: 2D np array 
        Stores the normalized count expression data for one group
    group1_size: int 
        The number of samples in the first category/group

    Returns
    -------
    k_list: float64 list
        Stores all calculated test statistics  
    """

    k_list = []
    for value in range(len(data)):
        category1_values = data[value, :group1_size].tolist()
        category2_values = data[value, group1_size:].tolist()
        np.random.shuffle(category1_values)
        np.random.shuffle(category2_values)
        number_of_groups = min(len(category1_values), len(category2_values))
        category1_values = category1_values[:number_of_groups]
        category2_values = category2_values[:number_of_groups]
        k = np.mean(category2_values) - np.mean(category1_values)
        k_list.append(k)
    return k_list

def calculate_k_bootstrapped(data, group1_size):
    """
    Calculates the test statsistic with boostrapping the smaller group 

    Parameters 
    ----------
    data: 2D np array 
        Stores the normalized count expression data for one group
    group1_size: int 
        The number of samples in the first category/group
        
    Returns
    -------
    k_list: float64 list
        Stores all calculated test statistics  
    """

    k_list = []
    
    for row in range(data.shape[0]): #iterates through each gene (row)
        #separate by groups 
        category1_values = data[row, :group1_size]
        category2_values = data[row, group1_size:]
        len1 = len(category1_values)
        len2 = len(category2_values)
        
        #find smaller group
        target_size = max(len1, len2)
        
        #bootstrap the smaller group to match the larger one 
        if len1 < len2:
            #bootstrap group1
            resampled_cat1 = np.random.choice(category1_values, size=target_size, replace=True) #resample with replacement
            resampled_cat2 = category2_values 
        elif len2 < len1:
            #bootstrap group2
            resampled_cat1 = category1_values
            resampled_cat2 = np.random.choice(category2_values, size=target_size, replace=True) #resample with replacement
        else:
            #if equal no change 
            resampled_cat1 = category1_values
            resampled_cat2 = category2_values
            
        #identify mean difference of bootstrapped groups 
        k = np.mean(resampled_cat2) - np.mean(resampled_cat1)
        
        k_list.append(k)
        
    return k_list

def calculate_log2_fold_change(data, group1_size):
    """
    Calculates the log2-fold-change based on log2((mu2+0.1)/(mu1+0.1))

    Parameters 
    ----------
    data: 2D np array 
        Stores the normalized count expression data for one group
    group1_size: int 
        The number of samples in the first category/group
        
    Returns
    -------
    lfc_list: float64 list
        Stores all log-2-fold-change values   
    """
    #get the means across group 1 and group 2 columns 
    mean_group_1 = np.mean(data[:, :group1_size], axis=1)
    mean_group_2 = np.mean(data[:, -group1_size:], axis=1)

    #determine the log-2-fold-changes for each row 
    lfc_list = np.log2(mean_group_2 + 0.1) - np.log2(mean_group_1 + 0.1) 
    
    #return the array of log2-fold-change values
    return(lfc_list)