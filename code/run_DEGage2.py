import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import csv
import time

import preprocessing_DEGage2 as prep
import parameters_DEGage2 as params
import DOTNB_DEGage2 as DEGage2_DOTNB

def DOTNB(data, geneNames, group1, group2, DOTNB_thresh):
    """
    Runs the workflow of DEGage2 

    Parameters 
    ----------
    data: 2D np array 
        Stores the normalized count expression data
    geneNames: 1D np string array
        Stores the names of genes 
    group1_size: int 
        The number of samples in the first category/group
    group2_size: int 
        The number of samples in the second category/group

    Returns
    -------
    results_df: pandas df
        Dataframe holding all of the parameters and results for each DEG
    """    

    print("Obtaining negative binomial parameters and test statistic...")

    #convert alpha to r (r = 1/alpha) | r shared due to shrinkage 
    r_list1 = np.array(params.calculate_r(data[:, :group1]))
    r_list2 = np.array(params.calculate_r(data[:, group1:]))

    #get the mu parameters 
    mu_list1 = np.array(data[:, :group1].mean(axis=1).tolist())    
    mu_list2 = np.array(data[:, group1:].mean(axis=1).tolist())  
    
    #get the p parameters
    p_list1 = np.array(r_list1 / (r_list1 + mu_list1))
    p_list2 = np.array(r_list2 / (r_list2 + mu_list2))
    q1_list = 1 - p_list1
    q2_list = 1 - p_list2
    
    #get the test statistic [k]
    k_list = params.calculate_k_bootstrapped(data, group1)

    #get the mean and variance 
    DOTNB_mean_list = (r_list1 * q1_list / p_list1) - (r_list2 * q2_list / p_list2)
    DOTNB_var_list = (r_list1 * q1_list / p_list1**2) - (r_list2 * q2_list / p_list2**2)

    #get the PMF and CDF values
    print("Running DOTNB on a ", DOTNB_thresh, " threshold...") 
    print("Getting probability mass function values...")
    pmf_list = [
        DEGage2_DOTNB.DOTNB_pmf(r1, p1, r2, p2, k)
        for r1, p1, r2, p2, k in zip(r_list1, p_list1, r_list2, p_list2, k_list)
    ]
    
    print("Getting cumulative distribution function values...")
    cdf_list = [
        DEGage2_DOTNB.DOTNB_cdf(r1, p1, r2, p2, k)
        for r1, p1, r2, p2, k in zip(r_list1, p_list1, r_list2, p_list2, k_list)
    ]
    
    #create a filter based on pvalues [CDF or PMF]
    cdf_list = np.array(cdf_list)
    pmf_list = np.array(pmf_list)
    # mask = cdf_list <= thresh 
    mask = pmf_list <= DOTNB_thresh

    #convert remaining items into np arrays 
    k_list = np.array(k_list)
    DOTNB_mean_list = np.array(DOTNB_mean_list)
    DOTNB_var_list = np.array(DOTNB_var_list)

    #filter the columns based on identified DEGs
    geneNames = geneNames[mask]
    cdf_list = cdf_list[mask]
    pmf_list = pmf_list[mask]
    r_list1 = r_list1[mask]
    r_list2 = r_list2[mask]
    mu_list1 = mu_list1[mask]
    mu_list2 = mu_list2[mask]
    p_list1 = p_list1[mask]
    p_list2 = p_list2[mask]
    k_list = k_list[mask]
    DOTNB_mean_list = DOTNB_mean_list[mask]
    DOTNB_var_list = DOTNB_var_list[mask]

    #create a dictionary of the results 
    results_dict = {
        'genes': geneNames,
        'mean': DOTNB_mean_list,
        'variance': DOTNB_var_list,
        'r1': r_list1,
        'r2': r_list2,
        'mu1': mu_list1,
        'mu2': mu_list2,
        'p1': p_list1,
        'p2': p_list2,
        'k': k_list,
        'pmf': pmf_list,
        'cdf': cdf_list
    }

    #convert to and return df of results 
    results_df = pd.DataFrame(results_dict)
    return results_df

def DEGage2_0(count_file, groups, optimizer_file = "NA" , DOTNB_thresh = -1.0 , perm_thresh = -1.0 , num_permutations = 1000):
    """
    Runs the workflow of DEGage2 

    Parameters 
    ----------
    count_file: string
        The location of the count data 
    groups: 1D np integer array
        Contains 0s and 1s corresponding to the condition identity of each column 
    optimizer_file: string
        Location of the file or files with optimized thresholds 
    DOTNB_thresh: float64
        Threshold for the DOTNB p-value
    perm_thresh: float64
        Threshold for the permutation test p-value
    num_permutations: integer
        The number of permutation completed for the permutation test 

    Returns
    -------
    elapsed_time: float64
        The time it took to run the program 
    results_df: pandas df
        Dataframe holding all of the parameters and results for each DEG
    DOTNB_thresh: float64
        Threshold for the DOTNB p-value
    perm_thresh: float64
        Threshold for the permutation test p-value
    """    

    #start the processing time and load the files 
    start_time = time.time()
    count_df = pd.read_csv(count_file)
    data = count_df.iloc[:, 1:len(count_df.columns)].to_numpy()
    geneNames = count_df.iloc[:, 0].to_numpy()

    #preprocess and handle the data 
    data = prep.preprocess(data)
    data_group1 = data[:, groups == 0]
    data_group2 = data[:, groups == 1]
    data = np.hstack((data_group1, data_group2))
    group1_size = data_group1.shape[1]
    group2_size = data_group2.shape[1]
    
    #set threshold to use if nothing is provided 
    if ((optimizer_file == "NA") and DOTNB_thresh == -1.0):
        print("Warning: Neither DOTNB threshold nor optimizer file has been entered...")
        print("Auto selecting DOTNB threshold to 0.05...")
        DOTNB_thresh = 0.05

    if ((optimizer_file == "NA") and perm_thresh == -1.0):
        print("Warning: Neither permutation threshold nor optimizer file has been entered...")
        print("Auto selecting permutation threshold to 0.01...")
        perm_thresh = 0.01

    #get optimal thresholds if unspecified 
    if (perm_thresh == -1 or DOTNB_thresh == -1):
        if (perm_thresh == -1):
            perm_thresh = prep.get_optimal_threshold(optimizer_file, group1_size, group2_size, threshold_type = "Perm Thresh")
        if (DOTNB_thresh == -1):
            DOTNB_thresh = prep.get_optimal_threshold(optimizer_file, group1_size, group2_size, threshold_type = "DOTNB Thresh")

    #run the permutation test
    data, geneNames = prep.permutation(data, geneNames, group1_size, perm_thresh, num_permutations,)

    #call DOTNB functions 
    results_df = DOTNB(data, geneNames, group1_size, group2_size, DOTNB_thresh)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(count_file, " processed")
    return (elapsed_time, results_df, DOTNB_thresh, perm_thresh) #return df of results 

