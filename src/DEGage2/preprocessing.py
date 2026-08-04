import numpy as np
import pandas as pd

def preprocess(data):
    """
    Handles 0s and Percentile Normalization of the data 

    Parameters 
    ----------
    data: np array 
        Stores the count expression data

    Returns
    -------
    data: int array
        Stores the normalized count expression data
    """
    
    data = np.nan_to_num(data, nan=0.0) #replace NaN values with 0
    percent95 = np.percentile(data, 95, axis=1, keepdims=True) #gets 95th percentile for each row
    safe_percent95 = np.where(percent95 == 0, 1, percent95) #replaces all 0 percentile values with 1
    capped = np.minimum(data, percent95) #caps all values in a row by the 95th percentile
    data = (capped / safe_percent95) * 100 #divides all rows by 95th percentile [1-100 scale]
    return np.maximum(data, 1.0) #returns normalized percentile and adds a count of 1 for regions where it is 0


def get_optimal_threshold(optimizer_file, group1_size, group2_size, threshold_type = "DOTNB Thresh"):
    """
    Gets the optimized DOTNB or Permutation threshold for the program based on user selection 

    Parameters 
    ----------
    optimizer_file: string
        The location of the csv file with optimized thresholds 
    group1_size: int 
        The number of samples in the first category/group
    group2_size: int 
        The number of samples in the second category/group
    threshold_type: string
        Identifyer and column name for DOTNB or permutation threshold

    Returns
    -------
    optimal_threshold: float64
        The optimal DOTNB or Permutation threshold from the optimizer file 
    """

    #load in the data as a df | format of Group1, Group2, DOTNB Thresh, Perm Thresh
    optimal_df = pd.read_csv(optimizer_file)

    #get optimal threshold based on the group sizings from original group combinations
    optimal_threshold_series = optimal_df.loc[(optimal_df["Group1"] == group1_size) & (optimal_df["Group2"] == group2_size), threshold_type] #will return a series
    if not optimal_threshold_series.empty: #if combination found return the optimal threshold
        optimal_threshold = float(optimal_threshold_series.iloc[0])
        return optimal_threshold

    #get optimal threshold based on the group sizings from rouned group combinations
    print(f"Group size combination {group1_size},{group2_size} not found. Rounding to nearest multiple of 5.")
    rounded_group1_size = int(round(group1_size / 5) * 5) #round to multiple of 5
    rounded_group2_size = int(round(group2_size / 5) * 5) #round to multiple of 5

    optimal_threshold_series = optimal_df.loc[(optimal_df["Group1"] == rounded_group1_size) & (optimal_df["Group2"] == rounded_group2_size), threshold_type]
    if not optimal_threshold_series.empty: #if combination found return the optimal threshold
        optimal_threshold = float(optimal_threshold_series.iloc[0])
        return optimal_threshold

    #get base threshold
    print(f"Group size combination {rounded_group1_size},{rounded_group2_size} does not exist.")
    if (threshold_type == "DOTNB Thresh"):
        optimal_threshold = 0.05
        print(f"Using base DOTNB threshold of {optimal_threshold}")
        return optimal_threshold
    else:
        optimal_threshold = 0.01
        print(f"Using base Permutation threshold of {optimal_threshold}")
        return optimal_threshold

def permutation(data, geneNames, group1_size, permutation_threshold, num_permutations):
    """
    Filters genes out with the use of a Permutation test

    Parameters 
    ----------
    data: 2D np array 
        Stores the normalized count expression data
    geneNames: 1D np string array
        Stores the names of genes 
    group1_size: int 
        The number of samples in the first category/group
    permutation_threshold: float64
        The permutation test p-value for filtering out genes
    num_permutation: int
        The number of permutation completed for the test 

    Returns
    -------
    data: 2D np array 
        Stores the normalized, filtered count expression data
    geneNames: 1D np string array
        Stores the filtered names of genes 
    """
    print(f"Running permutation test on a {permutation_threshold} threshold...")

    #gets the mean difference between samples per gene
    group1_mean = np.mean(data[:, :group1_size], axis=1)
    group2_mean = np.mean(data[:, group1_size:], axis=1)
    gmeanDiff = group1_mean - group2_mean

    #defines a counter to keep track of number of times the permuted mean is greater than the mean difference
    counter_list = np.zeros(len(gmeanDiff))

    #goes through the permutations and tracks the number of times the permuted mean difference is greater than the actual mean difference 
    for num in range(num_permutations): 
        shuffled = np.take(data, np.random.permutation(data.shape[1]), axis=1)
        permuted_stat = np.mean(shuffled[:, :group1_size], axis = 1) - np.mean(shuffled[:, group1_size:], axis = 1)
        counter_list += np.abs(permuted_stat) >= np.abs(gmeanDiff)
    
    #gets the p value [proportion of times permuted mean was greater than mean difference]
    p_value_perm = counter_list / num_permutations

    #filters genes based on the pvalues 
    mask = p_value_perm <= permutation_threshold
    geneNames = geneNames[mask]
    data = data[mask, :]
    return (data, geneNames)