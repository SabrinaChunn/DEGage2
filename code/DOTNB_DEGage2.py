import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod import families
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter 
from scipy.special import gammaln, hyp2f1 as hyp2f1_sc
from sympy.functions.special.hyper import hyper

import warnings
warnings.simplefilter("ignore", category=sm.tools.sm_exceptions.HessianInversionWarning)
warnings.simplefilter("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)
np.seterr(over='ignore', invalid='ignore')

def hyp_funct(w, x, y, z):
    """
    Runs hypergeometric function calculation with either SciPy or SymPy library 

    Parameters 
    ----------
    w: float64
        r1+k or r2+nk
    x: float64
        r2 or r1
    y: float64
        k+1 or nk + 1
    z: float64
        q1*q2 
        
    Returns
    -------
    result: float64
        Hypergeometric value result

    """

    result = hyp2f1_sc(w, x, y, z) #first uses scipy's library
    if np.isinf(result) or np.isnan(result) or result <= 0: #then utilizes sympy as a fallback
        result = float(hyper([w, x], [y], z).evalf(200))
    return result

def DOTNB_mean(r1, p1, r2, p2):
    """
    Determines the difference between the condition's negative binomial distribution means 

    Parameters 
    ----------
    r1: float64
        r parameter for condition 1
    p1: float64
        p parameter for condition 1
    r2: float64
        r parameter for condition 2
    p2: float64
        p parameter for condition 2
        
    Returns
    -------
    diff_mean: float64
        DOTNB mean or difference of condition means 

    """

    mean1 = r1 * (1 - p1) / p1
    mean2 = r2 * (1 - p2) / p2
    diff_mean = mean1 - mean2
    return diff_mean

def DOTNB_pmf(r1, p1, r2, p2, k):
    """
    Calculates the probability that k, the test statistic, is the difference of the condition's negative binomial distributions

    Parameters 
    ----------
    r1: float64
        r parameter for condition 1
    p1: float64
        p parameter for condition 1
    r2: float64
        r parameter for condition 2
    p2: float64
        p parameter for condition 2
    k: float64
        test statistic between conditions 
        
    Returns
    -------
    d: float64
        probability of k statistic being the difference between distributions 

    """

    q1 = 1 - p1
    q2 = 1 - p2
    d = 0
    if k > 0:
        d_val = (p1**r1)*(p2**r2)*(q1**k)
        gamma_val = np.exp(gammaln(r1 + k) - gammaln(r1) - gammaln(1 + k))
        hyper_val = hyp_funct(r1+k, r2, k+1, q1*q2)
        d = d_val * gamma_val * hyper_val
    else:
        nk = -k
        d_val = (p1**r1)*(p2**r2)*(q2**nk)
        gamma_val = np.exp(gammaln(r2 + nk) - gammaln(r2) - gammaln(1 + nk))
        hyper_val = hyp_funct(r2+nk, r1, nk+1, q1*q2)
        d = d_val * gamma_val * hyper_val
    d = float(d)
    if np.isnan(d):
        d = 0
    return d
    
def DOTNB_cdf(r1, p1, r2, p2, k):
    """
    Calculates the probability of values <=k representing the difference between the condition's negative binomial distributions 

    Parameters 
    ----------
    r1: float64
        r parameter for condition 1
    p1: float64
        p parameter for condition 1
    r2: float64
        r parameter for condition 2
    p2: float64
        p parameter for condition 2
    k: float64
        test statistic between conditions 
        
    Returns
    -------
    cvalue: float64
        cumulative probability of values <= k statistic being the difference between distributions 

    """

    cvalue = 0
    pdfv = DOTNB_pmf(r1, p1, r2, p2, k)
    TOLERANCE = 1.0e-200
    i = 0
    maxiter = 1000
    
    if k > 0: #flipps positive k values [neg log fold chngs to be positive treated; avoids issues]
        k = k * -1
        r1, p1, r2, p2 = r2, p2, r1, p1
    
    if pdfv < TOLERANCE:
        if pdfv > DOTNB_mean(r1, p1, r2, p2):
            cvalue = 1
        else:
            cvalue = 0
    else:
        k_values = np.arange(k, k - maxiter, -1)
        for i, ki in enumerate(k_values):
            pdfv = DOTNB_pmf(r1, p1, r2, p2, ki)
            if pdfv < TOLERANCE:
                break
            cvalue += pdfv
                
    return cvalue