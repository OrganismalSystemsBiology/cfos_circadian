
"""Analytic Cosinor Test Module

This module provides functions for detecting rhythmic patterns in time-series data 
using analytic cosinor testing. The primary functionality calculates the maximum 
Pearson correlation between input data and sine/cosine waves of a specified period,
along with statistical significance testing.

The module implements the analytic cosinor test which:
1. Generates normalized orthogonal sine and cosine basis vectors
2. Calculates the maximum correlation between input data and these basis vectors
3. Computes p-values for statistical significance
4. Optionally adjusts for standard error of the mean (SEM)

Main Functions:
    costest: Performs analytic cosinor test on a single time-series vector
    batch_costest: Vectorized version for processing multiple time-series at once

The method is particularly useful for detecting circadian or other periodic patterns
in biological and time-series data.

Author: ygriku
License: See LICENSE file
"""

import numpy as np

# Generate normalized orthogonal basis vectors
def _norm_bases(n_tp, n_tp_in_per):
    """Generate normalized sine and cosine basis vectors for correlation analysis.
    
    Args:
        n_tp (int): Total number of time points in the data
        n_tp_in_per (float): Number of time points in one period
        
    Returns:
        tuple: (normalized_sine_wave, normalized_cosine_wave) - orthogonal basis vectors
    """
    base_s = np.sin(np.arange(n_tp)/n_tp_in_per*2*np.pi)
    base_c = np.cos(np.arange(n_tp)/n_tp_in_per*2*np.pi)
    base_s_m = base_s - np.mean(base_s)
    base_c_m = base_c - np.mean(base_c)
    norm_base_s = (base_s_m)/np.sqrt(np.sum((base_s_m)*(base_s_m)))
    norm_base_c = (base_c_m)/np.sqrt(np.sum((base_c_m)*(base_c_m)))
    return norm_base_s, norm_base_c


# Calculate maximum correlation between input vector and basis vectors
def _calc_max_corr(vec, norm_base_s, norm_base_c):
    """Calculate the maximum correlation between an input vector and sine/cosine basis vectors.
    
    Args:
        vec (array): Input time-series vector
        norm_base_s (array): Normalized sine basis vector
        norm_base_c (array): Normalized cosine basis vector
        
    Returns:
        tuple: (max_correlation_value, phase_of_max_correlation) in radians
    """
    vec_m = vec - np.mean(vec)
    norm_vec = (vec_m)/np.sqrt(np.sum((vec_m)*(vec_m)))
    max_corr_value = np.sqrt(np.power(np.dot(norm_base_c, norm_vec), 2) + np.power(np.dot(norm_base_s, norm_vec), 2))
    # Ensure the maximum correlation does not exceed 1 due to possible numerical subtleties
    max_corr_value = min(1.0, max_corr_value)
    max_corr_phase = np.arctan2(np.dot(norm_base_s, norm_vec), np.dot(norm_base_c, norm_vec))
    return max_corr_value, max_corr_phase


# Calculate p-value for maximum correlation
def _max_corr_pval(num_datapoints, max_corr):
    """Calculate the p-value for the maximum Pearson correlation.
    
    Args:
        num_datapoints (int): Number of data points in the time series
        max_corr (float): Maximum correlation value
        
    Returns:
        float: P-value (probability under null hypothesis)
    """
    n = num_datapoints - 3
    p = np.power((1 - np.power(max_corr, 2)), n / 2)
    return p


# Calculate the SEM adjustment ratio
def _sem_ratio(avg_vector, sem_vector, sem_weight='low'):
    """Calculate the ratio for adjusting correlation based on standard error of the mean.
    
    This function computes how much the correlation should be adjusted based on the
    confidence interval defined by the SEM values, using a middle, high, or low weights:
        high:    95% confidence level.
        middle: 50% confidence level.
        low:     3% confidence level. (default)

    Args:
        avg_vector (array): Average values at each time point
        sem_vector (array): Standard error of the mean values at each time point
        sem_weight (str): Method for weighting SEM ('middle', 'high', 'low')
    Returns:
        float: Adjustment ratio (0.0 to 1.0) for the correlation value
    """
    if sem_weight == 'high':
        ci_factor = 1.96  # 95% confidence interval
    elif sem_weight == 'middle':
        ci_factor = 0.67  # 50% confidence interval
    else:
        ci_factor = 0.0375  # 3% confidence interval (default)


    a_vec = avg_vector - np.mean(avg_vector)
    ratio = 1.0 - ci_factor / np.sqrt(np.sum(np.power(a_vec / sem_vector, 2))) # ci_factor is the confidence interval
    return max(0, ratio)


# Main function for the analytic cosinor test
def costest(avg_vec, n_tp_in_per, sem_vec=None, sem_weight='low'):
    """Perform analytic cosinor test on a time-series vector.
    
    Calculates the maximum Pearson correlation between the input vector and 
    sine/cosine curves with the specified period, along with statistical significance.
    
    Args:
        avg_vec (array): Vector of averaged values at each time point
        n_tp_in_per (float): Number of time points in one period
        sem_vec (array, optional): Vector of standard error of the mean values 
                                  at each time point
        sem_weight (str): Weighting SEM ('middle', 'high', 'low'), default is 'low'
    
    Returns:
        tuple: (max_correlation, phase_radians, original_p_value, sem_adjusted_p_value)
            - max_correlation (float): Maximum correlation value
            - phase_radians (float): Phase in radians of the maximum correlation
            - original_p_value (float): Original p-value
            - sem_adjusted_p_value (float): SEM-adjusted p-value
    """
    # Get the number of time points in the data
    n_tp = len(avg_vec)

    # prepare bases
    norm_base_s, norm_base_c = _norm_bases(n_tp, n_tp_in_per)

    # prepare vector to handle nan
    avg_vector = avg_vec.copy()

    # replace nan in AVG with median
    avg_vector[np.isnan(avg_vector)] = np.nanmedian(avg_vector)

    if sem_vec is not None:
        ## prepare vector to handle nan
        sem_vector = sem_vec.copy()

        # each SEM needs to be >0
        sem_vector[~(sem_vector>0)] = np.nanmax(sem_vector)
        # replace nan in SEM with an effectively infinite SEM for missing averages
        sem_vector[np.isnan(avg_vector)] = np.nanmax(sem_vector)*1000000
        # taking the SEM into account
        sem_r = _sem_ratio(avg_vector, sem_vector, sem_weight)
    else:
        # if no SEM is provided, sem_r is set to 1.0
        sem_r = 1.0

    # tuple of max-correlation value and the phase of max correlation
    mc, mc_ph = _calc_max_corr(avg_vector, norm_base_s, norm_base_c)

    adj_mc = mc * sem_r
    p_org = _max_corr_pval(n_tp, mc)
    p_sem_adj = _max_corr_pval(n_tp, adj_mc)
    
    # max-correlation value, phase of max correlation, original p, SEM adjusted p
    return mc, mc_ph, p_org, p_sem_adj


# Vectorized batch version of the costest function
def batch_costest(avg_vec_matrix, n_tp_in_per, sem_vec_matrix=None, sem_weight='low'):
    """Perform analytic cosinor test on multiple time-series vectors.
    
    Calculates the maximum Pearson correlation between each input vector and 
    sine/cosine curves with the specified period, along with statistical significance.
    This vectorized implementation is more efficient for processing large datasets.

    Args:
        avg_vec_matrix (array): 2D array where each row is a vector of averaged 
                               values at each time point
        n_tp_in_per (float): Number of time points in one period
        sem_vec_matrix (array, optional): 2D array where each row is a vector of 
                                         standard error of the mean values at each time point
        sem_weight (str): Weighting SEM ('low', 'high', 'middle'), default is 'low'

    Returns:
        ndarray: Array with shape (n_vectors, 4) containing for each input vector:
            - Column 0: Maximum correlation value
            - Column 1: Phase in radians of the maximum correlation
            - Column 2: Original p-value
            - Column 3: SEM-adjusted p-value
    """
    # Convert to numpy array if not already
    avg_vec_matrix = np.asarray(avg_vec_matrix)
    n_vectors, n_tp = avg_vec_matrix.shape

    # prepare bases once
    norm_base_s, norm_base_c = _norm_bases(n_tp, n_tp_in_per)

    # Vectorized NaN handling for avg_vec_matrix
    avg_matrix_clean = avg_vec_matrix.copy()
    nan_mask = np.isnan(avg_matrix_clean)
    # Replace NaN with median for each row
    for i in range(n_vectors):
        if np.any(nan_mask[i]):
            avg_matrix_clean[i, nan_mask[i]] = np.nanmedian(avg_matrix_clean[i])

    # Vectorized correlation calculation
    # Center the data
    avg_means = np.mean(avg_matrix_clean, axis=1, keepdims=True)
    avg_centered = avg_matrix_clean - avg_means
    
    # Normalize the vectors
    avg_norms = np.sqrt(np.sum(avg_centered**2, axis=1, keepdims=True))
    norm_avg = avg_centered / avg_norms
    
    # Calculate correlations with sine and cosine bases
    corr_s = np.dot(norm_avg, norm_base_s)
    corr_c = np.dot(norm_avg, norm_base_c)
    
    # Calculate max correlation and phase
    mc_array = np.sqrt(corr_s**2 + corr_c**2)
    mc_array = np.minimum(mc_array, 1.0)  # Ensure max correlation doesn't exceed 1
    mc_ph_array = np.arctan2(corr_s, corr_c)
    
    # Calculate original p-values vectorized
    p_org_array = np.power((1 - mc_array**2), (n_tp - 3) / 2)
    
    # Handle SEM adjustments vectorized
    sem_r_array = np.ones(n_vectors)
    if sem_vec_matrix is not None:
        sem_vec_matrix = np.asarray(sem_vec_matrix)
        sem_matrix_clean = sem_vec_matrix.copy()
        
        # Vectorized SEM handling
        sem_nonpositive = ~(sem_matrix_clean > 0)
        sem_nan = np.isnan(sem_matrix_clean)
        
        # Replace non-positive and NaN values
        for i in range(n_vectors):
            sem_max = np.nanmax(sem_matrix_clean[i])
            sem_matrix_clean[i, sem_nonpositive[i]] = sem_max
            sem_matrix_clean[i, sem_nan[i]] = sem_max * 1000000
            
            # Calculate SEM ratio
            ratio_vec = avg_centered[i] / sem_matrix_clean[i]
            sem_r_array[i] = max(0, 1.0 - 0.67 / np.sqrt(np.sum(ratio_vec**2)))
    
        # Handle SEM weights
        if sem_vec_matrix is not None:
            if sem_weight == 'high':
                ci_factor = 1.96  # 95% confidence interval
            elif sem_weight == 'middle':
                ci_factor = 0.67  # 50% confidence interval
            else:
                ci_factor = 0.0375  # 3% confidence interval (default)

        for i in range(n_vectors):
            a_vec = avg_matrix_clean[i] - np.mean(avg_matrix_clean[i])
            ratio = 1.0 - ci_factor / np.sqrt(np.sum(np.power(a_vec / sem_matrix_clean[i], 2))) # ci_factor is the confidence interval
            sem_r_array[i] = max(0, ratio)
    # Calculate SEM-adjusted correlations and p-values
    adj_mc_array = mc_array * sem_r_array
    p_sem_adj_array = np.power((1 - adj_mc_array**2), (n_tp - 3) / 2)

    return np.column_stack([mc_array, mc_ph_array, p_org_array, p_sem_adj_array])