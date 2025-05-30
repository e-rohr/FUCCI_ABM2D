import numpy as np
import yaml
import os


def coarsen_density(density, I, h, bin_size):
    """
    Coarsens a 2D density grid by aggregating values in non-overlapping blocks.

    Parameters
    ----------
    density : ndarray of shape (I, I)
        The 2D array of density values on a fine grid.
    I : int
        Number of grid points along one dimension of the original square grid (density.shape must be (I, I)).
    h : float
        Grid spacing of the fine grid.
    bin_size : int
        Size of the coarsening bin. Must be one of [1, 2, 4, 8, 10, 20]. A value of 1 returns the input unchanged.

    Returns
    -------
    coarse_density : ndarray of shape (coarse_I * coarse_I,)
        A 1D array representing the coarsened density values, where coarse_I = I // bin_size. Each element corresponds to the average density over a (bin_size x bin_size) block in the original grid.

    Notes
    -----
    The function computes the average density in each non-overlapping bin by summing the values in a
    (bin_size x bin_size) sub-block of the original array, then dividing by bin area (bin_size^2).

    Raises
    ------
    AssertionError
        If bin_size is not one of the allowed values or if density.shape != (I, I).
    """
    
    assert bin_size in [1, 2, 4, 8, 10, 20]
    assert density.shape == (I, I)

    # Bin size 1 has no effect
    if bin_size == 1:
        return density

    # New control area and number of nodes
    coarse_h = h * bin_size
    coarse_I = I // bin_size

    # Initialize coarsened grid
    coarse_density = np.zeros(coarse_I**2)

    # Compute coarsened density by combining cell counts in subarrays of shape (bin_size, bin_size)
    for i in np.arange(coarse_I):
        for j in np.arange(coarse_I):
            coarse_density[i * coarse_I + j] = density[
                i * bin_size : i * bin_size + bin_size,
                j * bin_size : j * bin_size + bin_size,
            ].sum()

    # Divide cell counts by area to obtain density
    coarse_density /= bin_size**2

    return coarse_density


def compile_data(dataset_num):
    """
    Loads simulation data from `.npz` files, extracts density and cell count information,
    and saves the compiled results into `.npy` files for the specified dataset.

    Parameters
    ----------
    dataset_num : int
        The dataset number to compile, must be either 1 or 2.

        - If 1: Uses fixed parameters `c_a` and `eta1` from the config file and loops over a
          predefined 11x11 grid of parameter combinations.
        - If 2: Uses parameter combinations specified in the config file and processes each one
          similarly over an 11x11 grid.

    Returns
    -------
    None
        The function saves two NumPy arrays to disk:
        - `density.npy`: list of density arrays (`out['v']`) from each run.
        - `cell_counts.npy`: list of 4-element arrays containing cell counts (`Nr`, `Ny`, `Ng`, `Nd`).

    Output Files
    ------------
    ../data/dataset{dataset_num}/density.npy
    ../data/dataset{dataset_num}/cell_counts.npy

    Notes
    -----
    - Requires a YAML configuration file at `../src/parameters.yaml`.
    - Assumes simulation output files are located at:
        - `../data/dataset1/c_a=<...>_eta1=<...>_itr=<...>.npz` for dataset 1
        - `../data/dataset2/({param1},{param2})/{param1}=<...>_{param2}=<...>_itr=<...>.npz` for dataset 2
    - Each parameter varies over 11 logarithmically spaced values centered around its base value.

    Raises
    ------
    AssertionError
        If dataset_num is not 1 or 2.
    FileNotFoundError
        If expected `.npz` files are missing.
    KeyError
        If specified parameters or combinations are not present in the YAML configuration.
    """
    assert dataset_num in [1,2]

    with open("../src/parameters.yaml") as p:
        params = yaml.safe_load(p)
        dataset_info = params[f"dataset{dataset_num}"]

    if dataset_num == 1:

        cell_counts = []
        density = []
        
        for index in np.arange(121):
            i, j = divmod(index, 11)
            c_a_base = dataset_info["parameters"]["c_a"]
            eta1_base = dataset_info["parameters"]["eta1"]
            
            c_a_range = np.logspace(np.log2(c_a_base / 2), np.log2(2 * c_a_base), 11, base = 2)
            eta1_range = np.logspace(
                np.log2(eta1_base / 2), np.log2(2 * eta1_base), 11, base = 2
            )
            for itr in np.arange(10):
                file_name = f"../data/dataset1/c_a={c_a_range[i]:.5f}_eta1={eta1_range[j]:.5f}_itr={itr}.npz"
                out = np.load(file_name, allow_pickle = True)
                next_density_to_add = out['v']
                density.append(next_density_to_add)
                
                next_cell_counts_to_add = np.array([out['Nr'],
                                                    out['Ny'],
                                                    out['Ng'],
                                                    out['Nd']])
                cell_counts.append(next_cell_counts_to_add)
    
    if dataset_num == 2:
        cell_counts = []
        density = []
        for combo in dataset_info["parameter_combinations"]:
            param1, param2 = combo.split(',')
            
            for index in np.arange(121):
                i, j = divmod(index, 11)
                param1_base = dataset_info["parameters"][param1]
                param2_base = dataset_info["parameters"][param2]
            
                param1_range = np.logspace(
                    np.log2(param1_base / 2), np.log2(2 * param1_base), 11, base = 2
                )
                param2_range = np.logspace(
                    np.log2(param2_base / 2), np.log2(2 * param2_base), 11, base = 2
                )
                for itr in np.arange(10):
                    file_name = f"../data/dataset2/({param1},{param2})/{param1}={param1_range[i]:.5f}_{param2}={param2_range[j]:.5f}_itr={itr}.npz"
                    out = np.load(file_name, allow_pickle = True)
                    next_density_to_add = out['v']
                    density.append(next_density_to_add)
                
                    next_cell_counts_to_add = np.array([out['Nr'],
                                                    out['Ny'],
                                                    out['Ng'],
                                                    out['Nd']])
                    cell_counts.append(next_cell_counts_to_add)
        
    np.save(f"../data/dataset{dataset_num}/density.npy", density, allow_pickle = True)
    np.save(f"../data/dataset{dataset_num}/cell_counts.npy",cell_counts, allow_pickle = True)
    return

def split_data(X, itr_cutoff):
    """
    Splits data into training and testing subsets based on simulation iteration index.

    Parameters
    ----------
    X : array-like or list
        A sequence of data samples, assumed to be ordered such that every consecutive block of 10
        samples corresponds to parameter combinations with 10 different simulation iterations (itr = 0 to 9).
    itr_cutoff : int
        The number of iterations (0-based) to include in the training set for each parameter combination.
        Must be an integer between 1 and 9 (inclusive).

    Returns
    -------
    X_train : ndarray
        Subset of `X` corresponding to iterations with index less than `itr_cutoff`.
    X_test : ndarray
        Subset of `X` corresponding to iterations with index greater than or equal to `itr_cutoff`.

    Notes
    -----
    This function assumes that the data is structured such that every 10 samples correspond to a single
    parameter combination and vary only in iteration index from 0 to 9. It performs a modulo-based split
    using `i % 10`.
    """
    assert itr_cutoff in np.arange(1, 10)
    X_train = []
    X_test = []

    for i, sample in enumerate(X):
        if i % 10 < itr_cutoff:
            X_train.append(sample)
        elif i % 10 >= itr_cutoff:
            X_test.append(sample)

    return np.array(X_train), np.array(X_test)


def scale_density(X_train, X_test, X=[]):
    """
    Normalizes density data by centering and scaling based on training statistics.

    This function standardizes the training and test datasets using the mean and standard deviation
from the training set, and optionally standardizes a third dataset `X` using its own statistics.

    Parameters
    ----------
    X_train : ndarray
        Training data of shape (n_train, ...) to be normalized.
    X_test : ndarray
        Test data of shape (n_test, ...) to be normalized using statistics from `X_train`.
    X : ndarray, optional
        Additional dataset to normalize using its **own** mean and standard deviation.
        Defaults to an empty list, in which case it is ignored.

    Returns
    -------
    Xt_train : ndarray
        Normalized version of `X_train`.
    Xt_test : ndarray
        Normalized version of `X_test`.
    Xt : ndarray
        Normalized version of `X`. If `X` is empty, this will be an empty array.

    Notes
    -----
    - `X_train` and `X_test` are scaled using `X_train`'s mean and standard deviation.
    - `X` is scaled using its own mean and standard deviation, **not** the training statistics.
    - The normalization is done feature-wise (e.g., per pixel if images).
    - This function returns copies; the original arrays are not modified.

    Raises
    ------
    ValueError
        If `X_train` or `X_test` is empty or not compatible for broadcasting.
    """

    Xt_train = np.copy(X_train)
    Xt_test = np.copy(X_test)
    Xt = np.copy(X)
    
    # Normalize by pixel
    means = np.mean(X_train, axis=0)

    # Scale variance by whole dataset
    std = np.std(X_train)

    Xt_train -= means
    Xt_train /= std
    
    Xt_test -= means
    Xt_test /= std
    
    # Normalize by pixel
    means = np.mean(X, axis=0)

    # Scale variance by whole dataset
    std = np.std(X)
    
    Xt -= means
    Xt /= std
        
    return Xt_train, Xt_test, Xt


def scale_cell_counts(X_train, X_test, X):
    """
    Standardizes cell count time series data for multiple subpopulations across training, test,
    and auxiliary datasets.

    Each subpopulation's time series is normalized using the mean (per time step) and
    standard deviation (overall) computed from the training set.

    Parameters
    ----------
    X_train : ndarray
        Training data of shape (n_train, 4, T), where 4 is the number of subpopulations
        and T is the number of time points.
    X_test : ndarray
        Test data of the same shape as `X_train`.
    X : ndarray
        Additional dataset (e.g., full dataset) of the same shape to normalize using the
        statistics computed from `X_train`.

    Returns
    -------
    Xt_train : ndarray
        Normalized training data.
    Xt_test : ndarray
        Normalized test data using training statistics.
    Xt : ndarray
        Normalized `X` using training statistics.

    Notes
    -----
    - Each subpopulation (along axis 1) is normalized independently.
    - Mean is computed per time step (across samples), but the standard deviation is global
      (across all values for that subpopulation).
    - All outputs are copies; the original arrays are not modified.

    Raises
    ------
    ValueError
        If input arrays do not match the expected shape (n, 4, T).
    """


    
    Xt_train = np.copy(X_train)
    Xt_test = np.copy(X_test)
    Xt = np.copy(X)
    
    for subpopulation in range(4):
        subpopulation_curve = Xt_train[:, subpopulation, :]
        
        # Normalize by time point
        means = np.mean(subpopulation_curve, axis=0)
        std = np.std(subpopulation_curve)

        Xt_train[:, subpopulation, :] -= means
        Xt_train[:, subpopulation, :] /= std
        
        Xt_test[:, subpopulation, :] -= means
        Xt_test[:, subpopulation, :] /= std
        
        Xt[:, subpopulation, :] -= means
        Xt[:, subpopulation, :] /= std
        

    return Xt_train, Xt_test, Xt
