import numpy as np
import yaml
import os


def coarsen_density(density, I, h, bin_size):
    assert bin_size in [1, 2, 4, 8, 20]
    assert density.shape == (I, I)

    # Bin size 1 has no effect
    if bin_size == 1:
        return density

    # New control area and number of nodes
    coarse_h = h * bin_size
    coarse_I = I // bin_size

    # Initialize coarsened grid
    coarse_density = np.zeros(coarse_I**2)

    # Convert density to cell counts
    density *= h**2

    # Compute coarsened density by combining cell counts in subarrays of shape (bin_size, bin_size)
    for i in np.arange(coarse_I):
        for j in np.arange(coarse_I):
            coarse_density[i * coarse_I + j] = density[
                i * bin_size : i * bin_size + bin_size - 1,
                j * bin_size : j * bin_size + bin_size - 1,
            ].sum()

    # Divide cell counts by area to obtain density
    coarse_density /= coarse_h**2

    return coarse_density


def compile_data(dataset_num):
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
            
            c_a_range = np.logspace(np.log10(c_a_base / 2), np.log10(2 * c_a_base), 11)
            eta1_range = np.logspace(
                np.log10(eta1_base / 2), np.log10(2 * eta1_base), 11
            )
            for itr in np.arange(10):
                file_name = f"../data/dataset1/c_a={c_a_range[i]}_eta1={eta1_range[j]}_itr={itr}.npz"
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
                    np.log10(param1_base / 2), np.log10(2 * param1_base), 11
                )
                param2_range = np.logspace(
                    np.log10(param2_base / 2), np.log10(2 * param2_base), 11
                )
                for itr in np.arange(10):
                    file_name = f"../data/dataset2/({param1},{param2})/{param1}={param1_range[i]}_{param2}={param2_range[j]}_itr={itr}.npz"
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
    assert itr_cutoff in np.arange(1, 10)
    X_train = []
    X_test = []

    for i, sample in enumerate(X):
        if i % 10 < itr_cutoff:
            X_train.append(sample)
        elif i % 10 >= itr_cutoff:
            X_test.append(sample)

    return X_train, X_test


def scale_density(X, coarseness=1):

    X_scaled = np.copy(X)

    # Normalize by pixel
    means = np.mean(X, axis=0)

    # Scale variance by whole dataset
    std = np.std(X)

    X_scaled -= means
    X_scaled /= std

    return X_scaled


def scale_population_curves(X):
    X_scaled = np.copy(X)
    for subpopulation in range(4):
        population_curve = X_scaled[:, subpopulation, :]

        # Normalize by time point
        means = np.mean(population_curve, axis=0)
        std = np.std(population_curve)

        X_scaled[:, subpopulation, :] -= means
        X_scaled[:, subpopulation, :] /= std

    return X_scaled
