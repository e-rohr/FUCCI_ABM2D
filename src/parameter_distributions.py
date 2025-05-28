import yaml
import numpy as np

def write_param_combos_to_yaml():
    """
    Extracts all unique ordered pairs of parameter names from dataset2 in the YAML file
    and writes them back to the same YAML file under the key 'parameter_combinations'.

    The function skips pairs where both parameters are the same (e.g., (c_a, c_a)).

    It modifies the existing YAML file in-place by adding or overwriting
    the 'parameter_combinations' field under 'dataset2'.

    Returns
    -------
    None
    """
    with open("../src/parameters.yaml") as p:
        params_yml = yaml.safe_load(p)
        params = params_yml["dataset2"]["parameters"]
    
    param_combos = []
    for param1_itr in params.keys():
        for param2_itr in params.keys():
            if param2_itr == param1_itr:
                continue
            else:
                param_combos.append(f"{param1_itr},{param2_itr}")
    
    param_combos_yml = yaml.dump(param_combos, default_flow_style=False)
    with open('../src/parameters.yaml', 'w') as p:
        yaml.dump(param_combos_yml, p)
    
    return
    

def parameter_distributions(labels, cluster_num, n_itrs):
    """
    Computes the distribution of parameter values (log2-scaled relative to base) 
    for a specific cluster across all parameter combinations.

    Parameters
    ----------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each sample in the full dataset (ordered by combination, index, then iteration).
    cluster_num : int
        The specific cluster label to compute distributions for.
    n_itrs : int
        Number of iterations per parameter pair and setting (usually 10).

    Returns
    -------
    param_distns : dict of {str: ndarray}
        Dictionary mapping each parameter name to an array of log2-scaled relative values
        (i.e., `log2(actual / base)`) for samples assigned to the specified cluster.

    Notes
    -----
    - Assumes `parameters.yaml` includes both:
        - `dataset2.parameters`: dict of base values for parameters.
        - `dataset2.parameter_combinations`: list of comma-separated param name pairs.
    - Values are normalized by their base and log2-scaled to represent fold-change.
    - Each parameter's array may be empty if no corresponding values were assigned to the cluster.
    """
    with open("../src/parameters.yaml") as p:
        params_yml = yaml.safe_load(p)
        param_bases = params_yml["dataset2"]["parameters"]
        param_combos = params_yml["dataset2"]["parameter_combinations"]
    
    # Initialize dictionary of distributions and value ranges for each parameter
    param_distns = {}
    param_ranges = {}
    for param_name in param_bases.keys():
        param_distns[param_name] = []
        param_ranges[param_name] = np.logspace(
                np.log2(param_bases[param_name] / 2), np.log2(2 * param_bases[param_name]), 11, base = 2
            )
    
    for combo_index, combo in enumerate(param_combos):
        param1, param2 = combo.split(',')
        for param_index in np.arange(121):
            param1_index, param2_index = divmod(param_index, 11)
            for itr in np.arange(n_itrs):
                if labels[combo_index*121*n_itrs + param_index*n_itrs + itr] == cluster_num:
                    param_distns[param1].append(
                        np.log2(param_ranges[param1][param1_index]/param_bases[param1]) 
                    )
                    param_distns[param2].append(
                        np.log2(param_ranges[param2][param2_index]/param_bases[param2]) 
                    )
    
    for param_name in param_bases.keys():
        param_distns[param_name] = np.array(param_distns[param_name])
    
    return param_distns 