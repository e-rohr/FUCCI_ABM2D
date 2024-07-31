import yaml
import numpy

def write_param_combos_to_yaml():
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
    

def parameter_distributions(labels,cluster_num):
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
                np.log10(param_bases[param_name] / 2), np.log10(2 * param_bases[param_name]), 11
            )
    
    for combo_index, combo in enumerate(param_combos):
        param1, param2 = combo.split(',')
        for param_index in np.arange(121):
            param1_index, param2_index = divmod(param_index, 11)
            for itr in np.arange(10):
                if labels[combo_index*1210 + param_index*11 + itr] == cluster_num:
                    param_distns[param1].append(
                        np.log10(param_ranges[param1][param1_index]/param_bases[param_1]) 
                    )
                    param_distns[param2].append(
                        np.log10(param_ranges[param2][param2_index]/param_bases[param_2]) 
                    )
                    
                    
                        # transformed_value = np.log2(samples[combo][param_index, pair_loc]/param_base) 
                        # param_distn.append(transformed_value)
                        # param_distn.append(samples[combo][param_index, pair_loc]/param_base)
    
    for param_name in param_bases.keys():
        param_distns[param_name] = np.array(param_distns[param_name])
    
    return param_distns 