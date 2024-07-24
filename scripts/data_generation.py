import sys
import os
from src.abm2d import abm2d
import numpy as np
import yaml

if __name__ == "__main__":
    if len(sys.argv) > 2:
        dataset_num = int(sys.argv[1])
        index = int(sys.argv[2])
        

        assert dataset_num in [1, 2]

        if dataset_num == 1:

            # index = 0 ... 120
            assert index in np.arange(0, 121)

            # i,j are row and column indices for (η1, c_a) parameter grid
            i, j = divmod(index, 11)

            with open("../src/parameters.yaml") as p:
                params = yaml.safe_load(p)
                c_a_base = params["dataset1"]["parameters"]["c_a"]
                eta1_base = params["dataset1"]["parameters"]["eta1"]

            # 121x121 logspaced grid of parameter combinations
            c_a_range = np.logspace(np.log10(c_a_base / 2), np.log10(2 * c_a_base), 11)
            eta1_range = np.logspace(
                np.log10(eta1_base / 2), np.log10(2 * eta1_base), 11
            )

            # Simulate 10 realizations of ABM with the parameter combination
            for itr in np.arange(10):
                args = {
                    "c_a": c_a_range[i],
                    "eta1": eta1_range[j],
                    "path": "../data/dataset1",
                    "title": f"c_a={c_a_range[i]}_eta1={eta1_range[j]}_itr={itr}",
                }
                if not os.path.isdir(args['path']):
                    os.makedirs(args["path"])

                abm2d(**args)

        elif dataset_num == 2:

            # index = 0 ... 121
            assert index in np.arange(0, 121)

            # i,j are row and column indices for (η1, c_a) parameter grid
            i, j = divmod(index, 11)

            with open("../src/parameters.yaml") as p:
                params = yaml.safe_load(p)
                dataset2_params = params["dataset2"]["parameters"]

            for param1 in dataset2_params.keys():
                for param2 in dataset2_params.keys():
                    if param2 == param1:
                        continue

                    param1_base = dataset2_params[param1]
                    param2_base = dataset2_params[param2]

                    param1_range = np.logspace(
                        np.log10(param1_base / 2), np.log10(2 * param1_base), 11
                    )
                    param2_range = np.logspace(
                        np.log10(param2_base / 2), np.log10(2 * param2_base), 11
                    )

                    # Simulate 10 realizations of ABM with the parameter combination
                    for itr in np.arange(10):
                        args = {
                            "c_a": param1_range[i],
                            "eta1": param2_range[j],
                            "path": "../data/dataset2",
                            "title": f"{param1}={param1_range[i]}_{param2}={param2_range[j]}_itr={itr}",
                        }
                        if not os.path.isdir(args['path']):
                            os.makedirs(args["path"])

                        abm2d(**args)
