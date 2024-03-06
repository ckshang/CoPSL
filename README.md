# CoPSL

This is the code of paper: **Collaborative Pareto Set Learning in Multiple Multi-Objective Optimization Problems**.

The code is mainly designed to be simple and readable, it contains:

- `run_copsl.py` is a ~320-line script to run the CoPSL algorithms (including Pareto front visualization);
- `run_copsl_gn.py` is a ~240-line script to run the CoPSL algorithms with the GradNorm strategy;
- `run_psl.py` is a ~310-line script to run the PSL algorithms (including Pareto front visualization);
- `run_emo.py` is a ~120-line script to run the EMO algorithms;
- `model.py` contains both single-task and multi-task architecture models;
- `problem.py` includes all the test problems used in the paper for running PSL algorithms;
- `problem_emo.py` also contains all the test problems used in the paper for running EMO algorithms;
- The folder `pf_re` contains the files related to the approximate Pareto fronts.

### Algorithms

- PSL-LS
- PSL-COSMOS
- PSL-TCH
- PSL-MTCH
- NSGA-II
- NSGA-III
- MOEA/D

### Benchmarks

- Two-dimensional synthetic problems: F1 to F6;
- Three-dimensional real-world engineering design problems: RE31, RE32, RE33, RE34, RE37.

### Parameters

[//]: # (The following arguments to the `./run_copsl.py` file control the important parameters of the experiment.)
The parameters specified in the `./run_copsl.py` file are as follows:

| Parameter           | Description                                                |
|---------------------|------------------------------------------------------------|
| `ins_list`          | List of test problems.                                     |
| `n_run`             | Number of independent run.                                 |
| `loss_func`         | The loss function. Options: `ls`, `cosmos`, `tch`, `mtch`. |
| `n_steps`           | Number of learning steps.                                  |
| `n_pref_update`     | Number of sampled preferences per step.                    |
| `lr`                | Learning rate.                                             |
| `gamma`             | The gamma parameter for cosmos.                            |
| `device`            | The device to run the program.                             |
| `init_seed`         | Random seed.                                               |
