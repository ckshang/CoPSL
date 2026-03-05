# CoPSL

<div align="center">

<!-- Center-No-Ver-Bar-H1-H2 for GitHub, fork from https://gist.github.com/CodeByAidan/bb43bdb1c07c0933d8b67c23515fb912 -->
<div id="toc">
	<ul align="center" style="list-style: none; padding:0; margin:0;">
		<summary style="margin:0;">
			<h2 style="margin:5px 0;">Collaborative Pareto Set Learning in Multiple Multi-Objective Optimization Problems</h2>
			<h3 style="margin:5px 0;">IJCNN 2024 Oral</h3>
		</summary>
	</ul>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2503.13227-b31b1b.svg)](https://arxiv.org/abs/2404.01224)

![image](https://github.com/ckshang/CoPSL/blob/main/figs/CoPSL.png)

</div>

The code is mainly designed to be simple and readable, it contains:

- `run_copsl.py` is a ~250-line script to run the CoPSL algorithms (including Pareto front visualization);
- `run_copsl_gn.py` is a ~210-line script to run the CoPSL algorithms with the GradNorm strategy;
- `run_psl.py` is a ~240-line script to run the PSL algorithms (including Pareto front visualization);
- `run_emo.py` is a ~100-line script to run the EMO algorithms;
- `problem.py` includes all the test problems utilized in the paper for running PSL algorithms;
- `problem_emo.py` also includes all the test problems utilized in the paper for running EMO algorithms;
- `model.py` contains both single-task and multi-task architecture models;
- `utils.py` contains several reusable utility functions;
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
| `gamma`             | The $\gamma$ parameter for cosmos.                         |
| `device`            | The device to run the program.                             |
| `init_seed`         | Random seed.                                               |

### Citation

If you find our work helpful to your research, please cite our paper:
```
@inproceedings{shang2024collaborative,
  title={Collaborative Pareto set learning in multiple multi-objective optimization problems},
  author={Shang, Chikai and Ye, Rongguang and Jiang, Jiaqi and Gu, Fangqing},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2024},
  organization={IEEE}
}
```
