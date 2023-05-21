# DP-Fast MH: Private, Fast, and Accurate Metropolis-Hastings for Large-Scale Bayesian Inference

This repository contains code for the paper
[DP-Fast MH: Private, Fast, and Accurate Metropolis-Hastings for Large-Scale Bayesian Inference](https://arxiv.org/pdf/2303.06171.pdf), accepted in _International Conference on Machine Learning (ICML), 2023_.

```bibtex
@article{zhang2023dpfastmh,
  title={DP-Fast MH: Private, Fast, and Accurate Metropolis-Hastings for Large-Scale Bayesian Inference},
  author={Zhang, Wanrong and Zhang, Ruqi},
  journal={International Conference on Machine Learning},
  year={2023}
}
```

# Usage
## Gaussian Mixture
Please run
```
cd mog
julia dpfastmh.jl
```
Or
```
julia dpfastmh_full.jl
```
## Logistic Regression on MNIST 
Please run
```
cd logistic
julia dpfastmh.jl
```
Or
```
julia dpfastmh_full.jl
```
