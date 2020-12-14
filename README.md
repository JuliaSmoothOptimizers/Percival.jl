# Percival.jl - An augmented Lagrangian solver

[![CI](https://github.com/JuliaSmoothOptimizers/Percival.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaSmoothOptimizers/Percival.jl/actions)
[![codecov.io](https://codecov.io/github/JuliaSmoothOptimizers/Percival.jl/coverage.svg?branch=master)](https://codecov.io/github/JuliaSmoothOptimizers/Percival.jl?branch=master)
[![docs-latest](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/Percival.jl/latest)
[![docs-dev](https://img.shields.io/badge/docs-dev-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/Percival.jl/dev)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3969045-blue.svg?style=flat)](https://doi.org/10.5281/zenodo.3969045)

Percival is an implementation of the augmented Lagrangian solver described in

    S. Arreckx, A. Lambe, Martins, J. R. R. A., & Orban, D. (2016).
    A Matrix-Free Augmented Lagrangian Algorithm with Application to Large-Scale Structural Design Optimization.
    Optimization And Engineering, 17, 359–384. doi:10.1007/s11081-015-9287-9

with internal solver `tron` from [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl).
To use Percival, you have to pass it an [NLPModel](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

## How to Cite

If you use Percival.jl in your work, please cite using the format given in [CITATION.bib](CITATION.bib).

## Install

Use `]` to enter `pkg>` mode of Julia, then
```julia
pkg> add https://github.com/JuliaSmoothOptimizers/Percival.jl
```
## Use with JuMP

You can solve an JuMP model `m` by using NLPModels to convert it.
```
using NLPModelsJuMP, Percival
nlp = MathOptNLPModel(m)
output = percival(nlp)
```
