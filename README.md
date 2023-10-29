# Percival.jl - An augmented Lagrangian solver

[![CI](https://github.com/JuliaSmoothOptimizers/Percival.jl/workflows/CI/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/Percival.jl/actions)
[![codecov.io](https://codecov.io/github/JuliaSmoothOptimizers/Percival.jl/coverage.svg?branch=main)](https://codecov.io/github/JuliaSmoothOptimizers/Percival.jl?branch=main)
[![docs-stable](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/Percival.jl/stable)
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
pkg> add Percival
```

## Examples

Consider the following 2-dimensional optimization problem with an equality constraint

$$
\begin{equation}
\min_{(x_1,x_2)} \quad (x_1 - 1)^2 + 100 (x_2 - x_1^2)^2 \quad \text{s.to} \quad x_1^2 + x_2^2 = 1.
\end{equation}
$$

You can solve an JuMP model `model` by using [NLPModelsJuMP.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl) to convert it.
```julia
using JuMP, NLPModelsJuMP, Percival
model = Model(NLPModelsJuMP.Optimizer)
set_attribute(model, "solver", Percival.PercivalSolver)
@variable(model, x[i=1:2], start = [-1.2; 1.0][i])
@objective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)
@constraint(model, x[1]^2 + x[2]^2 == 1)
optimize!(model)
solution_summary(model)
```

`percival` accept as input any instance of `AbstractNLPModel`, for instance, using automatic differentiation via [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) to solve the same problem.
```julia
using ADNLPModels, Percival
nlp = ADNLPModel(
  x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
  [-1.2; 1.0],
  x -> [x[1]^2 + x[2]^2],
  [1.0],
  [1.0],
)
output = percival(nlp, verbose = 1)
```

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/Percival.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
