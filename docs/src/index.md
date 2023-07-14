 # Percival.jl

_an Augmented Lagrangian method_

```@contents
Pages = ["index.md"]
```

## Description

This package implements a JSO-compliant augmented Lagrangian method based on the paper

    S. Arreckx, A. Lambe, Martins, J. R. R. A., & Orban, D. (2016).
    A Matrix-Free Augmented Lagrangian Algorithm with Application to Large-Scale Structural Design Optimization.
    Optimization And Engineering, 17, 359–384. doi:10.1007/s11081-015-9287-9

It was implemented as part of the Master's dissertation of Egmara Antunes.

### JSO-compliant solver

The `percival` method expects a single mandatory argument - an [AbstractNLPModel](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) - and returns a GenericExecutionStats from [SolverCore.jl](https://github.com/JuliaSmoothOptimizers/SolverCore.jl).

We refer to [jso.dev](https://jso.dev/) for tutorials on the NLPModel API. 
The functions used to access the NLPModel in general, are defined in `NLPModels.jl`. So, for instance, you can access the objective function's documentation as follows
```julia
using NLPModels
? obj
```
or visit directly [NLPModels.jl's documentation](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/api/).
This framework allows the usage of models from Ampl (using [AmplNLReader.jl](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl)), CUTEst (using [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl)), JuMP (using [NLPModelsJuMP.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl)), PDE-constrained optimization problems (using [PDENLPModels.jl](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl)) and models defined with automatic differentiation (using [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl)).

## Installation

`Percival` is a registered package. To install this package, open the Julia REPL (i.e., execute the julia binary), type `]` to enter package mode, and install `Percival` as follows
```
add Percival
```

### Main exported functions and types

- [`percival`](@ref): The function to call the method. Pass an NLPModel to it.
- [`AugLagModel`](@ref): A model representing the bound-constrained augmented Lagrangian subproblem. This is a subtype of [AbstractNLPModel](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

## Example

How to solve the simple problem
```math
\min_{(x_1,x_2)} \quad (x_1 - 1)^2 + 100 (x_2 - x_1^2)^2 \quad \text{s.to} \quad x_1^2 + x_2^2 \leq 1.
```
The problem is modeled using `ADNLPModels.jl` with `[-1.2; 1.0]` as default initial point, and then solved using `percival`.
```@example ex1
using ADNLPModels, Percival

nlp = ADNLPModel(
    x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2; 1.0],
    x -> [x[1]^2 + x[2]^2],
    [-Inf],
    [1.0]
)

output = percival(nlp)
print(output)
```

You can find more tutorials on 
https://jso.dev/tutorials/ and select the tag `Percival.jl`.

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/Percival.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
