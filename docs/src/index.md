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

### JSO-compliant

The `percival` method expects a single mandatory argument - an [NLPModel](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) - and returns a GenericExecutionStats from [SolverTools.jl](https://github.com/JuliaSmoothOptimizers/SolverTools.jl).

### Main exported functions and types

- [`percival`](@ref): The function to call the method. Pass an NLPModel to it.
- [`AugLagModel`](@ref): A model representing the augmented Lagrangian subproblem, that allows better use of memory.

## Example

How to solve the simple problem
```math
\min \ (x_1 - 1)^2 + 100 (x_2 - x_1^2)^2 \quad \text{s.to} \quad x_1^2 + x_2^2 \leq 1.
```

```@example
using NLPModels, Percival

nlp = ADNLPModel(
    x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2; 1.0],
    x -> [x[1]^2 + x[2]^2],
    [-Inf],
    [1.0]
)

output = percival(nlp)
println(output)
```