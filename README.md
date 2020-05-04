# Percival.jl - An augmented Lagrangian solver

[![Build Status](https://travis-ci.org/JuliaSmoothOptimizers/Percival.jl.svg?branch=master)](https://travis-ci.org/JuliaSmoothOptimizers/Percival.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/kp1o6ejuu6kgskvp/branch/master?svg=true)](https://ci.appveyor.com/project/dpo/percival-jl/branch/master)
[![Build Status](https://api.cirrus-ci.com/github/JuliaSmoothOptimizers/Percival.jl.svg)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/Percival.jl)
[![Coveralls](https://coveralls.io/repos/JuliaSmoothOptimizers/Percival.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaSmoothOptimizers/Percival.jl?branch=master)
[![docs](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/Percival.jl/latest)

Percival is an implementation of the augmented Lagrangian solver described in

    S. Arreckx, A. Lambe, Martins, J. R. R. A., & Orban, D. (2016).
    A Matrix-Free Augmented Lagrangian Algorithm with Application to Large-Scale Structural Design Optimization.
    Optimization And Engineering, 17, 359–384. doi:10.1007/s11081-015-9287-9

with internal solver `tron` from [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl).
To use Percival, you have to pass it an [NLPModel](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

## Install

Use `]` to enter `pkg>` mode of Julia, then
```julia
pkg> add https://github.com/JuliaSmoothOptimizers/Percival.jl
```
