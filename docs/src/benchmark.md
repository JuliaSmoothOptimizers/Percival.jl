# Benchmarks

## CUTEst benchmark

With a JSO-compliant solver, such as Percival, we can run the solver on a set of problems, explore the results, and compare to other JSO-compliant solvers using specialized benchmark tools. 
We are following here the tutorial in [SolverBenchmark.jl](https://juliasmoothoptimizers.github.io/SolverBenchmark.jl/v0.3/tutorial/) to run benchmarks on JSO-compliant solvers.
``` @example ex1
using CUTEst
```

To test the implementation of Percival, we use the package [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl), which implements `CUTEstModel` an instance of `AbstractNLPModel`. 

``` @example ex1
using SolverBenchmark
```

Let us select problems from CUTEst with a maximum of 100 variables or constraints. After removing problems with fixed variables, examples with a constant objective, and infeasibility residuals.

``` @example ex1
_pnames = CUTEst.select(
  max_var = 100, 
  min_con = 1, 
  max_con = 100, 
  only_free_var = true,
  objtype = 3:6
)

#Remove all the problems ending by NE as Ipopt cannot handle them.
pnamesNE = _pnames[findall(x->occursin(r"NE\b", x), _pnames)]
pnames = setdiff(_pnames, pnamesNE)
cutest_problems = (CUTEstModel(p) for p in pnames)

length(cutest_problems) # number of problems
```

We compare here Percival with [Ipopt](https://link.springer.com/article/10.1007/s10107-004-0559-y) (Wächter, A., & Biegler, L. T. (2006). On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Mathematical programming, 106(1), 25-57.), via the [NLPModelsIpopt.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl) thin wrapper, with Percival on a subset of CUTEst problems.

``` @example ex1
using Percival, NLPModelsIpopt
```
 To make stopping conditions comparable, we set `Ipopt`'s parameters `dual_inf_tol=Inf`, `constr_viol_tol=Inf` and `compl_inf_tol=Inf` to disable additional stopping conditions related to those tolerances, `acceptable_iter=0` to disable the search for an acceptable point.

``` @example ex1
#Same time limit for all the solvers
max_time = 1200. #20 minutes
tol = 1e-5

solvers = Dict(
  :ipopt => nlp -> ipopt(
    nlp,
    print_level = 0,
    dual_inf_tol = Inf,
    constr_viol_tol = Inf,
    compl_inf_tol = Inf,
    acceptable_iter = 0,
    max_cpu_time = max_time,
    x0 = nlp.meta.x0,
    tol = tol,
  ),
  :percival => nlp -> percival(
      nlp,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
      atol = tol,
      rtol = tol,
      ctol = tol,
  ),
)

stats = bmark_solvers(solvers, cutest_problems)
```
The function `bmark_solvers` return a `Dict` of `DataFrames` with detailed information on the execution. This output can be saved in a data file.
``` @example ex1
using JLD2
@save "ipopt_percival_$(string(length(pnames))).jld2" stats
```
The result of the benchmark can be explored via tables,
``` @example ex1
pretty_stats(stats[:percival])
```
or it can also be used to make performance profiles.
``` @example ex1
using Plots
gr()

solved(df) = (df.status .== :first_order)
costs = [
  df -> .!solved(df) * Inf + df.elapsed_time,
  df -> .!solved(df) * Inf + df.neval_obj + df.neval_cons,
]
costnames = ["Time", "Evalutions of obj + cons"]
p = profile_solvers(stats, costs, costnames)
```
