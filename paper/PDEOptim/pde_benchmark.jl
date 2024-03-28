using Pkg; Pkg.activate("")
Pkg.add(url="https://github.com/tmigot/PDEOptimizationProblems")
using PDEOptimizationProblems
Pkg.instantiate()
using NLPModels, NLPModelsIpopt, Percival, SolverBenchmark

n = 5
pde_problems = (PDEOptimizationProblems.eval(p)(n = 5) for p in setdiff(names(PDEOptimizationProblems), [:PDEOptimizationProblems]))

max_time = 120.0 #2 minutes
tol = 1e-5
μ = 10.0

solvers = Dict(
  :ipopt => nlp -> ipopt(
      nlp,
      print_level = 0,
      dual_inf_tol = Inf,
      constr_viol_tol = Inf,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      max_cpu_time = max_time,
      tol = tol,
  ),
  :percival => nlp -> percival(
      nlp,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
      atol = tol,
      ctol = tol,
      rtol = tol,
      μ = μ,
  ),
)
stats = bmark_solvers(solvers, pde_problems)

using Dates, JLD2
@save "$(today())_PDEOP_ipopt_percival_n_$(n)_mu_$(μ).jld2" stats max_time tol μ n
