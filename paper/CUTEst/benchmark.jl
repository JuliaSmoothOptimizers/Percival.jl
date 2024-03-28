using Pkg; Pkg.activate("")
using CUTEst, NLPModels, NLPModelsIpopt, DCISolver, SolverBenchmark

problems = readlines("list_problems.dat")
cutest_problems = (CUTEstModel(p) for p in problems)

max_time = 1200.0 #20 minutes
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
      tol = tol,
  ),
  :dcildl => nlp -> dci(
      nlp,
      linear_solver = :ldlfact,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
      atol = tol,
      ctol = tol,
      rtol = tol,
  ),
)
stats = bmark_solvers(solvers, cutest_problems)

using JLD2
@save "ipopt_dcildl_$(string(length(problems))).jld2" stats max_time tol 
