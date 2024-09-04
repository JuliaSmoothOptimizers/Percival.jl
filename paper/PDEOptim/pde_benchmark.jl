using Pkg
path = dirname(@__FILE__)
Pkg.activate(path)
Pkg.add(url="https://github.com/tmigot/PDEOptimizationProblems")
Pkg.add(url="https://github.com/JuliaSmoothOptimizers/Percival.jl", rev="add-limits")
using PDEOptimizationProblems
Pkg.instantiate()
using NLPModels, NLPModelsIpopt, Percival, SolverBenchmark

n = 5
pde_problems = (PDEOptimizationProblems.eval(p)(n = 5) for p in setdiff(names(PDEOptimizationProblems), [:PDEOptimizationProblems]))

max_time = 120.0 #2 minutes
tol = 1e-3
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
stats = bmark_solvers(solvers, pde_problems, skipif = nlp -> unconstrained(nlp))

using Dates, JLD2
name = "$$(today())_PDEOP_$(Int(-log10(tol)))_ipopt_percival_n_$(n)_mu_$(μ)"
@save "$name.jld2" stats max_time tol μ n

#
#PREVIEW
#

using Plots

solved(df) = (df.status .== :first_order)
costs = [
  df -> .!solved(df) * Inf + df.elapsed_time,
  df -> .!solved(df) * Inf + df.neval_obj + df.neval_cons,
]
costnames = ["Time", "Evaluations of obj + cons"]
p = profile_solvers(stats, costs, costnames, width=400, height=400)
#plot!(xticks = ([0.0, 5.0, 10.0, 15.0], ["2⁰", "2⁵", "2¹⁰", "2¹⁵"]))
plot!(xtickfontsize=9)
png(p, name)

io = open("$name.dat", "w")
hdr_override = Dict(:name => "Name", :f => "f(x)", :t => "Time")
df = join(stats, [:status, :objective, :elapsed_time, :dual_feas, :primal_feas], invariant_cols=[:name, :nvar, :ncon], hdr_override=hdr_override)
pretty_stats(io, df)
close(io)
