using Pkg
path = dirname(@__FILE__)
Pkg.activate(path)
using CUTEst
using NLPModels, NLPModelsIpopt, Percival, SolverBenchmark

nmax = 10
problems = readlines("list_problems_$nmax.dat")
cutest_problems = (CUTEstModel(p) for p in problems)

max_time = 300.0 #20 minutes
tol = 1e-5

# Percival's parameters
# My logic is:
# η₀ = ω₀ = 1/μ
# we solve chain |c(x0)|=3.7 with mu=1e3 (peu importe inity)
# catmix: |c(x0)| = 0 ???
# channel: |c(x0)| = 0 ???
# gasoil: |c(x0)| = 1.3 (peut-être avec plus de temps - plutôt inity=false)
# marine: |c(x0)| = 1.5e4 (badly scaled)
params = Dict(
:μ => 1000.0, # default: 10 // 1e3 looks interesting (with :η₀ => 1e-3 and :ω₀ => 1e-3)
:μ_up => 10.0, # default: 10
:inity => false, # default: false
:η₀ => 0.001, # default: 0.5 Starting value for the contraints tolerance of the subproblem
:ω₀ => 0.001,  # default: 1 Starting value for relative tolerance of the subproblem;
:ω_min => sqrt(eps()), #default: atol
:α₁ => 0.9,  # default: 0.9 ``η = max(η / al_nlp.μ^α₁, ϵp)`` if ``‖c(xᵏ)‖ ≤ η``;
:β₀ => 0.1, # default: 1
:β₁ => 0.1, # default: 0.1 ``η = max(1 / al_nlp.μ^β₁, ϵp)`` if ``‖c(xᵏ)‖ > η``;
:subsolver_max_iter => typemax(Int),
# TRON args
#max_radius = # default:
#acceptance_threshold = # default:
#decrease_threshold = # default:
:increase_threshold => 0.75, # default: 0.75
#large_decrease_factor = # default:
#small_decrease_factor = # default:
#increase_factor = # default:
# MORE TRON ARGS
:μ₀ => 1e-2, #μ₀::T = T(1e-2): algorithm parameter in (0, 0.5).
:μ₁ => 1.0, #μ₁::T = one(T): algorithm parameter in (0, +∞).
:σ => 10.0, #σ::T = T(10)`: algorithm parameter in (1, +∞).
:verbose => 0,
:subsolver_verbose => 0,
)

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
      atol = tol, # = 0
      ctol = tol,
      rtol = tol,
      μ = params[:μ],
      μ_up = params[:μ_up],
      η₀ = params[:η₀],
      ω₀ = params[:ω₀],
      ω_min = params[:ω_min],
      α₁ = params[:α₁],
      β₀ = params[:β₀],
      β₁ = params[:β₁],
      μ₀ = params[:μ₀],
      μ₁ = params[:μ₁],
      σ  = params[:σ],
      increase_threshold = params[:increase_threshold],
      inity = params[:inity],
      subsolver_max_cgiter = 2 * nlp.meta.nvar,
      subsolver_max_iter = params[:subsolver_max_iter],
      subsolver_verbose = params[:subsolver_verbose],
      verbose = params[:verbose],
  ),
)

stats = bmark_solvers(solvers, cutest_problems, skipif = nlp -> !nlp.meta.minimize)

using Dates, JLD2
name = "$(today())_CUTEst_$(nmax)_ipopt_percival"
@save "$name.jld2" stats max_time tol params

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
