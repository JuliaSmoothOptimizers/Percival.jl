using Pkg
Pkg.activate(".")
using Percival

using NLPModels, NLPModelsModifiers
using ADNLPModels, OptimizationProblems
using NLPModelsIpopt

# TODO: Update values to have the COPS problems
COPS_INSTANCES = [
    # (OptimizationProblems.ADNLPProblems.bearing, (50, 50), -1.5482e-1, "bearing"),
    (OptimizationProblems.ADNLPProblems.chain, (800,), 5.06891, "chain"),
    (OptimizationProblems.ADNLPProblems.camshape, (1000,), 4.2791, "camshape"), # TODO: result is slightly different
    #(OptimizationProblems.ADNLPProblems.catmix, (100,), -4.80556e-2, "catmix"),
    (OptimizationProblems.ADNLPProblems.channel, (200,), 1.0, "channel"),
    (OptimizationProblems.ADNLPProblems.elec, (50,), 1.0552e3, "elec"),
    #(OptimizationProblems.ADNLPProblems.gasoil, (100,), 5.2366e-3, "gasoil"),
    #(OptimizationProblems.ADNLPProblems.glider, (100,), 1.25505e3, "glider"),
    (OptimizationProblems.ADNLPProblems.marine, (100,), 1.97462e7, "marine"),
    #(OptimizationProblems.ADNLPProblems.methanol, (100,), 9.02229e-3, "methanol"),
    #(OptimizationProblems.ADNLPProblems.minsurf, (50, 50), 2.51488, "minsurf50"),
    #(OptimizationProblems.ADNLPProblems.minsurf, (50, 75), 2.50568, "minsurf75"),
    #(OptimizationProblems.ADNLPProblems.minsurf, (50, 100), 2.50694, "minsurf100"),
    #(OptimizationProblems.ADNLPProblems.pinene, (100,), 1.98721e1, "pinene"),
    (OptimizationProblems.ADNLPProblems.polygon, (100,), -0.674981, "polygon"), # N.B: objective depends on the optimizer used.
    (OptimizationProblems.ADNLPProblems.robotarm, (200,), 9.14138, "robot"),
    #(OptimizationProblems.ADNLPProblems.rocket, (400,), 1.01283, "rocket"),
    #(OptimizationProblems.ADNLPProblems.steering, (200,), 5.54577e-1, "steering"),
    (OptimizationProblems.ADNLPProblems.tetra_duct15, (), 1.04951e4, "tetra_duct15"),
    (OptimizationProblems.ADNLPProblems.tetra_duct20, (), 4.82685e3, "tetra_duct20"),
    (OptimizationProblems.ADNLPProblems.tetra_foam5, (), 6.42560e3, "tetra_foam5"),
    (OptimizationProblems.ADNLPProblems.tetra_gear, (), 4.15163e3, "tetra_gear"),
    (OptimizationProblems.ADNLPProblems.tetra_hook, (), 6.05735e3, "tetra_hook"),
    #(OptimizationProblems.ADNLPProblems.torsion, (50, 50), -4.18087e-1, "torsion"),
    #(OptimizationProblems.ADNLPProblems.dirichlet, (20,), 1.71464e-2, "dirichlet"),
    #(OptimizationProblems.ADNLPProblems.henon, (10,), 6.667736, "henon"), # N.B: objective depends on the optimizer used.
    #(OptimizationProblems.ADNLPProblems.lane_emden, (20,), 9.11000, "lane_emden"),
    (OptimizationProblems.ADNLPProblems.triangle_deer, (), 2.01174e3, "triangle_deer"),
    (OptimizationProblems.ADNLPProblems.triangle_pacman, (), 1.25045e3, "triangle_pacman"),
    (OptimizationProblems.ADNLPProblems.triangle_turtle, (), 4.21523e3, "triangle_turtle"),
]
tol = 1e-3
# remove Hvprod backend
cops_problems = (instance(n = 100, name = name, matrix_free = true) for (instance, params, result, name) in COPS_INSTANCES)
#=
@time nlp = first(cops_problems)
@time grad(nlp, nlp.meta.x0)

reset!(nlp)
stats = percival(nlp, rtol = tol, ctol = tol, verbose = 1)

reset!(nlp)
subproblem_modifier = nlp -> NLPModelsModifiers.LBFGSModel(nlp) # default mem
stats_sub = percival(nlp, subproblem_modifier = subproblem_modifier, rtol = tol, ctol = tol, verbose = 1, max_eval = 400000)

reset!(nlp)
qn_nlp = NLPModelsModifiers.LBFGSModel(nlp)
stats_qn = percival(qn_nlp, rtol = tol, ctol = tol, verbose = 1, max_time = 60.0)
=#

max_time = 1200.0 #20 minutes
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
  :percival10 => nlp -> percival(
      nlp,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
      atol = eps(T),
      ctol = tol,
      rtol = tol,
      μ = μ,
  ),
  :percival10_lbfgs_subproblem => nlp -> percival(
      nlp,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
      atol = eps(T),
      ctol = tol,
      rtol = tol,
      μ = 10.0,
      subproblem_modifier = nlp -> NLPModelsModifiers.LBFGSModel(nlp), # default mem
  )
)

#=
using SolverBenchmark
cops_problems = (instance(n = n, name = name) for (instance, params, result, name) in COPS_INSTANCES)
stats = bmark_solvers(solvers, cops_problems, skipif = nlp -> nlp.meta.ncon == 0)
=#
