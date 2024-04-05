using Pkg; Pkg.activate("")
using Symbolics, ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems
using NLPModels, NLPModelsIpopt, Percival, SolverBenchmark

n = 1000 # targeted number of variables

# TODO: Update values to have the COPS problems
COPS_INSTANCES = [
    (OptimizationProblems.ADNLPProblems.bearing, (50, 50), -1.5482e-1, "bearing"),
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

cops_problems = (instance(n = n, name = name) for (instance, params, result, name) in COPS_INSTANCES)

max_time = 1200.0 #20 minutes
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
      atol = 0.0,
      ctol = tol,
      rtol = tol,
      μ = μ,
  ),
)

stats = bmark_solvers(solvers, cops_problems, skipif = nlp -> nlp.meta.ncon == 0)

using Dates, JLD2
@save "$(today())_AD_n_$(n)_COPS3_ipopt_percival.jld2" stats max_time tol μ n
