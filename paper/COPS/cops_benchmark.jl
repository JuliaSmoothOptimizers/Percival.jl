using Pkg; Pkg.activate("")
Pkg.add(url="https://github.com/MadNLP/COPSBenchmark.jl")
using COPSBenchmark, JuMP
using NLPModels, NLPModelsJuMP, NLPModelsIpopt, Percival, SolverBenchmark

COPS_INSTANCES = [
    (COPSBenchmark.bearing_model, (50, 50), -1.5482e-1, "bearing"),
    (COPSBenchmark.chain_model, (800,), 5.06891, "chain"),
    (COPSBenchmark.camshape_model, (1000,), 4.2791, "camshape"), # TODO: result is slightly different
    (COPSBenchmark.catmix_model, (100,), -4.80556e-2, "catmix"),
    (COPSBenchmark.channel_model, (200,), 1.0, "channel"),
    (COPSBenchmark.elec_model, (50,), 1.0552e3, "elec"),
    (COPSBenchmark.gasoil_model, (100,), 5.2366e-3, "gasoil"),
    (COPSBenchmark.glider_model, (100,), 1.25505e3, "glider"),
    (COPSBenchmark.marine_model, (100,), 1.97462e7, "marine"),
    (COPSBenchmark.methanol_model, (100,), 9.02229e-3, "methanol"),
    (COPSBenchmark.minsurf_model, (50, 50), 2.51488, "minsurf50"),
    (COPSBenchmark.minsurf_model, (50, 75), 2.50568, "minsurf75"),
    (COPSBenchmark.minsurf_model, (50, 100), 2.50694, "minsurf100"),
    (COPSBenchmark.pinene_model, (100,), 1.98721e1, "pinene"),
    (COPSBenchmark.polygon_model, (100,), -0.674981, "polygon"), # N.B: objective depends on the optimizer used.
    (COPSBenchmark.robot_model, (200,), 9.14138, "robot"),
    (COPSBenchmark.rocket_model, (400,), 1.01283, "rocket"),
    (COPSBenchmark.steering_model, (200,), 5.54577e-1, "steering"),
    (COPSBenchmark.tetra_duct15_model, (), 1.04951e4, "tetra_duct15"),
    (COPSBenchmark.tetra_duct20_model, (), 4.82685e3, "tetra_duct20"),
    (COPSBenchmark.tetra_foam5_model, (), 6.42560e3, "tetra_foam5"),
    (COPSBenchmark.tetra_gear_model, (), 4.15163e3, "tetra_gear"),
    (COPSBenchmark.tetra_hook_model, (), 6.05735e3, "tetra_hook"),
    (COPSBenchmark.torsion_model, (50, 50), -4.18087e-1, "torsion"),
    (COPSBenchmark.dirichlet_model, (20,), 1.71464e-2, "dirichlet"),
    (COPSBenchmark.henon_model, (10,), 6.667736, "henon"), # N.B: objective depends on the optimizer used.
    (COPSBenchmark.lane_emden_model, (20,), 9.11000, "lane_emden"),
    (COPSBenchmark.triangle_deer_model, (), 2.01174e3, "triangle_deer"),
    (COPSBenchmark.triangle_pacman_model, (), 1.25045e3, "triangle_pacman"),
    (COPSBenchmark.triangle_turtle_model, (), 4.21523e3, "triangle_turtle"),
]

cops_problems = [MathOptNLPModel(instance(params...), name = name) for (instance, params, result, name) in COPS_INSTANCES]

max_time = 120.0 #2 minutes
tol = 1e-5

# Percival's parameters
μ = 10.0 # default: 10
μ_up = 10 # default: 10
inity = true # default: false
η₀ = 1 // 2 # default: 0.5 Starting value for the contraints tolerance of the subproblem
ω₀ = 1  # default: 1 Starting value for relative tolerance of the subproblem;
α₁ = 9 // 10  # default: 0.9 ``η = max(1 / al_nlp.μ^α₁, ϵp)`` if ``‖c(xᵏ)‖ ≤ η``;
β₁ = 1 // 10 # default: 0.1 ``η = max(1 / al_nlp.μ^β₁, ϵp)`` if ``‖c(xᵏ)‖ > η``;

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
      inity = inity,
      subsolver_verbose = 1,
      verbose = 1,
  ),
)

stats = bmark_solvers(solvers, cops_problems)

using Dates, JLD2
@save "$(today())_COPS3_ipopt_percival.jld2" stats max_time tol μ
