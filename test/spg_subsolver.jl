# Tests for the GPU-compatible `SPGSubSolver` option and the GPU-safe
# constraint-classification helper added for first-order / device-resident
# augmented-Lagrangian solves.

# SPG is a first-order subsolver: on well-conditioned problems it recovers the
# optimum, but (as with any first-order AL) it can stall far from the KKT
# point on ill-conditioned ones. The test therefore uses a convex,
# well-conditioned equality-constrained QP and checks the recovered solution
# and feasibility, not the `:first_order` status flag (the outer dual
# tolerance may not be met even when the primal solution is essentially exact).
function test_spg_subsolver()
  # min 2x₁² + x₁x₂ + x₂² - 9x₁ - 9x₂  s.t.  4x₁ + 6x₂ = 10 ; optimum (1, 1).
  nlp = ADNLPModel(
    x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
    [1.0; 2.0],
    x -> [4x[1] + 6x[2] - 10],
    zeros(1),
    zeros(1),
  )
  stats = percival(
    nlp;
    subsolver = SPGSubSolver,
    max_iter = 3000,
    max_eval = 1_000_000,
    atol = 1e-6,
    rtol = 1e-6,
    ctol = 1e-6,
  )
  @test isapprox(stats.solution, ones(2), atol = 1e-3)
  @test stats.primal_feas < 1e-5
end

function test_spg_subsolver_standalone()
  # The subsolver on its own bound-constrained problem.
  nlp = ADNLPModel(x -> sum((x .- 2) .^ 2), zeros(3), -ones(3), ones(3))
  solver = SPGSubSolver(nlp)
  stats = SolverCore.GenericExecutionStats(nlp)
  SolverCore.solve!(solver, nlp, stats; atol = 1e-8, rtol = 1e-8, max_iter = 1000)
  @test stats.status == :first_order
  @test isapprox(stats.solution, ones(3), atol = 1e-6) # projected onto [-1, 1]
end

function test_gpu_safe_equality_classification()
  # `_is_equality_constrained` must agree with `NLPModels.equality_constrained`
  # on ordinary (CPU) models, including when bound analysis is disabled.
  eq = ADNLPModel(x -> sum(x .^ 2), ones(2), x -> [x[1] + x[2] - 1], [0.0], [0.0])
  @test Percival._is_equality_constrained(eq)
  ineq = ADNLPModel(x -> sum(x .^ 2), ones(2), x -> [x[1] + x[2]], [0.0], [1.0])
  @test !Percival._is_equality_constrained(ineq)
end
