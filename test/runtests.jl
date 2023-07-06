using Percival

using ADNLPModels, JSOSolvers, LinearAlgebra, Logging, SolverTest, SparseArrays, Test

using NLPModels, SolverCore, NLPModelsModifiers, NLPModelsTest

mutable struct DummyModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
end

include("allocs.jl")

function test()
  nlp = DummyModel(NLPModelMeta(1, minimize = false))
  @test_throws ErrorException("Percival only works for minimization problem") percival(
    Val(:equ),
    nlp,
  )

  lbfgs_mem = 4
  @testset "Unconstrained tests" begin
    unconstrained_nlp(percival)
  end
  @testset "Bound-constrained tests" begin
    bound_constrained_nlp(percival)
  end

  @testset "Small equality constrained problems $name" for (x0, m, f, c, sol, name) in [
    (
      [1.0; 2.0],
      1,
      x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
      x -> [4x[1] + 6x[2] - 10],
      ones(2),
      "1",
    ),
    ([-1.2; 1.0], 1, x -> (x[1] - 1)^2, x -> [10 * (x[2] - x[1]^2)], ones(2), "2"),
    (
      [-1.2; 1.0],
      1,
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      x -> [(x[1] - 2)^2 + (x[2] - 2)^2 - 2],
      ones(2),
      "3",
    ),
    ([2.0; 1.0], 2, x -> -x[1], x -> [x[1]^2 + x[2]^2 - 25; x[1] * x[2] - 12], [4.0; 3.0], "4"),
  ]
    nlp = ADNLPModel(f, x0, c, zeros(m), zeros(m))
    output = with_logger(NullLogger()) do
      percival(nlp)
    end

    @test isapprox(output.solution, sol, rtol = 1e-6)
    @test output.primal_feas < 1e-6
    @test output.dual_feas < 1e-6
    @test output.status == :first_order

    # LBFGS approximation of the augmented Lagrangian
    output = with_logger(NullLogger()) do
      subproblem_modifier = m -> NLPModelsModifiers.LBFGSModel(m, mem = lbfgs_mem)
      percival(nlp, subproblem_modifier = subproblem_modifier, rtol = 1e-4)
    end

    @test isapprox(output.solution, sol, rtol = 1e-3)
    @test output.primal_feas < 1e-3
    @test output.dual_feas < 1e-3
    @test output.status == :first_order
  end

  @testset "Small equality bound-constrained problems $name" for (
    x0,
    m,
    f,
    lvar,
    uvar,
    c,
    sol,
    name,
  ) in [
    (
      [1.0; 2.0],
      1,
      x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
      zeros(2),
      2 * ones(2),
      x -> [4x[1] + 6x[2] - 10],
      ones(2),
      "1",
    ),
    (
      [1.0; 2.0],
      1,
      x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
      zeros(2),
      [Inf; 0.5],
      x -> [4x[1] + 6x[2] - 10],
      [1.75; 0.5],
      "2",
    ),
    (
      [-1.2; 1.0],
      1,
      x -> (x[1] - 1)^2,
      [-Inf; 0.0],
      [0.9, Inf],
      x -> [10 * (x[2] - x[1]^2)],
      [0.9; 0.81],
      "3",
    ),
    (
      [-1.2; 1.0],
      1,
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      zeros(2),
      ones(2),
      x -> [(x[1] - 2)^2 + (x[2] - 2)^2 - 2],
      ones(2),
      "4",
    ),
    (
      [2.0; 1.0],
      2,
      x -> -x[1],
      [2.5; 3.5],
      [3.5; 4.5],
      x -> [x[1]^2 + x[2]^2 - 25; x[1] * x[2] - 12],
      [3.0; 4.0],
      "5",
    ),
  ]
    nlp = ADNLPModel(f, x0, lvar, uvar, c, zeros(m), zeros(m))
    output = with_logger(NullLogger()) do
      percival(nlp)
    end

    @test isapprox(output.solution, sol, rtol = 1e-6)
    @test output.primal_feas < 1e-6
    @test output.dual_feas < 1e-6
    @test output.status == :first_order

    # LBFGS approximation of the augmented Lagrangian
    output = with_logger(NullLogger()) do
      subproblem_modifier = m -> NLPModelsModifiers.LBFGSModel(m, mem = lbfgs_mem)
      percival(nlp, subproblem_modifier = subproblem_modifier, rtol = 1e-5)
    end

    @test isapprox(output.solution, sol, rtol = 1e-4)
    @test output.primal_feas < 1e-4
    @test output.dual_feas < 1e-4
    @test output.status == :first_order
  end

  @testset "Small inequality constrained problems $name" for (x0, m, f, c, lcon, ucon, sol, name) in
                                                             [
    (
      [1.0; 2.0],
      1,
      x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
      x -> [4x[1] + 6x[2] - 10],
      [-Inf],
      [0.0],
      ones(2),
      "1",
    ),
    (
      [-1.2; 1.0],
      1,
      x -> (x[1] - 1)^2 + 0.01 * x[2],
      x -> [10 * (x[2] - x[1]^2)],
      [0.0],
      [Inf],
      [0.990099; 0.980296],
      "2",
    ),
    (
      [-1.2; 1.0],
      1,
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      x -> [x[1]^2 + x[2]^2],
      [0.0],
      [4.0],
      ones(2),
      "3",
    ),
    (
      [-1.2; 1.0],
      1,
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      x -> [(x[1] - 2)^2 + (x[2] - 2)^2],
      [2.0],
      [4.0],
      ones(2),
      "4",
    ),
    (
      [2.0; 1.0],
      2,
      x -> -x[1],
      x -> [x[1]^2 + x[2]^2; x[1] * x[2]],
      [25.0; 9.0],
      [30.0; 12.0],
      [5.196152; 1.732051],
      "5",
    ),
  ]
    nlp = ADNLPModel(f, x0, c, lcon, ucon)
    output = with_logger(NullLogger()) do
      percival(nlp)
    end

    @test isapprox(output.solution, sol, rtol = 1e-6)
    @test output.primal_feas < 1e-6
    @test output.dual_feas < 1e-6
    @test output.status == :first_order

    # LBFGS approximation of the augmented Lagrangian
    output = with_logger(NullLogger()) do
      subproblem_modifier = m -> NLPModelsModifiers.LBFGSModel(m, mem = lbfgs_mem)
      percival(nlp, subproblem_modifier = subproblem_modifier, rtol = 1e-5)
    end

    @test isapprox(output.solution, sol, rtol = 1e-4)
    @test output.primal_feas < 1e-4
    @test output.dual_feas < 1e-4
    @test output.status == :first_order
  end
end

test()

include("restart.jl")

include("callback.jl")

@testset "Change TRON solver parameters" begin
  @testset "Test max_radius in TRON" begin
    max_radius = 0.00314
    increase_factor = 5.0
    function cb(nlp, solver, stats)
      @test solver.sub_solver.tr.radius ≤ max_radius
    end

    nlp = ADNLPModel(
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      [-1.2; 1.0],
      x -> [x[1]^2 + x[2]^2],
      [-Inf],
      [1.0],
    )
    x = nlp.meta.x0
    stats = percival(
      nlp,
      x = x,
      callback = cb,
      max_radius = max_radius,
      increase_factor = increase_factor,
    )

    nls = ADNLSModel(
      x -> [100 * (x[2] - x[1]^2); x[1] - 1],
      [-1.2; 1.0],
      2,
      x -> [x[1]^2 + x[2]^2],
      [-Inf],
      [1.0],
    )
    x = nls.meta.x0
    stats = percival(
      nls,
      x = x,
      callback = cb,
      max_radius = max_radius,
      increase_factor = increase_factor,
    )
  end
end
