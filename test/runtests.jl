using Percival

using JSOSolvers, LinearAlgebra, Logging, SparseArrays, Test

const jsosolvers_folder = joinpath(dirname(pathof(JSOSolvers)), "..", "test", "solvers")
include(joinpath(jsosolvers_folder, "unconstrained.jl"))
include(joinpath(jsosolvers_folder, "bound-constrained.jl"))

using NLPModels

function test()
  # unconstrained tests from JSOSolvers
  test_unconstrained_solver(percival)

  # Bound constrained tests from JSOSolvers
  test_bound_constrained_solver(percival)

  @testset "Small equality constrained problems" begin
    for (x0, m, f, c, sol) in [([1.0; 2.0], 1,
                                x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
                                x -> [4x[1] + 6x[2] - 10],
                                ones(2)
                               ),
                               ([-1.2; 1.0], 1,
                                x -> (x[1] - 1)^2,
                                x -> [10 * (x[2] - x[1]^2)],
                                ones(2)
                               ),
                               ([-1.2; 1.0], 1,
                                x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
                                x -> [(x[1] - 2)^2 + (x[2] - 2)^2 - 2],
                                ones(2)
                               ),
                               ([2.0; 1.0], 2,
                                x -> -x[1],
                                x -> [x[1]^2 + x[2]^2 - 25; x[1] * x[2] - 12],
                                [4.0; 3.0]
                               )
                              ]
      nlp = ADNLPModel(f, x0, c=c, lcon=zeros(m), ucon=zeros(m))
      output = with_logger(NullLogger()) do
        percival(nlp)
      end

      @test isapprox(output.solution, sol, rtol=1e-6)
      @test output.primal_feas < 1e-6
      @test output.dual_feas < 1e-6
      @test output.status == :first_order
    end
  end

  @testset "Small equality bound-constrained problems" begin
    for (x0, m, f, lvar, uvar, c, sol) in [([1.0; 2.0], 1,
                                            x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
                                            zeros(2), 2 * ones(2),
                                            x -> [4x[1] + 6x[2] - 10],
                                            ones(2)
                                           ),
                                           ([1.0; 2.0], 1,
                                            x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
                                            zeros(2), [Inf; 0.5],
                                            x -> [4x[1] + 6x[2] - 10],
                                            [1.75; 0.5]
                                           ),
                                           ([-1.2; 1.0], 1,
                                            x -> (x[1] - 1)^2,
                                            [-Inf; 0.0], [0.9, Inf],
                                            x -> [10 * (x[2] - x[1]^2)],
                                            [0.9; 0.81],
                                           ),
                                           ([-1.2; 1.0], 1,
                                            x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
                                            zeros(2), ones(2),
                                            x -> [(x[1] - 2)^2 + (x[2] - 2)^2 - 2],
                                            ones(2)
                                           ),
                                           ([2.0; 1.0], 2,
                                            x -> -x[1],
                                            [2.5; 3.5], [3.5; 4.5],
                                            x -> [x[1]^2 + x[2]^2 - 25; x[1] * x[2] - 12],
                                            [3.0; 4.0]
                                           )
                                          ]
      nlp = ADNLPModel(f, x0, lvar=lvar, uvar=uvar, c=c, lcon=zeros(m), ucon=zeros(m))
      output = with_logger(NullLogger()) do
        percival(nlp)
      end

      @test isapprox(output.solution, sol, rtol=1e-6)
      @test output.primal_feas < 1e-6
      @test output.dual_feas < 1e-6
      @test output.status == :first_order
    end
  end

  @testset "Small inequality constrained problems" begin
    for (x0, m, f, c, lcon, ucon, sol) in [([1.0; 2.0], 1,
                                            x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
                                            x -> [4x[1] + 6x[2] - 10],
                                            [-Inf], [0.0],
                                            ones(2)
                                           ),
                                           ([-1.2; 1.0], 1,
                                            x -> (x[1] - 1)^2 + 0.01 * x[2],
                                            x -> [10 * (x[2] - x[1]^2)],
                                            [0.0], [Inf],
                                            [0.990099;0.980296]
                                           ),
                                           ([-1.2; 1.0], 1,
                                            x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
                                            x -> [x[1]^2 + x[2]^2],
                                            [0.0], [4.0],
                                            ones(2)
                                           ),
                                           ([-1.2; 1.0], 1,
                                            x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
                                            x -> [(x[1] - 2)^2 + (x[2] - 2)^2],
                                            [2.0], [4.0],
                                            ones(2)
                                           ),
                                           ([2.0; 1.0], 2,
                                            x -> -x[1],
                                            x -> [x[1]^2 + x[2]^2; x[1] * x[2]],
                                            [25.0; 9.0], [30.0; 12.0],
                                            [5.196152; 1.732051]
                                           )
                                          ]
      nlp = ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon)
      output = with_logger(NullLogger()) do
        percival(nlp)
      end

      @test isapprox(output.solution, sol, rtol=1e-6)
      @test output.primal_feas < 1e-6
      @test output.dual_feas < 1e-6
      @test output.status == :first_order
    end
  end

end

test()
