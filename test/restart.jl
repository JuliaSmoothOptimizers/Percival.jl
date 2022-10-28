@testset "Test restart with a different initial guess" begin
  nlp = ADNLPModels.ADNLPModel(
    x -> 0.0,
    [-1.2; 1.0],
    [1],
    [1],
    [-1.0],
    x -> [10 * (x[2] - x[1]^2)],
    [-1.0, 0.0],
    [-1.0, 0.0],
    name = "mgh01feas";
  )

  stats = GenericExecutionStats(nlp)
  solver = PercivalSolver(nlp)
  SolverCore.solve!(solver, nlp, stats)
  @test stats.status_reliable && stats.status == :first_order
  @test stats.solution_reliable && isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  nlp.meta.x0 .= 10.0
  SolverCore.reset!(solver)

  SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10)
  @test stats.status_reliable && stats.status == :first_order
  @test stats.solution_reliable && isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)
end

@testset "Test restart with a different problem" begin
  nlp = ADNLPModels.ADNLPModel(
    x -> 0.0,
    [-1.2; 1.0],
    [1],
    [1],
    [-1.0],
    x -> [10 * (x[2] - x[1]^2)],
    [-1.0, 0.0],
    [-1.0, 0.0],
    name = "mgh01feas";
  )

  stats = GenericExecutionStats(nlp)
  solver = PercivalSolver(nlp)
  stats = SolverCore.solve!(solver, nlp, stats)
  @test stats.status_reliable && stats.status == :first_order
  @test stats.solution_reliable && isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  f2(x) = (x[1])^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f2, [-1.2; 1.0])
  nlp = ADNLPModels.ADNLPModel(
    x -> 0.0,
    [-1.2; 1.0],
    [1],
    [1],
    [-1.0],
    x -> [10 * (x[2] - x[1]^2)],
    [0.0, 0.0],
    [0.0, 0.0],
    name = "mgh01feas";
  )
  SolverCore.reset!(solver, nlp)

  stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10)
  @test stats.status_reliable && stats.status == :first_order
  @test stats.solution_reliable && isapprox(stats.solution, [0.0; 0.0], atol = 1e-6)
end
