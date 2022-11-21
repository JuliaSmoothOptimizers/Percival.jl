@testset "Test callback" begin
  @testset "Test callback unconstrained" begin
    f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
    nlp = ADNLPModel(f, [-1.2; 1.0])
    function cb(nlp, solver, stats)
      if stats.iter == 4
        stats.status = :user
      end
    end
    stats = percival(nlp, verbose = 0, callback = cb)
    # just test that it runs
    @test stats.status != :unknown
  end

  @testset "Test callback equality-constrained" begin
    f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
    nlp = ADNLPModel(f, [-1.2; 1.0], x -> [sum(x)], ones(1), ones(1))
    function cb(nlp, solver, stats)
      if stats.iter == 4
        stats.status = :user
      end
    end
    stats = percival(nlp, verbose = 0, callback = cb)
    @test stats.iter == 4
  end

  @testset "Test callback inequality-constrained" begin
    f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
    nlp = ADNLPModel(f, [-1.2; 1.0], x -> [sum(x)], zeros(1), ones(1))
    function cb(nlp, solver, stats)
      if stats.iter == 4
        stats.status = :user
      end
    end
    stats = percival(nlp, verbose = 0, callback = cb)
    @test stats.iter == 4
  end
end
