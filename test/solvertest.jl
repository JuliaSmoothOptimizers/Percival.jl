@testset "Testing Percival with SolverTest" begin
  @testset "Percival NLP" for foo in [
    unconstrained_nlp,
    bound_constrained_nlp,
    equality_constrained_nlp,
  ]
    foo(nlp -> percival(nlp))
  end

  @testset "Percival NLS" for foo in [
    unconstrained_nls,
    bound_constrained_nls,
    equality_constrained_nls,
  ]
    foo(nlp -> percival(nlp))
  end

  @testset "Multiprecision tests" begin
    for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen]
      multiprecision_nlp(
        (nlp; kwargs...) -> percival(nlp; kwargs...),
        ptype,
        precisions = (Float16, Float32, Float64, BigFloat),
      ) # precisions = (Float16, Float32, Float64, BigFloat)
    end
  end
end
