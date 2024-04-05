"""
    @wrappedallocs(expr)
Given an expression, this macro wraps that expression inside a new function
which will evaluate that expression and measure the amount of memory allocated
by the expression. Wrapping the expression in a new function allows for more
accurate memory allocation detection when using global variables (e.g. when
at the REPL).
For example, `@wrappedallocs(x + y)` produces:
```julia
function g(x1, x2)
    @allocated x1 + x2
end
g(x, y)
```
You can use this macro in a unit test to verify that a function does not
allocate:
```
@test @wrappedallocs(x + y) == 0
```
"""
macro wrappedallocs(expr)
  argnames = [gensym() for a in expr.args]
  quote
    function g($(argnames...))
      @allocated $(Expr(expr.head, argnames...))
    end
    $(Expr(:call, :g, [esc(a) for a in expr.args]...))
  end
end

if v"1.7" <= VERSION
  @testset "Allocation tests" begin
    @testset "Test 0-allocations of NLPModel API for AugLagModel" begin
      list_of_problems = NLPModelsTest.nlp_problems

      T = Float64
      for problem in list_of_problems
        nlp = eval(Symbol(problem))(T)
        if nlp.meta.ncon > 0
          μ = one(T)
          x = nlp.meta.x0
          fx = obj(nlp, x)
          y = nlp.meta.y0
          cx = similar(y)
          model = Percival.AugLagModel(nlp, y, μ, x, fx, cx)

          test_zero_allocations(model, exclude = [hess])
        end
      end
    end

    @testset "Allocation tests $(model)" for model in union(
      NLPModelsTest.nls_problems,
      NLPModelsTest.nlp_problems,
    )
      nlp = eval(Meta.parse(model))()

      nlp.meta.ncon > 0 || continue

      if !equality_constrained(nlp)
        nlp = nlp isa AbstractNLSModel ? SlackNLSModel(nlp) : SlackModel(nlp)
      end

      solver = PercivalSolver(nlp)
      stats = GenericExecutionStats(nlp)
      SolverCore.solve!(solver, nlp, stats)
      reset!(solver)
      reset!(nlp)
      al = @wrappedallocs SolverCore.solve!(solver, nlp, stats)
      @test al == 0
    end
  end
end
