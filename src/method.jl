export percival, PercivalSolver, solve!

using Logging, SolverCore, SolverTools, NLPModels

using JSOSolvers, Krylov
import SolverCore.solve!

function percival(nlp::AbstractNLPModel; kwargs...)
  if unconstrained(nlp) || bound_constrained(nlp)
    return percival(Val(:tron), nlp; kwargs...)
  elseif equality_constrained(nlp)
    return percival(Val(:equ), nlp; kwargs...)
  else # has inequalities
    return percival(Val(:ineq), nlp; kwargs...)
  end
end

function percival(
  ::Val{:tron},
  nlp::AbstractNLPModel;
  max_iter::Int = 2000,
  max_time::Real = 30.0,
  max_eval::Int = 200000,
  atol::Real = 1e-8,
  rtol::Real = 1e-8,
  verbose::Integer = 0,
  subproblem_modifier = identity,
  subsolver_logger::AbstractLogger = NullLogger(),
  subsolver_kwargs = Dict(:max_cgiter => nlp.meta.nvar),
)
  if !(unconstrained(nlp) || bound_constrained(nlp))
    error(
      "percival(::Val{:tron}, nlp) should only be called for unconstrained or bound-constrained problems. Use percival(nlp)",
    )
  end
  @warn "Problem does not have general constraints; calling tron"
  return tron(
    subproblem_modifier(nlp);
    subsolver_logger = subsolver_logger,
    atol = atol,
    rtol = rtol,
    max_eval = max_eval,
    max_time = max_time,
    verbose = verbose,
    subsolver_kwargs...,
  )
end

function percival(::Val{:ineq}, nlp::AbstractNLPModel; kwargs...)
  if nlp.meta.ncon == 0 || equality_constrained(nlp)
    error(
      "percival(::Val{:ineq}, nlp) should only be called for problems with inequalities. Use percival(nlp)",
    )
  end
  snlp = SlackModel(nlp)
  output = percival(Val(:equ), snlp; kwargs...)
  output.solution = output.solution[1:(nlp.meta.nvar)]
  return output
end

"""
    percival(nlp)

A factorization-free augmented Lagrangian for nonlinear optimization.

For advanced usage, first define a `PercivalSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = PercivalSolver(nlp)
    solve!(solver, nlp)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess;
- `atol::T = T(1e-8)`: absolute tolerance.
- `rtol::T = T(1e-8)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `ctol::T = T(1e-8)`: absolute tolerance on the feasibility ‖c(xᵏ)‖ ≤ ctol.
- `max_eval::Int = 100000`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = 2000`: maximum number of iterations.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `μ::Real = T(10.0)`: Starting value of the penalty parameter.
- `subsolver_logger::AbstractLogger = NullLogger()`: logger passed to `tron`.
- `inity = nothing`: initial values of the Lagrangian multipliers. If `nothing` the algorithm uses `Krylov.cgls` to compute an approximation.
- `subsolver_kwargs = Dict(:max_cgiter => nlp.meta.nvar)`: subsolver keyword arguments as a dictionary.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Examples
```jldoctest
using Percival, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3), x -> [x[1]], zeros(1), zeros(1))
stats = percival(nlp)

# output

"Execution stats: first-order stationary"
```
```jldoctest
using Percival, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3), x -> [x[1]], zeros(1), zeros(1))
solver = PercivalSolver(nlp)
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct PercivalSolver{V} <: AbstractOptimizationSolver
  x::V
  gx::V
  gL::V
  gp::V
  Jtv::V
end

function PercivalSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  gx = V(undef, nvar)
  gL = V(undef, nvar)
  gp = V(undef, nvar)
  Jtv = V(undef, nvar)
  return PercivalSolver{V}(x, gx, gL, gp, Jtv)
end

@doc (@doc PercivalSolver) function percival(::Val{:equ}, nlp::AbstractNLPModel; kwargs...)
  if !(nlp.meta.minimize)
    error("Percival only works for minimization problem")
  end
  if nlp.meta.ncon == 0 || !equality_constrained(nlp)
    error(
      "percival(::Val{:equ}, nlp) should only be called for equality-constrained problems with bounded variables. Use percival(nlp)",
    )
  end
  solver = PercivalSolver(nlp)
  SolverCore.solve!(solver, nlp; kwargs...)
end

function SolverCore.reset!(solver::PercivalSolver)
  solver
end
SolverCore.reset!(solver::PercivalSolver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::PercivalSolver{V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  x::V = nlp.meta.x0,
  μ::Real = T(10.0),
  max_iter::Int = 2000,
  max_time::Real = 30.0,
  max_eval::Int = 200000,
  atol::Real = T(1e-8),
  rtol::Real = T(1e-8),
  ctol::Real = T(1e-8),
  subsolver_logger::AbstractLogger = NullLogger(),
  inity = nothing,
  subproblem_modifier = identity,
  subsolver_max_eval = max_eval,
  subsolver_kwargs = Dict(:max_cgiter => nlp.meta.nvar),
  verbose::Integer = 0,
) where {T, V}
  counter_cost(nlp) = neval_obj(nlp) + 2 * neval_grad(nlp)

  x = solver.x .= x
  gx = solver.gx
  x .= max.(nlp.meta.lvar, min.(x, nlp.meta.uvar))

  gp = solver.gp
  gp .= zero(T)
  Jx = jac_op(nlp, x)
  fx, gx = objgrad!(nlp, x, gx)

  # Lagrange multiplier
  y = inity === nothing ? with_logger(subsolver_logger) do
    cgls(Jx', gx)[1]
  end : inity
  # tolerance
  η = T(0.5)
  ω = T(1.0)

  # create initial subproblem
  al_nlp = AugLagModel(nlp, y, T(μ), x, fx, cons(nlp, x) - nlp.meta.lcon)

  # stationarity measure
  jtprod!(nlp, x, y, solver.Jtv)
  gL = solver.gL
  gL .= gx .- solver.Jtv
  project_step!(gp, x, -gL, nlp.meta.lvar, nlp.meta.uvar) # Proj(x - gL) - x
  normgp = norm(gp)
  normcx = norm(al_nlp.cx)

  # tolerance for optimal measure
  ϵd = atol + rtol * normgp
  ϵp = ctol

  iter = 0
  start_time = time()
  el_time = 0.0
  rem_eval = max_eval

  if verbose > 0
    @info log_header(
      [:iter, :fx, :normgp, :normcx, :μ, :normy, :sumc, :inner_status, :iter_type],
      [Int, Float64, Float64, Float64, Float64, Float64, Int, Symbol, Symbol],
    )
    @info log_row(Any[iter, fx, normgp, normcx, al_nlp.μ, norm(y), counter_cost(nlp)])
  end

  solved = normgp ≤ ϵd && normcx ≤ ϵp
  infeasible = false
  penalty_too_large = false
  tired = iter > max_iter || el_time > max_time || neval_obj(nlp) > max_eval

  while !(solved || infeasible || tired || penalty_too_large)
    # solve subproblem
    S = with_logger(subsolver_logger) do
      tron(
        subproblem_modifier(al_nlp);
        x = copy(al_nlp.x),
        cgtol = ω,
        rtol = ω,
        atol = ω,
        max_time = max_time - el_time,
        max_eval = min(subsolver_max_eval, rem_eval),
        subsolver_kwargs...,
      )
    end
    inner_status = S.status

    normcx = norm(al_nlp.cx)
    fx = S.objective + dot(al_nlp.y, al_nlp.cx) - normcx^2 * al_nlp.μ / 2

    iter_type = if normcx <= η
      update_y!(al_nlp)
      η = max(η / al_nlp.μ^T(0.9), ϵp)
      ω /= al_nlp.μ
      :update_y
    else
      update_μ!(al_nlp, 10 * al_nlp.μ)
      η = max(1 / al_nlp.μ^T(0.1), ϵp)
      ω = 1 / al_nlp.μ
      :update_μ
    end

    # stationarity measure
    grad!(nlp, al_nlp.x, gx)
    jtprod!(nlp, al_nlp.x, al_nlp.y, solver.Jtv)
    gL .= gx .- solver.Jtv
    project_step!(gp, al_nlp.x, -gL, nlp.meta.lvar, nlp.meta.uvar) # Proj(x - gL) - x
    normgp = norm(gp)

    iter += 1
    el_time = time() - start_time
    rem_eval = max_eval - neval_obj(nlp)
    solved = normgp ≤ ϵd && normcx ≤ ϵp
    jtprod!(nlp, al_nlp.x, al_nlp.cx, solver.Jtv)
    penalty_too_large = al_nlp.μ > 1 / eps(T)
    infeasible = penalty_too_large && norm(solver.Jtv) < √ϵp * normcx
    tired = iter > max_iter || el_time > max_time || neval_obj(nlp) > max_eval

    verbose > 0 && mod(iter, verbose) == 0 && @info log_row(
      Any[iter, fx, normgp, normcx, al_nlp.μ, norm(y), counter_cost(nlp), inner_status, iter_type],
    )
  end

  if solved
    status = :first_order
  elseif infeasible
    status = :infeasible
  elseif tired
    if iter > max_iter
      status = :max_iter
    elseif el_time > max_time
      status = :max_time
    elseif neval_obj(nlp) > max_eval
      status = :max_eval
    end
  elseif penalty_too_large
    status = :stalled
  end

  set_status!(stats, status)
  set_solution!(stats, al_nlp.x)
  set_objective!(stats, fx)
  set_residuals!(stats, normcx, normgp)
  set_iter!(stats, iter)
  set_time!(stats, el_time)
  set_constraint_multipliers!(stats, y)
  stats
end
