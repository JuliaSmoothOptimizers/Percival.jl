export percival

using Logging, SolverCore, SolverTools, NLPModels

using JSOSolvers, Krylov

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
    solver = PercivalSolver(nlp)
    solve!(solver, nlp)

Implementation of an augmented Lagrangian method. The following keyword parameters can be passed:
- μ: Starting value of the penalty parameter (default: 10.0)
- atol: Absolute tolerance used in dual feasibility measure (default: 1e-8)
- rtol: Relative tolerance used in dual feasibility measure (default: 1e-8)
- ctol: (Absolute) tolerance used in primal feasibility measure (default: 1e-8)
- max_iter: Maximum number of iterations (default: 1000)
- max_time: Maximum elapsed time in seconds (default: 30.0)
- max_eval: Maximum number of objective function evaluations (default: 100000)
- subsolver_logger: Logger passed to `tron` (default: NullLogger)
- inity: Initial values of the Lagrangian multipliers
- subsolver_kwargs: subsolver keyword arguments as a dictionary
"""
mutable struct PercivalSolver{V}
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
  solver = PercivalSolver(nlp)
  solve!(Val(:equ), solver, nlp; kwargs...)
end

function solve!(
  ::Val{:equ},
  solver::PercivalSolver{V},
  nlp::AbstractNLPModel{T, V};
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
) where {T, V}
  if !(nlp.meta.minimize)
    error("Percival only works for minimization problem")
  end
  if nlp.meta.ncon == 0 || !equality_constrained(nlp)
    error(
      "percival(::Val{:equ}, nlp) should only be called for equality-constrained problems with bounded variables. Use percival(nlp)",
    )
  end

  counter_cost(nlp) = neval_obj(nlp) + 2 * neval_grad(nlp)

  x = solver.x
  gx = solver.gx
  x .= max.(nlp.meta.lvar, min.(nlp.meta.x0, nlp.meta.uvar))

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

  @info log_header(
    [:iter, :fx, :normgp, :normcx, :μ, :normy, :sumc, :inner_status, :iter_type],
    [Int, Float64, Float64, Float64, Float64, Float64, Int, Symbol, Symbol],
  )
  @info log_row(Any[iter, fx, normgp, normcx, al_nlp.μ, norm(y), counter_cost(nlp)])

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

    @info log_row(
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

  return GenericExecutionStats(
    status,
    nlp,
    solution = al_nlp.x,
    objective = fx,
    dual_feas = normgp,
    primal_feas = normcx,
    multipliers = y,
    iter = iter,
    elapsed_time = el_time,
  )
end
