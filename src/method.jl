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
function percival(
  ::Val{:equ},
  nlp::AbstractNLPModel{T, V};
  μ::Real = T(10.0),
  max_iter::Int = 2000,
  max_time::Real = 30.0,
  max_eval::Int = 200000,
  atol::Real = 1e-8,
  rtol::Real = 1e-8,
  ctol::Real = 1e-8,
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

  x = copy(nlp.meta.x0)
  x .= max.(nlp.meta.lvar, min.(x, nlp.meta.uvar))

  gp = zeros(T, nlp.meta.nvar)
  Jx = jac_op(nlp, x)
  fx, gx = objgrad(nlp, x)

  # Lagrange multiplier
  y = inity === nothing ? with_logger(subsolver_logger) do
    cgls(Jx', gx)[1]
  end : inity
  # tolerance
  η = 0.5
  ω = 1.0

  # create initial subproblem
  al_nlp = AugLagModel(nlp, y, T(μ), x, fx, cons(nlp, x) - nlp.meta.lcon)

  # stationarity measure
  gL = grad(nlp, x) - jtprod(nlp, x, y)
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
  tired = iter > max_iter || el_time > max_time || neval_obj(nlp) > max_eval

  while !(solved || infeasible || tired)
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
    gL = grad(nlp, al_nlp.x) - jtprod(nlp, al_nlp.x, al_nlp.y)
    project_step!(gp, al_nlp.x, -gL, nlp.meta.lvar, nlp.meta.uvar) # Proj(x - gL) - x
    normgp = norm(gp)

    iter += 1
    el_time = time() - start_time
    rem_eval = max_eval - neval_obj(nlp)
    solved = normgp ≤ ϵd && normcx ≤ ϵp
    infeasible = al_nlp.μ > 1e16 && norm(jtprod(nlp, al_nlp.x, al_nlp.cx)) < √ϵp * normcx
    tired = iter > max_iter || el_time > max_time || neval_obj(nlp) > max_eval || al_nlp.μ > 1e16

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
    elseif al_nlp.μ > 1e16
      status = :stalled
    end
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
