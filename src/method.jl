export percival

using Logging, SolverTools, NLPModels

using JSOSolvers, Krylov

function percival(nlp :: AbstractNLPModel; kwargs...)
  if unconstrained(nlp) || bound_constrained(nlp)
    return percival(Val(:tron), nlp; kwargs...)
  elseif equality_constrained(nlp)
    return percival(Val(:equ), nlp; kwargs...)
  else # has inequalities
    return percival(Val(:ineq), nlp; kwargs...)
  end
end

function percival(::Val{:tron}, nlp :: AbstractNLPModel;
                  max_iter :: Int = 1000, max_time :: Real = 30.0, max_eval :: Int=100000,
                  atol :: Real = 1e-8, rtol :: Real = 1e-8,
                  subsolver_logger :: AbstractLogger=NullLogger(),
                 )
  if !(unconstrained(nlp) || bound_constrained(nlp))
    error("percival(::Val{:tron}, nlp) should only be called for unconstrained or bound-constrained problems. Use percival(nlp)")
  end
  @warn "Problem does not have general constraints; calling tron"
  return tron(nlp, subsolver_logger=subsolver_logger, atol=atol, rtol=rtol, max_eval=max_eval, max_time=max_time)
end

function percival(::Val{:ineq}, nlp :: AbstractNLPModel; kwargs...)
  if nlp.meta.ncon == 0 || equality_constrained(nlp)
    error("percival(::Val{:ineq}, nlp) should only be called for problems with inequalities. Use percival(nlp)")
  end
  snlp = SlackModel(nlp)
  output = percival(Val(:equ), snlp; kwargs...)
  output.solution = output.solution[1:nlp.meta.nvar]
  return output
end

"""Implementation of an augmented Lagrangian method for:

  min f(x)  s.t.  c(x) = 0, l ≦ x ≦ u"""

function percival(::Val{:equ}, nlp :: AbstractNLPModel; μ :: Real = eltype(nlp.meta.x0)(10.0),
            max_iter :: Int = 1000, max_time :: Real = 30.0, max_eval :: Int=100000,
            atol :: Real = 1e-8, rtol :: Real = 1e-8,
            subsolver_logger :: AbstractLogger=NullLogger(),
           )
  if nlp.meta.ncon == 0 || !equality_constrained(nlp)
    error("percival(::Val{:equ}, nlp) should only be called for equality-constrained problems with bounded variables. Use percival(nlp)")
  end

  T = eltype(nlp.meta.x0)

  x = copy(nlp.meta.x0)
  x = T.(x)

  gp = zeros(T, nlp.meta.nvar)
  Jx = jac_op(nlp, x)
  fx, gx = objgrad(nlp, x)
  lvar = eltype(nlp.meta.lvar) == T ? nlp.meta.lvar : T.(nlp.meta.lvar)
  uvar = eltype(nlp.meta.uvar) == T ? nlp.meta.uvar : T.(nlp.meta.uvar)

  # Lagrange multiplier
  y = with_logger(subsolver_logger) do
    cgls(Jx', gx)[1]
  end
  # tolerance
  η = 0.5
  ω = 1.0

  # create initial subproblem
  al_nlp = AugLagModel(nlp, y, T(μ), x, cons(nlp, x))

  # stationarity measure
  gL =  grad(nlp, x) - jtprod(nlp, x, y)
  project_step!(gp, x, -gL, lvar, uvar) # Proj(x - gL) - x
  normgp = norm(gp)
  normcx = norm(al_nlp.cx)

  # tolerance for optimal measure
  ϵd = atol + rtol * normgp
  ϵp = atol

  iter = 0
  start_time = time()
  el_time = 0.0

  @info log_header([:iter, :fx, :normgp, :normcx, :μ, :normy, :sumc, :inner_status, :iter_type],
                   [Int, Float64, Float64, Float64, Float64, Float64, Int, Symbol, Symbol])
  @info log_row(Any[iter, fx, normgp, normcx, al_nlp.μ, norm(y), sum_counters(nlp)])

  solved = normgp ≤ ϵd && normcx ≤ ϵp
  infeasible = false
  tired = iter > max_iter || el_time > max_time || neval_obj(nlp) > max_eval

  while !(solved || infeasible || tired)
    # solve subproblem
    S = with_logger(subsolver_logger) do
      tron(al_nlp, x = copy(al_nlp.x), rtol=ω, atol=ω, max_time=max_time-el_time)
    end
    inner_status = S.status

    if normcx ≤ ϵp && inner_status != :success
      Hx = hess_op(nlp, al_nlp.x, -al_nlp.μc_y)
      Jx = jac_op(nlp,  al_nlp.x)
      zerop = opZeros(nlp.meta.ncon, nlp.meta.ncon)
      W = [Hx  -Jx'; -Jx  zerop]
      cx = al_nlp.cx
      rhs = [-gp; cx]
      Δxy = symmlq(W, rhs)[1]
      Δx = Δxy[1:nlp.meta.nvar]
      project_step!(Δx, al_nlp.x, -Δx, lvar, uvar)
      update_cx!(al_nlp, al_nlp.x + Δx)
      inner_status = :safeguard
    end

    normcx = norm(al_nlp.cx)
    fx = S.objective + dot(al_nlp.y, al_nlp.cx) - normcx^2 * al_nlp.μ / 2

    iter_type = if normcx <= η
      update_y!(al_nlp)
      η = max(η / al_nlp.μ^T(0.9), ϵp)
      ω /= al_nlp.μ
      :update_y
    else
      update_μ!(al_nlp, 100 * al_nlp.μ)
      η = max(1 / al_nlp.μ^T(0.1), ϵp)
      ω = 1 / al_nlp.μ
      :update_μ
    end

    # stationarity measure
    gL = grad(nlp, al_nlp.x) - jtprod(nlp, al_nlp.x, al_nlp.y)
    project_step!(gp, al_nlp.x, -gL, lvar, uvar) # Proj(x - gL) - x
    normgp = norm(gp)

    iter += 1
    el_time = time() - start_time
    solved = normgp ≤ ϵd && normcx ≤ ϵp
    infeasible = al_nlp.μ > 1e16 && norm(jtprod(nlp, al_nlp.x, al_nlp.cx)) < √ϵp * normcx
    tired = iter > max_iter || el_time > max_time || neval_obj(nlp) > max_eval || al_nlp.μ > 1e16

    @info log_row(Any[iter, fx, normgp, normcx, al_nlp.μ, norm(y), sum_counters(nlp), inner_status, iter_type])
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

  return GenericExecutionStats(status, nlp, solution = al_nlp.x,
                               objective = fx, dual_feas = normgp, primal_feas = normcx,
                               multipliers = y, iter = iter, elapsed_time = el_time)
end
