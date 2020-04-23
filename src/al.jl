export al

using Logging, SolverTools, NLPModels

using JSOSolvers, Krylov

function al(nlp :: AbstractNLPModel; kwargs...)
  if unconstrained(nlp) || bound_constrained(nlp)
    return al(Val(:tron), nlp; kwargs...)
  elseif equality_constrained(nlp)
    return al(Val(:equ), nlp; kwargs...)
  else # has inequalities
    return al(Val(:ineq), nlp; kwargs...)
  end
end

function al(::Val{:tron}, nlp :: AbstractNLPModel;
            max_iter :: Int = 1000, max_time :: Real = 30.0, max_eval :: Int=-1,
            atol :: Real = 1e-8, rtol :: Real = 1e-8,
            subsolver_logger :: AbstractLogger=NullLogger(),
           )
  if !(unconstrained(nlp) || bound_constrained(nlp))
    error("al(::Val{:tron}, nlp) should only be called for unconstrained or bound-constrained problems. Use al(nlp)")
  end
  @warn "Problem does not have general constraints; calling tron"
  return tron(nlp, subsolver_logger=subsolver_logger, atol=atol, rtol=rtol, max_eval=max_eval, max_time=max_time)
end

function al(::Val{:ineq}, nlp :: AbstractNLPModel; kwargs...)
  if nlp.meta.ncon == 0 || equality_constrained(nlp)
    error("al(::Val{:ineq}, nlp) should only be called for problems with inequalities. Use al(nlp)")
  end
  snlp = SlackModel(nlp)
  output = al(Val(:equ), snlp; kwargs...)
  output.solution = output.solution[1:nlp.meta.nvar]
  return output
end

"""Implementation of an augmented Lagrangian method for:

  min f(x)  s.t.  c(x) = 0, l ≦ x ≦ u"""

function al(::Val{:equ}, nlp :: AbstractNLPModel; mu :: Real = eltype(nlp.meta.x0)(10.0),
            max_iter :: Int = 1000, max_time :: Real = 30.0, max_eval :: Int=-1,
            atol :: Real = 1e-8, rtol :: Real = 1e-8,
            subsolver_logger :: AbstractLogger=NullLogger(),
           )
  if nlp.meta.ncon == 0 || !equality_constrained(nlp)
    error("al(::Val{:equ}, nlp) should only be called for equality-constrained problems with bounded variables. Use al(nlp)")
  end

  T = eltype(nlp.meta.x0)

  x = copy(nlp.meta.x0)
  x = T.(x)

  gp = zeros(T, nlp.meta.nvar)
  Jx = jac(nlp, x)
  fx, gx = objgrad(nlp, x)
  lvar = eltype(nlp.meta.lvar) == T ? nlp.meta.lvar : T.(nlp.meta.lvar)
  uvar = eltype(nlp.meta.uvar) == T ? nlp.meta.uvar : T.(nlp.meta.uvar)

  # Lagrange multiplier
  y = with_logger(subsolver_logger) do
    cgls(Jx', gx)[1]
  end
  # tolerance
  eta = 0.5

  # create initial subproblem
  al_nlp = AugLagModel(nlp, y, T(mu), x, cons(nlp, x))

  # stationarity measure
  gL =  grad(nlp, x) - jtprod(nlp, x, y)
  project_step!(gp, x, -gL, lvar, uvar) # Proj(x - gL) - x
  normgp = norm(gp)
  normcx = norm(al_nlp.cx)

  # tolerance for optimal measure
  tol = atol + rtol*normgp

  iter = 0
  start_time = time()
  el_time = 0.0

  @info log_header([:iter, :fx, :normgp, :normcx], [Int, Float64, Float64, Float64])
  @info log_row(Any[iter, fx, normgp, normcx])

  solved = normgp ≤ tol && normcx ≤ 1e-8
  tired = iter > max_iter || el_time > max_time

  #adaptive tolerance
  #atol = 0.5

  while !(solved || tired)
    # solve subproblem
    S = with_logger(subsolver_logger) do
      tron(al_nlp, x = copy(al_nlp.x))
    end

    normcx = norm(al_nlp.cx)
    fx = S.objective + dot(al_nlp.y, al_nlp.cx) - normcx^2 * al_nlp.mu / 2

    if normcx <= eta
      update_y!(al_nlp)
      eta = eta / al_nlp.mu^T(0.9)
    else
      update_mu!(al_nlp, 100 * al_nlp.mu)
      eta = 1 / al_nlp.mu^T(0.1)
    end

    # stationarity measure
    gL = grad(nlp, al_nlp.x) - jtprod(nlp, al_nlp.x, al_nlp.y)
    project_step!(gp, al_nlp.x, -gL, lvar, uvar) # Proj(x - gL) - x
    normgp = norm(gp)

    iter += 1
    el_time = time() - start_time
    solved = normgp ≤ tol && normcx ≤ 1e-8
    tired = iter > max_iter || el_time > max_time

    @info log_row(Any[iter, fx, normgp, normcx])
  end

  if solved
    status = :first_order
  elseif tired
    if iter > max_iter
      status = :max_iter
    end
    if el_time > max_time
      status = :max_time
    end
  end

  return GenericExecutionStats(status, nlp, solution = al_nlp.x,
                               objective = fx, dual_feas = normgp, primal_feas = normcx,
                               multipliers = y, iter = iter, elapsed_time = el_time)
end
