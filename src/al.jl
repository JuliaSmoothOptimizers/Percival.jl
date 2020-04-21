export al

using Logging, SolverTools, NLPModels

using JSOSolvers, Krylov

"""Implementation of a augmented Lagrangian method for:

  min f(x)  s.t.  c(x) = 0, l ≦ x ≦ u"""

function al(nlp :: AbstractNLPModel; max_iter :: Int = 1000, max_time :: Real = 30.0, max_eval :: Int=-1, atol :: Real = 1e-7, rtol :: Real = 1e-7)

  T = eltype(nlp.meta.x0)

  if nlp.meta.ncon == 0 # unconstrained

    S = with_logger(NullLogger()) do
      tron(nlp)
    end

    x = S.solution
    status = S.status
    el_time = S.elapsed_time
    iter = S.iter
    normgp = S.dual_feas
    normcx = T.(0.0)

    return GenericExecutionStats(status, nlp, solution = x, objective = obj(nlp, x), dual_feas = normgp, primal_feas = normcx,
                                 iter = iter, elapsed_time = el_time)
  end

  # number of slack variables
  ns = nlp.meta.ncon - length(nlp.meta.jfix)

  # SlackModel create slack variables if necessary
  nlp = SlackModel(nlp)

  x = copy(nlp.meta.x0)
  x = T.(x)

  gp = zeros(T, nlp.meta.nvar)
  Jx = jac(nlp, x)
  fx, gx = objgrad(nlp, x)

  # penalty parameter
  μ = T.(10.0)
  # Lagrange multiplier
  y = T.(cgls(Jx', gx)[1])
  # tolerance
  eta = 0.5

  # create initial subproblem
  al_nlp = AugLagModel(nlp, y, μ, x, cons(nlp, x))

  # stationarity measure
  gL =  grad(nlp, x) - jtprod(nlp, x, y)
  project_step!(gp, x, -gL, T.(nlp.meta.lvar), T.(nlp.meta.uvar)) # Proj(x - gL) - x
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
    S = with_logger(NullLogger()) do
      tron(al_nlp, x = copy(al_nlp.x))
    end

    normcx = norm(al_nlp.cx)

    if normcx <= eta
      update_y!(al_nlp)
      eta = T.(eta / (al_nlp.mu)^0.9)
    else
      μ = 100 * μ
      update_mu!(al_nlp, μ)
      eta = T.(1 / μ^0.1)
    end

    # stationarity measure
    gL = grad(nlp, al_nlp.x) - jtprod(nlp, al_nlp.x, al_nlp.y)
    project_step!(gp, al_nlp.x, -gL, T.(nlp.meta.lvar), T.(nlp.meta.uvar)) # Proj(x - gL) - x
    normgp = norm(gp)

    iter += 1
    el_time = time() - start_time
    solved = normgp ≤ tol && normcx ≤ 1e-8
    tired = iter > max_iter || el_time > max_time

    @info log_row(Any[iter, S.objective, normgp, normcx])
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

  return GenericExecutionStats(status, nlp, solution = al_nlp.x[1:nlp.meta.nvar-ns], objective = obj(nlp, al_nlp.x), dual_feas = normgp, primal_feas = normcx,
                               iter = iter, elapsed_time = el_time)
end
