export SPGSubSolver

using SolverCore, NLPModels, LinearAlgebra

"""
    SPGSubSolver(nlp; mem = 10)

A spectral projected gradient (SPG / Birgin–Martínez–Raydan) solver for
bound-constrained problems

    min f(x)   s.t.   ℓ ≤ x ≤ u,

intended as a **GPU-compatible subproblem solver** for Percival's augmented
Lagrangian: pass `subsolver = SPGSubSolver` to `percival`. Every operation in
the hot loop is a whole-array broadcast, a `dot`, or an `obj`/`grad!` call, so
the solver runs unchanged on `Vector`, `CuVector`, `JLArray`, ... — unlike the
default `TronSolver`, whose projected-CG inner loop uses scalar `x[i]`
indexing that GPU arrays disallow.

Only first-order information is used (`obj`, `grad!`), so the augmented
Lagrangian subproblem needs no Hessian — use it with `subproblem_modifier =
identity` (the default) to avoid the quasi-Newton wrapper entirely.
"""
mutable struct SPGSubSolver{T, V} <: AbstractOptimizationSolver
  x::V
  gx::V
  xt::V
  gt::V
  d::V
  s::V
  y::V
  fhist::Vector{T}
end

function SPGSubSolver(nlp::AbstractNLPModel{T, V}; mem::Int = 10, kwargs...) where {T, V}
  n = nlp.meta.nvar
  vec() = fill!(V(undef, n), zero(T))
  return SPGSubSolver{T, V}(vec(), vec(), vec(), vec(), vec(), vec(), vec(), fill(T(-Inf), mem))
end

SolverCore.reset!(solver::SPGSubSolver) = solver
SolverCore.reset!(solver::SPGSubSolver, ::AbstractNLPModel) = solver

function SolverCore.solve!(
  solver::SPGSubSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats;
  x::V = nlp.meta.x0,
  atol::Real = sqrt(eps(T)),
  rtol::Real = sqrt(eps(T)),
  max_iter::Int = 10_000,
  max_time::Real = 60.0,
  max_eval::Int = -1,
  verbose::Integer = 0,
  αmin::T = T(1e-10),
  αmax::T = T(1e10),
  σ1::T = T(0.1),
  σ2::T = T(0.9),
  γ::T = T(1e-4),
  kwargs...,  # swallow Tron-only kwargs (cgtol, max_cgiter, ...)
) where {T, V}
  SolverCore.reset!(stats)
  ℓ, u = nlp.meta.lvar, nlp.meta.uvar
  xk, gk, xt, gt = solver.x, solver.gx, solver.xt, solver.gt
  d, s, y = solver.d, solver.s, solver.y
  fill!(solver.fhist, T(-Inf))
  mem = length(solver.fhist)

  xk .= x
  xk .= clamp.(xk, ℓ, u)
  fk = obj(nlp, xk)
  grad!(nlp, xk, gk)
  solver.fhist[1] = fk

  # First projected-gradient residual sets the stopping threshold.
  d .= clamp.(xk .- gk, ℓ, u) .- xk
  ω0 = maximum(abs, d)
  ϵ = atol + rtol * ω0

  α = one(T)
  start_time = time()
  iter = 0
  neval = 1
  status = :max_iter
  while true
    d .= clamp.(xk .- α .* gk, ℓ, u) .- xk
    ωk = maximum(abs, d)
    if ωk ≤ ϵ
      status = :first_order
      break
    end
    if iter ≥ max_iter
      status = :max_iter
      break
    end
    if time() - start_time > max_time
      status = :max_time
      break
    end
    if max_eval ≥ 0 && neval ≥ max_eval
      status = :max_eval
      break
    end
    # Nonmonotone Armijo line search along the projected direction.
    fref = maximum(view(solver.fhist, 1:min(iter + 1, mem)))
    gtd = dot(gk, d)
    λ = one(T)
    ft = fk
    accepted = false
    for _ in 1:30
      xt .= xk .+ λ .* d
      ft = obj(nlp, xt)
      neval += 1
      if ft ≤ fref + γ * λ * gtd
        accepted = true
        break
      end
      # Safeguarded quadratic backtracking of the step length.
      λn = -gtd * λ^2 / (2 * (ft - fk - λ * gtd))
      λ = (σ1 * λ ≤ λn ≤ σ2 * λ) ? λn : λ / 2
    end
    grad!(nlp, xt, gt)
    # Barzilai–Borwein spectral step from (s, y).
    s .= xt .- xk
    y .= gt .- gk
    sy = dot(s, y)
    ss = dot(s, s)
    α = sy > 0 ? clamp(ss / sy, αmin, αmax) : αmax
    xk .= xt
    gk .= gt
    fk = ft
    iter += 1
    solver.fhist[(iter % mem) + 1] = fk
    if verbose > 0 && iter % verbose == 0
      @info "SPG" iter fk ωk α
    end
    accepted || (α = max(α / 2, αmin))
  end

  SolverCore.set_solution!(stats, xk)
  SolverCore.set_objective!(stats, fk)
  SolverCore.set_residuals!(stats, zero(T), maximum(abs, d))
  SolverCore.set_iter!(stats, iter)
  SolverCore.set_time!(stats, time() - start_time)
  SolverCore.set_status!(stats, status)
  return stats
end
