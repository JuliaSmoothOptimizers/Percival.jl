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
  callback = (args...) -> nothing,
  max_iter::Int = 2000,
  max_time::Real = 30.0,
  max_eval::Int = 200000,
  atol::Real = 1e-8,
  rtol::Real = 1e-8,
  verbose::Integer = 0,
  subproblem_modifier = identity,
  kwargs...,
)
  if !(unconstrained(nlp) || bound_constrained(nlp))
    error(
      "percival(::Val{:tron}, nlp) should only be called for unconstrained or bound-constrained problems. Use percival(nlp)",
    )
  end
  @warn "Problem does not have general constraints; calling tron"
  return tron(
    subproblem_modifier(nlp);
    callback = callback,
    atol = atol,
    rtol = rtol,
    max_eval = max_eval,
    max_iter = max_iter,
    max_time = max_time,
    verbose = verbose,
    kwargs...,
  )
end

function percival(::Val{:ineq}, nlp::AbstractNLPModel; x::V = nlp.meta.x0, kwargs...) where {V}
  if nlp.meta.ncon == 0 || equality_constrained(nlp)
    error(
      "percival(::Val{:ineq}, nlp) should only be called for problems with inequalities. Use percival(nlp)",
    )
  end
  snlp = nlp isa AbstractNLSModel ? SlackNLSModel(nlp) : SlackModel(nlp)
  if length(x) != snlp.meta.nvar
    x0 = fill!(V(undef, snlp.meta.nvar), zero(eltype(V)))
    x0[1:(nlp.meta.nvar)] .= x
  else
    x0 = x
  end
  output = percival(Val(:equ), snlp; x = x0, kwargs...)
  output.solution = output.solution[1:(nlp.meta.nvar)]
  return output
end

"""
    percival(nlp)

A factorization-free augmented Lagrangian for nonlinear optimization.

For advanced usage, first define a `PercivalSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = PercivalSolver(nlp)
    solve!(solver, nlp)

    stats = GenericExecutionStats(nlp)
    solver = PercivalSolver(nlp)
    solve!(solver, nlp, stats)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess;
- `atol::T = T(1e-8)`: absolute tolerance;
- `rtol::T = T(1e-8)`: relative tolerance;
- `ctol::T = atol > 0 ? atol : rtol`: absolute tolerance on the feasibility;
- `max_eval::Int = 100000`: maximum number of evaluation of the objective function;
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_iter::Int = 2000`: maximum number of iterations;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `μ::Real = T(10.0)`: Starting value of the penalty parameter;
- `η₀::T = T(0.5)`: Starting value for the contraints tolerance of the subproblem;
- `ω₀::T = T(1)`: Starting value for relative tolerance of the subproblem;
- `α₁::T = T(9 // 10)`: ``η = max(1 / al_nlp.μ^α₁, ϵp)`` if ``‖c(xᵏ)‖ ≤ η``;
- `β₁::T = T(1 // 10)`: ``η = max(1 / al_nlp.μ^β₁, ϵp)`` if ``‖c(xᵏ)‖ > η``;
- `μ_up::T = T(10)`: Multiplicative factor of `μ` if not ``‖c(xᵏ)‖ > η``;
- `subsolver_logger::AbstractLogger = NullLogger()`: logger passed to `tron`;
- `cgls_verbose::Int = 0`: verbosity level in `Krylov.cgls`;
- `inity::Bool = false`: If `true` the algorithm uses `Krylov.cgls` to compute an approximation, otherwise we use `nlp.meta.y0`;
other `kwargs` are passed to the subproblem solver.

The algorithm stops when ``‖c(xᵏ)‖ ≤ ctol`` and ``‖P∇L(xᵏ,λᵏ)‖ ≤ atol + rtol * ‖P∇L(x⁰,λ⁰)‖`` where ``P∇L(x,λ) := Proj_{l,u}(x - ∇f(x) + ∇c(x)ᵀλ) - x``.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.dual_feas`: norm of current projected gradient of Lagrangian;
  - `stats.primal_feas`: norm of the feasibility residual;
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.multipliers`: current estimate of Lagrange multiplier associated with the equality constraint;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.

# Examples
```jldoctest
using Percival, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3), x -> [x[1]], zeros(1), zeros(1))
stats = percival(nlp)

# output

"Execution stats: first-order stationary"
```
```jldoctest
using Percival, ADNLPModels, SolverCore
nlp = ADNLPModel(x -> sum(x.^2), ones(3), x -> [x[1]], zeros(1), zeros(1))
stats = GenericExecutionStats(nlp)
solver = PercivalSolver(nlp)
stats = solve!(solver, nlp, stats)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct PercivalSolver{T, V, Op, M, ST} <: AbstractOptimizationSolver
  x::V
  xc::V
  y::V
  gx::V
  gL::V
  gp::V
  Jv::V
  Jtv::V
  Jx::Op
  cgls_solver::CglsSolver{T, T, V}
  sub_pb::AugLagModel{M, T, V}
  sub_solver::ST
  sub_stats::GenericExecutionStats{T, V, V, Any}
end

function PercivalSolver(
  nlp::AbstractNLPModel{T, V};
  subproblem_modifier = identity,
  kwargs...,
) where {T, V}
  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
  x = V(undef, nvar)
  xc = V(undef, nvar)
  y = V(undef, ncon)
  gx = V(undef, nvar)
  gL = V(undef, nvar)
  gp = V(undef, nvar)

  Jv = V(undef, ncon)
  Jtv = V(undef, nvar)

  Jx = jac_op!(nlp, x, Jv, Jtv)
  Op = typeof(Jx)
  cgls_solver = CglsSolver(Jx', gx)

  sub_pb = AugLagModel(nlp, V(undef, ncon), T(0), x, T(0), V(undef, ncon))
  model = subproblem_modifier(sub_pb)
  sub_solver = if typeof(model) <: AbstractNLSModel
    TronNLSSolver(model; kwargs...)
  else
    TronSolver(model; kwargs...)
  end
  ST = typeof(sub_solver)
  sub_stats = GenericExecutionStats(sub_pb)
  return PercivalSolver{T, V, Op, typeof(nlp), ST}(
    x,
    xc,
    y,
    gx,
    gL,
    gp,
    Jv,
    Jtv,
    Jx,
    cgls_solver,
    sub_pb,
    sub_solver,
    sub_stats,
  )
end

# List of keywords accepted by PercivalSolver
const trustregion_keys = (
  :max_radius,
  :acceptance_threshold,
  :decrease_threshold,
  :increase_threshold,
  :large_decrease_factor,
  :small_decrease_factor,
  :increase_factor,
)

@doc (@doc PercivalSolver) function percival(
  ::Val{:equ},
  nlp::AbstractNLPModel;
  subproblem_modifier = identity,
  kwargs...,
)
  if !(nlp.meta.minimize)
    error("Percival only works for minimization problem")
  end
  if nlp.meta.ncon == 0 || !equality_constrained(nlp)
    error(
      "percival(::Val{:equ}, nlp) should only be called for equality-constrained problems with bounded variables. Use percival(nlp)",
    )
  end
  subsolver_kwargs = Dict(kwargs)
  subsolver_keys = intersect(keys(subsolver_kwargs), trustregion_keys)
  solver_kwargs = Dict(k => subsolver_kwargs[k] for k in subsolver_keys)
  solver = PercivalSolver(nlp; subproblem_modifier = subproblem_modifier, solver_kwargs...)
  sub_kwargs = copy(subsolver_kwargs)
  for k in subsolver_keys
    pop!(sub_kwargs, k)
  end
  SolverCore.solve!(solver, nlp; subproblem_modifier = subproblem_modifier, sub_kwargs...)
end

function SolverCore.reset!(solver::PercivalSolver)
  solver
end
function SolverCore.reset!(solver::PercivalSolver, model::AbstractNLPModel)
  solver.Jx = jac_op!(model, solver.x, solver.Jv, solver.Jtv)
  solver.sub_pb.model = model
  solver.sub_pb.meta.x0 .= model.meta.x0
  solver.sub_pb.meta.lvar .= model.meta.lvar
  solver.sub_pb.meta.uvar .= model.meta.uvar
  solver
end

function reinit!(al_nlp::AugLagModel{M, T, V}, model::M, fx::T, μ::T, x::V, y::V) where {M, T, V}
  reset!(al_nlp)
  al_nlp.store_Jv .= zero(T)
  al_nlp.store_Jtv .= zero(T)
  al_nlp.fx = fx
  al_nlp.y .= y
  al_nlp.x .= x
  al_nlp.μ = μ
  cons!(model, x, al_nlp.cx)
  al_nlp.cx .-= model.meta.lcon
  al_nlp.μc_y .= μ .* al_nlp.cx .- y
  al_nlp
end

counter_cost(nlp) = neval_obj(nlp) + 2 * neval_grad(nlp)

"""
    reset_subproblem!(solver::PercivalSolver{T, V}, model::AbstractNLPModel{T, V})

Specialize `SolverCore.reset!` function to percival's context.
"""
function reset_subproblem!(solver::PercivalSolver{T, V}, model::AbstractNLPModel{T, V}) where {T, V}
  reset!(solver.sub_solver, model)
end

function reset_subproblem!(
  solver::PercivalSolver{T, V, Op, M, ST},
  model::AugLagModel{M, T, V},
) where {T, V, Op, M, ST <: TronSolver{T, V}}
  solver.sub_solver.xc .= model.x
  reset!(solver.sub_solver)
end

function SolverCore.solve!(
  solver::PercivalSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  μ::Real = T(10),
  η₀::T = T(1 // 2),
  ω₀::T = T(1),
  α₁::T = T(9 // 10),
  β₁::T = T(1 // 10),
  μ_up::T = T(10),
  max_iter::Int = 2000,
  max_time::Real = 30.0,
  max_eval::Int = 200000,
  atol::Real = T(1e-8),
  rtol::Real = T(1e-8),
  ctol::Real = atol > 0 ? atol : rtol,
  subsolver_verbose::Int = 0,
  cgls_verbose::Int = 0,
  inity::Bool = false,
  subproblem_modifier = identity,
  subsolver_max_eval = max_eval,
  verbose::Integer = 0,
  kwargs...,
) where {T, V}
  reset!(stats)
  @lencheck nlp.meta.nvar x
  x = solver.x .= x
  gx = solver.gx
  x .= max.(nlp.meta.lvar, min.(x, nlp.meta.uvar))

  al_nlp = solver.sub_pb

  gp = solver.gp
  gp .= zero(T)
  Jx = solver.Jx
  fx, gx = if nlp isa AbstractNLSModel
    objgrad!(nlp, x, gx, solver.sub_pb.Fx)
  else
    objgrad!(nlp, x, gx)
  end
  set_objective!(stats, fx)

  # Lagrange multiplier
  y = solver.y
  if inity
    cgls!(solver.cgls_solver, Jx', gx, verbose = cgls_verbose)
    y = solver.cgls_solver.x
  else
    y .= nlp.meta.y0
  end
  set_constraint_multipliers!(stats, y)

  # create initial subproblem
  reinit!(al_nlp, nlp, fx, μ, x, y)

  # stationarity measure
  jtprod!(nlp, x, y, solver.Jtv)
  gL = solver.gL
  gL .= gx .- solver.Jtv
  gL .*= -1
  project_step!(gp, x, gL, nlp.meta.lvar, nlp.meta.uvar) # Proj(x - gL) - x
  normgp = norm(gp)
  normcx = norm(al_nlp.cx)
  set_residuals!(stats, normcx, normgp)

  # tolerance for optimal measure
  ϵd = atol + rtol * normgp
  ϵp = ctol
  # tolerance
  η = max(η₀, ϵp)
  ω = ω₀

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  rem_eval = max_eval

  if verbose > 0
    @info log_header(
      [:iter, :fx, :normgp, :normcx, :μ, :normy, :sumc, :inner_status, :iter_type],
      [Int, Float64, Float64, Float64, Float64, Float64, Int, Symbol, Symbol],
    )
    @info log_row(Any[stats.iter, fx, normgp, normcx, al_nlp.μ, norm(y), counter_cost(nlp)])
  end

  solved = normgp ≤ ϵd && normcx ≤ ϵp

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      iter = stats.iter,
      optimal = solved,
      infeasible = false,
      penalty_too_large = false,
      max_eval = max_eval,
      max_time = max_time,
      max_iter = max_iter,
    ),
  )

  callback(nlp, solver, stats)

  done = stats.status != :unknown

  while !done
    # solve subproblem
    model = subproblem_modifier(al_nlp)
    reset_subproblem!(solver, model)
    S = solver.sub_stats
    reset!(S)
    solve!(
      solver.sub_solver,
      model,
      S;
      x = al_nlp.x,
      cgtol = ω,
      rtol = ω,
      atol = ω,
      max_time = max_time - stats.elapsed_time,
      max_eval = min(subsolver_max_eval, rem_eval),
      verbose = subsolver_verbose,
      max_cgiter = nlp.meta.nvar,
      kwargs...,
    )

    inner_status = S.status

    normcx = norm(al_nlp.cx)
    fx = S.objective + dot(al_nlp.y, al_nlp.cx) - normcx^2 * al_nlp.μ / 2
    set_objective!(stats, fx)

    iter_type = if normcx <= η
      update_y!(al_nlp)
      η = max(η / al_nlp.μ^α₁, ϵp)
      ω /= al_nlp.μ
      set_constraint_multipliers!(stats, al_nlp.y)
      :update_y
    else
      update_μ!(al_nlp, μ_up * al_nlp.μ)
      η = max(1 / al_nlp.μ^β₁, ϵp)
      ω = 1 / al_nlp.μ
      :update_μ
    end

    # stationarity measure
    if nlp isa AbstractNLSModel
      grad!(nlp, al_nlp.x, gx, al_nlp.Fx)
    else
      grad!(nlp, al_nlp.x, gx)
    end
    jtprod!(nlp, al_nlp.x, al_nlp.y, solver.Jtv)
    gL .= gx .- solver.Jtv
    gL .*= -1
    project_step!(gp, al_nlp.x, gL, nlp.meta.lvar, nlp.meta.uvar) # Proj(x - gL) - x
    normgp = norm(gp)
    set_residuals!(stats, normcx, normgp)

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    rem_eval = max_eval - neval_obj(nlp)
    solved = normgp ≤ ϵd && normcx ≤ ϵp
    jtprod!(nlp, al_nlp.x, al_nlp.cx, solver.Jtv)
    penalty_too_large = al_nlp.μ > 1 / eps(T)
    infeasible = penalty_too_large && norm(solver.Jtv) < √ϵp * normcx

    verbose > 0 &&
      mod(stats.iter, verbose) == 0 &&
      @info log_row(
        Any[
          stats.iter,
          fx,
          normgp,
          normcx,
          al_nlp.μ,
          norm(al_nlp.y),
          counter_cost(nlp),
          inner_status,
          iter_type,
        ],
      )

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        iter = stats.iter,
        optimal = solved,
        infeasible = infeasible,
        penalty_too_large = penalty_too_large,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter,
      ),
    )

    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  stats
end

function get_status(
  nlp;
  elapsed_time = 0.0,
  iter = 0,
  optimal = false,
  infeasible = false,
  penalty_too_large = false,
  unbounded = false,
  max_eval = Inf,
  max_time = Inf,
  max_iter = Inf,
)
  if optimal
    :first_order
  elseif infeasible
    :infeasible
  elseif unbounded
    :unbounded
  elseif iter > max_iter
    :max_iter
  elseif neval_obj(nlp) > max_eval ≥ 0
    :max_eval
  elseif elapsed_time > max_time
    :max_time
  elseif penalty_too_large
    :stalled
  else
    :unknown
  end
end
