export AugLagModel, update_cx!, update_y!, update_μ!

using NLPModels, NLPModelsModifiers, LinearAlgebra, LinearOperators
using NLPModels: increment!, @lencheck # @lencheck is not exported in 0.12.0

@doc raw"""
    AugLagModel(model, y, μ, x, fx, cx)

Given a model
```math
\min \ f(x) \quad s.t. \quad c(x) = 0, \quad l ≤ x ≤ u,
```
this new model represents the subproblem of the augmented Lagrangian method
```math
\min \ f(x) - yᵀc(x) + \tfrac{1}{2} μ \|c(x)\|^2 \quad s.t. \quad l ≤ x ≤ u,
```
where y is an estimates of the Lagrange multiplier vector and μ is the penalty parameter.

In addition to keeping `meta` and `counters` as any NLPModel, an AugLagModel also stores
- `model`: The internal model defining ``f``, ``c`` and the bounds,
- `y`: The multipliers estimate,
- `μ`: The penalty parameter,
- `x`: Reference to the last point at which the function `c(x)` was computed,
- `fx`: Reference to `f(x)`,
- `cx`: Reference to `c(x)`,
- `μc_y`: storage for y - μ * cx,
- `store_Jv` and `store_JtJv`: storage used in `hprod!`.

Use the functions `update_cx!`, `update_y!` and `update_μ!` to update these values.
"""
mutable struct AugLagModel{M <: AbstractNLPModel, T <: AbstractFloat, V <: AbstractVector} <:
               AbstractNLPModel{T, V}
  meta::NLPModelMeta{T, V}
  counters::Counters
  model::M
  y::V
  μ::T
  x::V # save last iteration of subsolver
  fx::T # save last objective value of subsolver
  cx::V # save last constraint value of subsolver
  μc_y::V # y - μ * cx
  store_Jv::Vector{T}
  store_Jtv::Vector{T}
end

function AugLagModel(model::AbstractNLPModel{T, V}, y::V, μ::T, x::V, fx::T, cx::V) where {T, V}
  nvar, ncon = model.meta.nvar, model.meta.ncon
  @lencheck ncon y cx
  @lencheck nvar x
  μ ≥ 0 || error("Penalty parameter μ should be ≥ 0")

  meta = NLPModelMeta(
    nvar,
    x0 = model.meta.x0,
    lvar = model.meta.lvar,
    uvar = model.meta.uvar,
    name = "AugLagModel-$(model.meta.name)",
  )

  return AugLagModel(
    meta,
    Counters(),
    model,
    y,
    μ,
    x,
    fx,
    cx,
    y - μ * cx,
    zeros(T, ncon),
    zeros(T, nvar),
  )
end

"""
    update_cx!(nlp, x)

Given an `AugLagModel`, if `x != nlp.x`, then updates the internal value `nlp.cx` calling `cons`
on `nlp.model`, and reset `nlp.fx` to a NaN. Also updates `nlp.μc_y`.
"""
function update_cx!(nlp::AugLagModel, x::AbstractVector{T}) where {T}
  @lencheck nlp.meta.nvar x
  if x != nlp.x
    cons!(nlp.model, x, nlp.cx)
    nlp.cx .-= nlp.model.meta.lcon
    nlp.x .= x
    nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
    nlp.fx = T(NaN)
  end
end

"""
    update_fxcx!(nlp, x)

Given an `AugLagModel`, if `x != nlp.x`, then updates the internal value `nlp.cx` calling `objcons`
on `nlp.model`. Also updates `nlp.μc_y`. Returns fx only.
"""
function update_fxcx!(nlp::AugLagModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  if x != nlp.x
    nlp.fx, _ = objcons!(nlp.model, x, nlp.cx)
    nlp.cx .-= nlp.model.meta.lcon
    nlp.x .= x
    nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
  elseif isnan(nlp.fx)
    nlp.fx = obj(nlp.model, x)
  end
end

"""
    update_y!(nlp)

Given an `AugLagModel`, update `nlp.y = -nlp.μc_y` and updates `nlp.μc_y` accordingly.
"""
function update_y!(nlp::AugLagModel)
  nlp.y .= .-nlp.μc_y
  nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
end

"""
    update_μ!(nlp, μ)

Given an `AugLagModel`, updates `nlp.μ = μ` and `nlp.μc_y` accordingly.
"""
function update_μ!(nlp::AugLagModel, μ::AbstractFloat)
  nlp.μ = μ
  nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
end

function NLPModels.obj(nlp::AugLagModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  update_fxcx!(nlp, x)
  return nlp.fx - dot(nlp.y, nlp.cx) + (nlp.μ / 2) * dot(nlp.cx, nlp.cx)
end

function NLPModels.grad!(nlp::AugLagModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar g
  increment!(nlp, :neval_grad)
  update_cx!(nlp, x)
  grad!(nlp.model, x, g)
  g .+= jtprod!(nlp.model, x, nlp.μc_y, nlp.store_Jtv)
  return g
end

function NLPModels.objgrad!(nlp::AugLagModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar g
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  update_fxcx!(nlp, x)
  f = nlp.fx - dot(nlp.y, nlp.cx) + (nlp.μ / 2) * dot(nlp.cx, nlp.cx)
  grad!(nlp.model, x, g)
  g .+= jtprod!(nlp.model, x, nlp.μc_y, nlp.store_Jtv)
  return f, g
end

function NLPModels.hprod!(
  nlp::AugLagModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar v
  @lencheck nlp.meta.nvar Hv
  increment!(nlp, :neval_hprod)
  update_cx!(nlp, x)
  jprod!(nlp.model, x, v, nlp.store_Jv)
  jtprod!(nlp.model, x, nlp.store_Jv, nlp.store_Jtv)
  hprod!(nlp.model, x, nlp.μc_y, v, Hv, obj_weight = obj_weight)
  Hv .+= nlp.μ .* nlp.store_Jtv
  return Hv
end
