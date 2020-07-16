export AugLagModel, update_cx!, update_y!, update_μ!

using NLPModels, LinearAlgebra, LinearOperators
using NLPModels: increment!, @lencheck # @lencheck is not exported in 0.12.0

"""Given a model
  min f(x)  s.t.  c(x) = 0, l ≦ x ≦ u,
this new model represents the subproblem of the augmented Lagrangian method
  min f(x) - y'*c(x) + μ/2*‖c(x)‖^2  s.t.  l ≦ x ≦ u,
where y is an estimates of the Lagrange multiplier vector and μ is the penalty parameter.
"""

mutable struct AugLagModel{M <: AbstractNLPModel, T <: AbstractFloat, V <: AbstractVector} <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
  model :: M
  y     :: V
  μ     :: T
  x     :: V # save last iteration of subsolver
  cx    :: V # save last constraint value of subsolver
  μc_y  :: V # y - μ * cx
  store_Jv   :: Vector{T}
  store_JtJv :: Vector{T}
end

function AugLagModel(model :: AbstractNLPModel, y :: AbstractVector, μ :: AbstractFloat, x :: AbstractVector, cx :: AbstractVector)
  nvar, ncon = model.meta.nvar, model.meta.ncon
  @lencheck ncon y cx
  @lencheck nvar x
  μ ≥ 0 || error("Penalty parameter μ should be ≥ 0")

  meta = NLPModelMeta(nvar, x0=model.meta.x0, lvar=model.meta.lvar, uvar=model.meta.uvar)
  T = eltype(x)

  return AugLagModel(meta, Counters(), model, y, μ, x, cx, y - μ * cx, zeros(T, ncon), zeros(T, nvar))
end

function update_cx!(nlp :: AbstractNLPModel, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  if x != nlp.x
    cons!(nlp.model, x, nlp.cx)
    nlp.x .= x
    nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
  end
end

function update_y!(nlp :: AbstractNLPModel)
  nlp.y .= -nlp.μc_y
  nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
end

function update_μ!(nlp :: AbstractNLPModel, μ :: AbstractFloat)
  nlp.μ = μ
  nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
end

function NLPModels.obj(nlp :: AugLagModel, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  update_cx!(nlp, x)
  return obj(nlp.model, x) - dot(nlp.y, nlp.cx) + (nlp.μ / 2) * dot(nlp.cx, nlp.cx)
end

function NLPModels.grad!(nlp :: AugLagModel, x :: AbstractVector, g :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar g
  increment!(nlp, :neval_grad)
  update_cx!(nlp, x)
  grad!(nlp.model, x, g)
  g .+= jtprod(nlp.model, x, nlp.μc_y)
  return g
end

function NLPModels.objgrad!(nlp :: AugLagModel, x :: AbstractVector, g :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar g
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  update_cx!(nlp, x)
  f = obj(nlp.model, x) - dot(nlp.y, nlp.cx) + (nlp.μ / 2) * dot(nlp.cx, nlp.cx)
  grad!(nlp.model, x, g)
  g .+= jtprod(nlp.model, x, nlp.μc_y)
  return f, g
end

function NLPModels.hprod!(nlp :: AugLagModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64 = 1.0)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar v
  @lencheck nlp.meta.nvar Hv
  increment!(nlp, :neval_hprod)
  update_cx!(nlp, x)
  jprod!(nlp.model, x, v, nlp.store_Jv)
  jtprod!(nlp.model, x, nlp.store_Jv, nlp.store_JtJv)
  hprod!(nlp.model, x, nlp.μc_y, v, Hv, obj_weight = obj_weight)
  Hv .+= nlp.μ * nlp.store_JtJv
  return Hv
end