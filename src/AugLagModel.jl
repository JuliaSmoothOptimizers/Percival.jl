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
  μ    :: T
  x     :: V # save last iteration of subsolver
  cx    :: V # save last constraint value of subsolver
  μc_y :: V # y - μ * cx
end

function AugLagModel(model :: AbstractNLPModel, y :: AbstractVector, μ :: AbstractFloat, x :: AbstractVector, cx :: AbstractVector)
  @lencheck model.meta.ncon y cx
  @lencheck model.meta.nvar x
  μ ≥ 0 || error("Penalty parameter μ should be ≥ 0")

  meta = NLPModelMeta(model.meta.nvar, x0=model.meta.x0, lvar=model.meta.lvar, uvar=model.meta.uvar)

  return AugLagModel(meta, Counters(), model, y, μ, x, cx, y - μ * cx)
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
  Jv = jprod(nlp.model, x, v)
  Hv .= hprod(nlp.model, x, nlp.μc_y, v, obj_weight = obj_weight) + nlp.μ * jtprod(nlp.model, x, Jv)
  return Hv
end

#function NLPModels.hess_structure!(nlp :: AugLagModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
#  return hess_structure!(nlp.model, rows, cols) # because is the same structure of hessian of f(x)
#end

#=
function NLPModels.hess_coord!(nlp :: AugLagModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer},
                    vals :: AbstractVector; obj_weight :: Float64 = 1.0)
  # Hessian of auglag
  Hx = NLPModels.hess(nlp, x, obj_weight = obj_weight)

  # accessing by columns and storing elements in vals
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j] # in place not working
      k += 1
    end
  end

  return rows, cols, vals
end
=#
