export AugLagModel

using NLPModels, LinearAlgebra, LinearOperators
using NLPModels: increment!

"""Given a model
  min f(x)  s.t.  c(x) = 0, l ≦ x ≦ u,
this new model represents the subproblem of the augmented Lagrangian method
  min f(x) - y'*c(x) + μ/2*‖c(x)‖^2  s.t.  l ≦ x ≦ u,
where y is an estimates of the Lagrange multiplier vector and μ is the penalty parameter.
"""

mutable struct AugLagModel <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
  model :: AbstractNLPModel
  y :: AbstractVector
  mu :: Real
  cx :: AbstractVector
end

function AugLagModel(model :: AbstractNLPModel, y :: AbstractVector, mu :: Real)

  x0 = model.meta.x0
  ncon = 0
  lvar = model.meta.lvar
  uvar = model.meta.uvar
  nnzh = model.meta.nnzh

  meta = NLPModelMeta(model.meta.nvar, x0 = x0, ncon = ncon, lvar = lvar, uvar = uvar)

  # Preallocation
  cx = zeros(model.meta.ncon)

  return AugLagModel(meta, Counters(), model, y, mu, cx)
end

function NLPModels.obj(nlp :: AugLagModel, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  cons!(nlp.model, x, nlp.cx)
  return obj(nlp.model, x) - dot(nlp.y, nlp.cx) + (nlp.mu / 2) * dot(nlp.cx, nlp.cx)
end

function NLPModels.grad!(nlp :: AugLagModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  cons!(nlp.model, x, nlp.cx)
  g .= grad(nlp.model, x) - jtprod(nlp.model, x, nlp.y) + nlp.mu * jtprod(nlp.model, x, nlp.cx)
end

function NLPModels.hprod!(nlp :: AugLagModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
  obj_weight :: Float64 = 1.0)
  cons!(nlp.model, x, nlp.cx)
  Jv = jprod(nlp.model, x, v)
  Hv .= hprod(nlp.model, x, v, obj_weight = obj_weight, y = nlp.mu * nlp.cx - nlp.y) + nlp.mu * jtprod(nlp.model, x, Jv)
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
