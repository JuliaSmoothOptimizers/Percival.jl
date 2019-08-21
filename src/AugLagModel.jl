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
end

function AugLagModel(model :: AbstractNLPModel, y :: AbstractVector, mu :: Real)

	x0 = model.meta.x0
	ncon = 0
	lvar = model.meta.lvar
	uvar = model.meta.uvar

	meta = NLPModelMeta(model.meta.nvar, x0 = x0, ncon = ncon, lvar = lvar, uvar = uvar)

	return AugLagModel(meta, Counters(), model, y, mu)
end

function NLPModels.obj(nlp :: AugLagModel, x :: AbstractVector)
	increment!(nlp, :neval_obj)
	cx = cons(nlp.model, x)
	return obj(nlp.model, x) - dot(nlp.y, cx) + (nlp.mu / 2) * dot(cx,cx)
end

function NLPModels.grad(nlp :: AugLagModel, x :: AbstractVector)
	increment!(nlp, :neval_grad)
	cx = cons(nlp.model, x)
	return grad(nlp.model, x) - jtprod(nlp.model, x, nlp.y) + nlp.mu * jtprod(nlp.model, x, cx)
end

function NLPModels.hess(nlp :: AugLagModel, x :: AbstractVector; obj_weight :: Float64 = 1.0)
	increment!(nlp, :neval_hess)
	cx = cons(nlp.model, x)
	Jx = jac(nlp.model, x)
	return obj_weight * (hess(nlp.model, x, obj_weight = obj_weight, y = nlp.mu * cx - nlp.y) + nlp.mu * tril(Jx' * Jx))
end

function NLPModels.hprod(nlp :: AugLagModel, x :: AbstractVector, v :: AbstractVector; obj_weight :: Float64 = 1.0)
	increment!(nlp, :neval_hprod)
	cx = cons(nlp.model, x)
	Jv = jprod(nlp.model, x, v) # J(x)v
	return hprod(nlp.model, x, v, obj_weight = obj_weight, y = nlp.mu * cx - nlp.y) + nlp.mu * jtprod(nlp.model, x, Jv)
end

function NLPModels.hess_op(nlp :: AugLagModel, x :: AbstractVector; obj_weight :: Float64 = 1.0)
	return LinearOperator(Float64, nlp.meta.nvar, nlp.meta.nvar, true, true,
			v -> NLPModels.hprod(nlp, x, v; obj_weight = obj_weight))
end

function NLPModels.hprod!(nlp :: AugLagModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
	obj_weight :: Float64 = 1.0)
	# test
	cx = cons(nlp.model, x)
	Jv = jprod(nlp.model, x, v)
	Hv .= hprod(nlp.model, x, v, obj_weight = obj_weight, y = nlp.mu * cx - nlp.y) + nlp.mu * jtprod(nlp.model, x, Jv)
end

function NLPModels.hess_op!(nlp :: AugLagModel, x :: AbstractVector, Hv :: AbstractVector;
	obj_weight :: Float64 = 1.0)
	return LinearOperator(Float64, nlp.meta.nvar, nlp.meta.nvar, true, true,
			v -> NLPModels.hprod!(nlp, x, v, Hv; obj_weight = obj_weight))
end
