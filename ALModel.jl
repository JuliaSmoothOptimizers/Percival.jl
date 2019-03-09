"""Given a model
	min f(x)  s.t.  c(x) = 0, l ≦ x ≦ u,
this new model represents the subproblem of the augmented Lagrangian method
	min f(x) - y'*c(x) + μ/2*‖c(x)‖^2  s.t.  l ≦ x ≦ u,
where y is an estimates of the Lagrange multiplier vector and μ is the penalty parameter.
"""

#import NLPModels
using LinearAlgebra

mutable struct ALModel <: AbstractNLPModel
	meta :: NLPModelMeta
	model :: AbstractNLPModel
	y :: AbstractVector
	mu :: Real
end

function ALModel(model :: AbstractNLPModel, y :: AbstractVector, mu :: Real)

	x0 = model.meta.x0
	lvar = model.meta.lvar
	uvar = model.meta.uvar

	meta = NLPModelMeta(model.meta.nvar, x0 = x0, lvar = lvar, uvar = uvar)

	return ALModel(meta, model, y, mu)
end

function NLPModels.obj(nlp :: ALModel, x :: AbstractVector)
	cx = cons(nlp.model, x)
	return obj(nlp.model, x) - nlp.y'*cx + (nlp.mu/2)*dot(cx,cx)
end
