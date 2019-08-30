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
    nnzh = model.meta.nnzh

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

function NLPModels.grad!(nlp :: AugLagModel, x :: AbstractVector, g :: AbstractVector)
	increment!(nlp, :neval_grad)
	cx = cons(nlp.model, x)
	g .= grad(nlp.model, x) - jtprod(nlp.model, x, nlp.y) + nlp.mu * jtprod(nlp.model, x, cx)
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
	prod = v -> NLPModels.hprod(nlp, x, v; obj_weight = obj_weight)
	F = typeof(prod)
	return LinearOperator{Float64,F,F,F}(nlp.meta.nvar, nlp.meta.nvar, true, true, prod, prod, prod)
end

function NLPModels.hprod!(nlp :: AugLagModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
	obj_weight :: Float64 = 1.0)
	cx = cons(nlp.model, x)
	Jv = jprod(nlp.model, x, v)
	Hv .= hprod(nlp.model, x, v, obj_weight = obj_weight, y = nlp.mu * cx - nlp.y) + nlp.mu * jtprod(nlp.model, x, Jv)
end

function NLPModels.hess_op!(nlp :: AugLagModel, x :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64 = 1.0)
	prod = v -> NLPModels.hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
	F = typeof(prod)
	return LinearOperator{Float64,F,F,F}(nlp.meta.nvar, nlp.meta.nvar, true, true, prod, prod, prod)
end

function NLPModels.hess_structure(nlp :: AugLagModel)
    rows = Vector{Int}(undef, nlp.meta.nnzh) # nnzh (nonzeros) is number of elements needed to store values
    cols = Vector{Int}(undef, nlp.meta.nnzh)
    NLPModels.hess_structure!(nlp, rows, cols)
end

function NLPModels.hess_structure!(nlp :: AugLagModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
    return hess_structure!(nlp.model, rows, cols) # because is the same structure of hessian of f(x)
end

function NLPModels.hess_coord(nlp :: AugLagModel, x :: AbstractVector)

    rows = Vector{Int}(undef, nlp.meta.nnzh)
    cols = Vector{Int}(undef, nlp.meta.nnzh)
    vals = Vector{eltype(x)}(undef, nlp.meta.nnzh)
    NLPModels.hess_structure!(nlp, rows, cols)
    return NLPModels.hess_coord!(nlp, x, rows, cols, vals)
end

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

