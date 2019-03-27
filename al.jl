using Logging, SolverTools

"""Implementation of a augmented Lagrangian method for:

	min f(x)  s.t.  c(x) = 0, l ≦ x ≦ u"""

function al(nlp :: AbstractNLPModel)

	x = copy(nlp.meta.x0)
	gp = zeros(nlp.meta.nvar)
	cx = cons(nlp, x)
	gx = grad(nlp, x)
	Jx = jac(nlp, x)

	# penalty parameter
	μ = 10
	# Lagrange multiplier
	y = cgls(Jx', gx)[1]
	# tolerance
	eta = 0.5

	# create initial subproblem
	al_nlp = AugLagModel(nlp, y, μ)

	# stationarity measure
	gLA = grad(al_nlp, x)
	project_step!(gp, x, -gLA, nlp.meta.lvar, nlp.meta.uvar) # Proj(x - gLA) - x
	normgp = norm(gp)

	normcx = norm(cx)
	iter = 0

 	@info log_header([:iter, :normgp, :normcx], [Int, Float64, Float64])
	@info log_row(Any[iter, normgp, normcx])

	# TODO: Add keyword arguments atol, rtol, max_eval, max_iter
	solved = normgp ≤ 1e-5 && normcx ≤ 1e-5
	tired = iter ≥ 1000

	while !(solved || tired)

		# solve subproblem
		S = with_logger(NullLogger()) do
			tron(al_nlp, x = x)
		end
		x = S.solution
		cx = cons(nlp, x)
		normcx = norm(cx)

		if normcx <= eta
			al_nlp.y = al_nlp.y - al_nlp.mu * cx
			eta = eta / (al_nlp.mu)^0.9
		else
			μ = 100 * μ
			al_nlp.mu  = μ
			eta = 1 / μ^0.1
		end

		# stationarity measure
		gLA = grad(al_nlp, x)
		project_step!(gp, x, -gLA, nlp.meta.lvar, nlp.meta.uvar) # Proj(x - gLA) - x
		normgp = norm(gp)

		iter += 1
		solved = normgp ≤ 1e-5 && normcx ≤ 1e-5
		tired = iter ≥ 1000

		@info log_row(Any[iter, normgp, normcx])
	end

	# TODO: Use GenericExecutionStats
	return x, obj(nlp,x), normgp, normcx, iter
end
