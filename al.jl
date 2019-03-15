"""Implementation of a augmented Lagrangian method for:

	min f(x)  s.t.  c(x) = 0, l ≦ x ≦ u"""

function al(nlp :: AbstractNLPModel)

	x = nlp.meta.x0
	c(x) = cons(nlp, x)
	g(x) = grad(nlp, x)
	J(x) = jac(nlp, x)
	cx = c(x)
	gx = g(x)
	Jx = J(x)

	# penalty parameter
	μ = 10
	# Lagrange multiplier
	y = cgls(Jx', gx)[1]
	# tolerance
	eta = 0.5

	normgL = norm(gx - Jx' * y)
	normcx = norm(cx)
	iter = 0

	while (normgL > 1e-5 || normcx > 1e-5) && (iter < 1000)

		# create subproblem
		al_nlp = AugLagModel(nlp, y, μ)

		# solve subproblem
		S = tron(al_nlp, x = x)
		x = S.solution

		cx = c(x)
		gx = g(x)
		Jx = J(x)
		normcx = norm(cx)

		if normcx <= eta
			#y = y - μ * cx
			y = cgls(Jx', gx)[1]
			eta = eta / μ^0.9
		else
			μ = 100 * μ
			eta = 1 / μ^0.1
		end

		normgL = norm(gx - Jx' * y)
		iter += 1
	end

	return x, obj(nlp,x), normgL, normcx, iter
end
