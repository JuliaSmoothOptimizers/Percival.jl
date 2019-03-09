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
	μ = 1
	# Lagrange multiplier
	y = cgls(Jx', gx)[1]

	# tolerance
	eta = 1e-5

	normgL = norm(gx - Jx'*y)
	normcx = norm(cx)
	iter = 0

	while (normgL > 1e-5 || normcx > 1e-5) && (iter < 1000)

		al_nlp = ADNLPModel(x -> obj(nlp,x) - y'*c(x) + (μ/2)*dot(c(x),c(x)), x, lvar=nlp.meta.lvar, uvar=nlp.meta.uvar)
		x, fx = ipopt(al_nlp)

		#al_nlp = ALModel(nlp, y, μ)
		#? = tron(al_nlp,x)

		cx = c(x)
		gx = g(x)
		Jx = J(x)

		normcx = norm(cx)
		if normcx <= eta
			y = y - μ*cx
			#eta =
		else
			μ = 100*μ
			#eta =
		end

		normgL = norm(gx - Jx'*y)
		iter += 1
	end

	return x, obj(nlp,x), normgL, normcx, iter
end
