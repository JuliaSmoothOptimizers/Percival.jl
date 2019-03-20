using Logging, SolverTools

"""Implementation of a augmented Lagrangian method for:

	min f(x)  s.t.  c(x) = 0, l ≦ x ≦ u"""

function al(nlp :: AbstractNLPModel)

  x = copy(nlp.meta.x0)
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

  @info log_header([:iter, :normgL, :normcx], [Int, Float64, Float64])
  @info log_row(Any[iter, normgL, normcx])

  # TODO: Add keyword arguments atol, rtol, max_eval, max_iter
  solved = normgL ≤ 1e-5 && normcx ≤ 1e-5
  tired = iter ≥ 1000

  while !(solved || tired)

    # TODO: Reuse AugLagModel
		# create subproblem
		al_nlp = AugLagModel(nlp, y, μ)

		# solve subproblem
    S = with_logger(NullLogger()) do
      tron(al_nlp, x = x)
    end
		x = S.solution

		cx = c(x)
		gx = g(x)
		Jx = J(x)
		normcx = norm(cx)

		if normcx <= eta
			y = y - μ * cx
			#y = cgls(Jx', gx)[1]
			eta = eta / μ^0.9
		else
			μ = 100 * μ
			eta = 1 / μ^0.1
		end

    # TODO: Improve dual measure
    gL = gx - Jx' * y
    #project_step!(gpL, ...)
    normgL = norm(gL)
		iter += 1
    solved = normgL ≤ 1e-5 && normcx ≤ 1e-5
    tired = iter ≥ 1000

    @info log_row(Any[iter, normgL, normcx])
	end

  # TODO: Use GenericExecutionStats
	return x, obj(nlp,x), normgL, normcx, iter
end
