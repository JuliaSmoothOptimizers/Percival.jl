var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [Percival]","category":"page"},{"location":"reference/#Percival.AugLagModel","page":"Reference","title":"Percival.AugLagModel","text":"AugLagModel(model, y, μ, x, cx)\n\nGiven a model\n\nmin  f(x) quad st quad c(x) = 0 quad l  x  u\n\nthis new model represents the subproblem of the augmented Lagrangian method\n\nmin  f(x) - yᵀc(x) + tfrac12 μ c(x)^2 quad st quad l  x  u\n\nwhere y is an estimates of the Lagrange multiplier vector and μ is the penalty parameter.\n\nIn addition to keeping meta and counters as any NLPModel, an AugLagModel also stores\n\nmodel: The internal model defining f, c and the bounds,\ny: The multipliers estimate,\nμ: The penalty parameter,\nx: Reference to the last point at which the function c(x) was computed,\ncx: Reference to c(x),\nμc_y: storage for y - μ * cx,\nstore_Jv and store_JtJv: storage used in hprod!.\n\nUse the functions update_cx!, update_y! and update_μ! to update these values.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Percival.percival-Tuple{Val{:equ}, NLPModels.AbstractNLPModel}","page":"Reference","title":"Percival.percival","text":"percival(nlp)\n\nImplementation of an augmented Lagrangian method. The following keyword parameters can be passed:\n\nμ: Starting value of the penalty parameter (default: 10.0)\natol: Absolute tolerance used in dual feasibility measure (default: 1e-8)\nrtol: Relative tolerance used in dual feasibility measure (default: 1e-8)\nctol: (Absolute) tolerance used in primal feasibility measure (default: 1e-8)\nmax_iter: Maximum number of iterations (default: 1000)\nmax_time: Maximum elapsed time in seconds (default: 30.0)\nmax_eval: Maximum number of objective function evaluations (default: 100000)\nsubsolver_logger: Logger passed to tron (default: NullLogger)\ninity: Initial values of the Lagrangian multipliers\nsubsolver_kwargs: subsolver keyword arguments as a dictionary\n\n\n\n\n\n","category":"method"},{"location":"reference/#Percival.update_cx!-Tuple{AugLagModel, AbstractVector{T} where T}","page":"Reference","title":"Percival.update_cx!","text":"update_cx!(nlp, x)\n\nGiven an AugLagModel, if x != nlp.x, then updates the internal value nlp.cx calling cons on nlp.model. Also updates nlp.μc_y.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Percival.update_fxcx!-Tuple{AugLagModel, AbstractVector{T} where T}","page":"Reference","title":"Percival.update_fxcx!","text":"update_fxcx!(nlp, x)\n\nGiven an AugLagModel, if x != nlp.x, then updates the internal value nlp.cx calling objcons on nlp.model. Also updates nlp.μc_y. Returns fx only.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Percival.update_y!-Tuple{AugLagModel}","page":"Reference","title":"Percival.update_y!","text":"update_y!(nlp)\n\nGiven an AugLagModel, update nlp.y = -nlp.μc_y and updates nlp.μc_y accordingly.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Percival.update_μ!-Tuple{AugLagModel, AbstractFloat}","page":"Reference","title":"Percival.update_μ!","text":"update_μ!(nlp, μ)\n\nGiven an AugLagModel, updates nlp.μ = μ and nlp.μc_y accordingly.\n\n\n\n\n\n","category":"method"},{"location":"#Percival.jl","page":"Percival.jl","title":"Percival.jl","text":"","category":"section"},{"location":"","page":"Percival.jl","title":"Percival.jl","text":"an Augmented Lagrangian method","category":"page"},{"location":"","page":"Percival.jl","title":"Percival.jl","text":"Pages = [\"index.md\"]","category":"page"},{"location":"#Description","page":"Percival.jl","title":"Description","text":"","category":"section"},{"location":"","page":"Percival.jl","title":"Percival.jl","text":"This package implements a JSO-compliant augmented Lagrangian method based on the paper","category":"page"},{"location":"","page":"Percival.jl","title":"Percival.jl","text":"S. Arreckx, A. Lambe, Martins, J. R. R. A., & Orban, D. (2016).\nA Matrix-Free Augmented Lagrangian Algorithm with Application to Large-Scale Structural Design Optimization.\nOptimization And Engineering, 17, 359–384. doi:10.1007/s11081-015-9287-9","category":"page"},{"location":"","page":"Percival.jl","title":"Percival.jl","text":"It was implemented as part of the Master's dissertation of Egmara Antunes.","category":"page"},{"location":"#JSO-compliant","page":"Percival.jl","title":"JSO-compliant","text":"","category":"section"},{"location":"","page":"Percival.jl","title":"Percival.jl","text":"The percival method expects a single mandatory argument - an NLPModel - and returns a GenericExecutionStats from SolverCore.jl.","category":"page"},{"location":"#Main-exported-functions-and-types","page":"Percival.jl","title":"Main exported functions and types","text":"","category":"section"},{"location":"","page":"Percival.jl","title":"Percival.jl","text":"percival: The function to call the method. Pass an NLPModel to it.\nAugLagModel: A model representing the augmented Lagrangian subproblem, that allows better use of memory.","category":"page"},{"location":"#Example","page":"Percival.jl","title":"Example","text":"","category":"section"},{"location":"","page":"Percival.jl","title":"Percival.jl","text":"How to solve the simple problem","category":"page"},{"location":"","page":"Percival.jl","title":"Percival.jl","text":"min  (x_1 - 1)^2 + 100 (x_2 - x_1^2)^2 quad textsto quad x_1^2 + x_2^2 leq 1","category":"page"},{"location":"","page":"Percival.jl","title":"Percival.jl","text":"using ADNLPModels, Percival\n\nnlp = ADNLPModel(\n    x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,\n    [-1.2; 1.0],\n    x -> [x[1]^2 + x[2]^2],\n    [-Inf],\n    [1.0]\n)\n\noutput = percival(nlp)\nprintln(output)","category":"page"}]
}