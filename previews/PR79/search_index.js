var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [Percival]","category":"page"},{"location":"reference/#Percival.AugLagModel","page":"Reference","title":"Percival.AugLagModel","text":"AugLagModel(model, y, μ, x, fx, cx)\n\nGiven a model\n\nmin  f(x) quad st quad c(x) = 0 quad l  x  u\n\nthis new model represents the subproblem of the augmented Lagrangian method\n\nmin  f(x) - yᵀc(x) + tfrac12 μ c(x)^2 quad st quad l  x  u\n\nwhere y is an estimates of the Lagrange multiplier vector and μ is the penalty parameter.\n\nIn addition to keeping meta and counters as any NLPModel, an AugLagModel also stores\n\nmodel: The internal model defining f, c and the bounds,\ny: The multipliers estimate,\nμ: The penalty parameter,\nx: Reference to the last point at which the function c(x) was computed,\nfx: Reference to f(x),\ncx: Reference to c(x),\nμc_y: storage for y - μ * cx,\nstore_Jv and store_JtJv: storage used in hprod!.\n\nUse the functions update_cx!, update_y! and update_μ! to update these values.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Percival.percival-Union{Tuple{V}, Tuple{T}, Tuple{Val{:equ}, NLPModels.AbstractNLPModel{T, V}}} where {T, V}","page":"Reference","title":"Percival.percival","text":"percival(nlp)\n\nImplementation of an augmented Lagrangian method. The following keyword parameters can be passed:\n\nμ: Starting value of the penalty parameter (default: 10.0)\natol: Absolute tolerance used in dual feasibility measure (default: 1e-8)\nrtol: Relative tolerance used in dual feasibility measure (default: 1e-8)\nctol: (Absolute) tolerance used in primal feasibility measure (default: 1e-8)\nmax_iter: Maximum number of iterations (default: 1000)\nmax_time: Maximum elapsed time in seconds (default: 30.0)\nmax_eval: Maximum number of objective function evaluations (default: 100000)\nsubsolver_logger: Logger passed to tron (default: NullLogger)\ninity: Initial values of the Lagrangian multipliers\nsubsolver_kwargs: subsolver keyword arguments as a dictionary\n\n\n\n\n\n","category":"method"},{"location":"reference/#Percival.update_cx!-Union{Tuple{T}, Tuple{AugLagModel, AbstractVector{T}}} where T","page":"Reference","title":"Percival.update_cx!","text":"update_cx!(nlp, x)\n\nGiven an AugLagModel, if x != nlp.x, then updates the internal value nlp.cx calling cons on nlp.model, and reset nlp.fx to a NaN. Also updates nlp.μc_y.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Percival.update_fxcx!-Tuple{AugLagModel, AbstractVector{T} where T}","page":"Reference","title":"Percival.update_fxcx!","text":"update_fxcx!(nlp, x)\n\nGiven an AugLagModel, if x != nlp.x, then updates the internal value nlp.cx calling objcons on nlp.model. Also updates nlp.μc_y. Returns fx only.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Percival.update_y!-Tuple{AugLagModel}","page":"Reference","title":"Percival.update_y!","text":"update_y!(nlp)\n\nGiven an AugLagModel, update nlp.y = -nlp.μc_y and updates nlp.μc_y accordingly.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Percival.update_μ!-Tuple{AugLagModel, AbstractFloat}","page":"Reference","title":"Percival.update_μ!","text":"update_μ!(nlp, μ)\n\nGiven an AugLagModel, updates nlp.μ = μ and nlp.μc_y accordingly.\n\n\n\n\n\n","category":"method"},{"location":"benchmark/#Benchmarks","page":"Benchmark","title":"Benchmarks","text":"","category":"section"},{"location":"benchmark/#CUTEst-benchmark","page":"Benchmark","title":"CUTEst benchmark","text":"","category":"section"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"With a JSO-compliant solver, such as Percival, we can run the solver on a set of problems, explore the results, and compare to other JSO-compliant solvers using specialized benchmark tools.  We are following here the tutorial in SolverBenchmark.jl to run benchmarks on JSO-compliant solvers.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using CUTEst","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"To test the implementation of Percival, we use the package CUTEst.jl, which implements CUTEstModel an instance of AbstractNLPModel. ","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using SolverBenchmark","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"Let us select problems from CUTEst with a maximum of 100 variables or constraints. After removing problems with fixed variables, examples with a constant objective, and infeasibility residuals.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"_pnames = CUTEst.select(\n  max_var = 100, \n  min_con = 1, \n  max_con = 100, \n  only_free_var = true,\n  objtype = 3:6\n)\n\n#Remove all the problems ending by NE as Ipopt cannot handle them.\npnamesNE = _pnames[findall(x->occursin(r\"NE\\b\", x), _pnames)]\npnames = setdiff(_pnames, pnamesNE)\ncutest_problems = (CUTEstModel(p) for p in pnames)\n\nlength(cutest_problems) # number of problems","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"We compare here Percival with Ipopt (Wächter, A., & Biegler, L. T. (2006). On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Mathematical programming, 106(1), 25-57.), via the NLPModelsIpopt.jl thin wrapper, with Percival on a subset of CUTEst problems.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using Percival, NLPModelsIpopt","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"To make stopping conditions comparable, we set Ipopt's parameters dual_inf_tol=Inf, constr_viol_tol=Inf and compl_inf_tol=Inf to disable additional stopping conditions related to those tolerances, acceptable_iter=0 to disable the search for an acceptable point.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"#Same time limit for all the solvers\nmax_time = 1200. #20 minutes\ntol = 1e-5\n\nsolvers = Dict(\n  :ipopt => nlp -> ipopt(\n    nlp,\n    print_level = 0,\n    dual_inf_tol = Inf,\n    constr_viol_tol = Inf,\n    compl_inf_tol = Inf,\n    acceptable_iter = 0,\n    max_cpu_time = max_time,\n    x0 = nlp.meta.x0,\n    tol = tol,\n  ),\n  :percival => nlp -> percival(\n      nlp,\n      max_time = max_time,\n      max_iter = typemax(Int64),\n      max_eval = typemax(Int64),\n      atol = tol,\n      rtol = tol,\n      ctol = tol,\n  ),\n)\n\nstats = bmark_solvers(solvers, cutest_problems)","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"The function bmark_solvers return a Dict of DataFrames with detailed information on the execution. This output can be saved in a data file.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using JLD2\n@save \"ipopt_percival_$(string(length(pnames))).jld2\" stats","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"The result of the benchmark can be explored via tables,","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"pretty_stats(stats[:percival])","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"or it can also be used to make performance profiles.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using Plots\ngr()\n\nsolved(df) = (df.status .== :first_order)\ncosts = [\n  df -> .!solved(df) * Inf + df.elapsed_time,\n  df -> .!solved(df) * Inf + df.neval_obj + df.neval_cons,\n]\ncostnames = [\"Time\", \"Evalutions of obj + cons\"]\np = profile_solvers(stats, costs, costnames)","category":"page"},{"location":"#Percival.jl","page":"Introduction","title":"Percival.jl","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"an Augmented Lagrangian method","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Pages = [\"index.md\"]","category":"page"},{"location":"#Description","page":"Introduction","title":"Description","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"This package implements a JSO-compliant augmented Lagrangian method based on the paper","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"S. Arreckx, A. Lambe, Martins, J. R. R. A., & Orban, D. (2016).\nA Matrix-Free Augmented Lagrangian Algorithm with Application to Large-Scale Structural Design Optimization.\nOptimization And Engineering, 17, 359–384. doi:10.1007/s11081-015-9287-9","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"It was implemented as part of the Master's dissertation of Egmara Antunes.","category":"page"},{"location":"#JSO-compliant-solver","page":"Introduction","title":"JSO-compliant solver","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"The percival method expects a single mandatory argument - an AbstractNLPModel - and returns a GenericExecutionStats from SolverCore.jl.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"We refer to juliasmoothoptimizers.github.io for tutorials on the NLPModel API. This framework allows the usage of models from Ampl (using AmplNLReader.jl), CUTEst (using CUTEst.jl), JuMP (using NLPModelsJuMP.jl), PDE-constrained optimization problems (using PDENLPModels.jl) and models defined with automatic differentiation (using ADNLPModels.jl).","category":"page"},{"location":"#Installation","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Percival is a registered package. To install this package, open the Julia REPL (i.e., execute the julia binary), type ] to enter package mode, and install Percival as follows","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"add Percival","category":"page"},{"location":"#Main-exported-functions-and-types","page":"Introduction","title":"Main exported functions and types","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"percival: The function to call the method. Pass an NLPModel to it.\nAugLagModel: A model representing the bound-constrained augmented Lagrangian subproblem. This is a subtype of AbstractNLPModel.","category":"page"},{"location":"#Example","page":"Introduction","title":"Example","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"How to solve the simple problem","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"min_(x_1x_2) quad (x_1 - 1)^2 + 100 (x_2 - x_1^2)^2 quad textsto quad x_1^2 + x_2^2 leq 1","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The problem is modeled using ADNLPModels.jl with [-1.2; 1.0] as default initial point, and then solved using percival.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using ADNLPModels, Percival\n\nnlp = ADNLPModel(\n    x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,\n    [-1.2; 1.0],\n    x -> [x[1]^2 + x[2]^2],\n    [-Inf],\n    [1.0]\n)\n\noutput = percival(nlp)\nprint(output)","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Introduction","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"If you want to ask a question not suited for a bug report, feel free to start a discussion here. This forum is for general discussion about this repository and the JuliaSmoothOptimizers, so questions about any of our packages are welcome.","category":"page"}]
}
