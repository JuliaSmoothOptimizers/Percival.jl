using NLPModels, LinearAlgebra, LinearOperators
using Krylov
using JSOSolvers

include("AugLagModel.jl")
include("al.jl")

using SolverTools
using CUTEst
using NLPModelsIpopt

pnames = CUTEst.select(max_var=2, max_con=2, only_equ_con=true)
problems = (CUTEstModel(p) for p in pnames[1:5])

#stats = solve_problems(al, [nlp])

solvers = Dict(:AugLag => al, :ipopt => ipopt)
stats = bmark_solvers(solvers, problems)

using SolverBenchmark
markdown_table(stdout,stats[:ipopt],cols = [:name,:status,:objective,:elapsed_time])
markdown_table(stdout,stats[:AugLag],cols = [:name,:status,:objective,:elapsed_time])
