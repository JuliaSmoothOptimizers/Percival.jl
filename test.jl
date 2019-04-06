using NLPModels, LinearAlgebra, LinearOperators
using Krylov
using JSOSolvers

include("AugLagModel.jl")
include("al.jl")

using SolverTools
using CUTEst

pnames = CUTEst.select(max_var=2, max_con=2, only_equ_con=true)
problems = (CUTEstModel(p) for p in pnames)

stats = solve_problems(al, problems) # ta na pasta bmark
