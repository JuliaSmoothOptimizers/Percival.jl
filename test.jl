using NLPModels, LinearAlgebra, LinearOperators
using Krylov
using JSOSolvers

include("AugLagModel.jl")
include("al.jl")

using SolverTools
using CUTEst
using NLPModelsIpopt

pnames = CUTEst.select(min_var = 5, max_var = 10, min_con = 5, max_con = 10, only_equ_con = true)
problems = (CUTEstModel(p) for p in pnames)

solvers = Dict(:auglag => al, :ipopt => ipopt)
stats = bmark_solvers(solvers, problems)

using SolverBenchmark
#markdown_table(stdout,stats[:auglag],cols = [:name,:status,:objective,:elapsed_time])
#markdown_table(stdout,stats[:ipopt],cols = [:name,:status,:objective,:elapsed_time])

using Plots
plotly()
p = performance_profile(stats, df->df.elapsed_time)

open("table_ipopt.tex","w") do io
    latex_table(io,stats[:ipopt],cols = [:name,:status,:objective,:elapsed_time,:iter])
end
open("table_AugLag.tex","w") do io
    latex_table(io,stats[:auglag],cols = [:name,:status,:objective,:elapsed_time,:iter])
end
