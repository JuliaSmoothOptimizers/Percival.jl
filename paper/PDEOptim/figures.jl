using Pkg; Pkg.activate(".")
using JLD2, Plots, SolverBenchmark
using PyPlot
pyplot()

# We really need DataFrames 1.4
# https://discourse.julialang.org/t/reading-dataframes-from-jld2-files/88543
name_pdeoptim = "PDEOptim/2024-03-28_PDEOP_ipopt_percival_n_5_mu_10.0"
@load "$name_pdeoptim.jld2" stats max_time tol μ n
solved(df) = (df.status .== :first_order)

# Number of problems solved by ipopt
@show sum(solved(stats[:ipopt]))
# Number of problems solved by percival
@show sum(solved(stats[:percival]))

tim_percival = stats[:percival][solved(stats[:percival]), :elapsed_time]
tim_ipopt = stats[:ipopt][solved(stats[:percival]), :elapsed_time]
# Number of problems where Ipopt is fastest
@show sum(tim_percival .> tim_ipopt)
# Number of problems where Percival is fastest
@show sum(tim_percival .< tim_ipopt)

obj_percival = stats[:percival][solved(stats[:percival]), :neval_obj]
obj_ipopt = stats[:ipopt][solved(stats[:percival]), :neval_obj]
con_percival = stats[:percival][solved(stats[:percival]), :neval_cons]
con_ipopt = stats[:ipopt][solved(stats[:percival]), :neval_cons]

# Number of problems where Ipopt use less evaluations
@show sum((obj_percival .+ con_percival) .> (obj_ipopt .+ con_ipopt))
# Number of problems where percival use less evaluations
@show sum((obj_percival .+ con_percival) .< (obj_ipopt .+ con_ipopt))
# Number where it is a tie
@show sum((obj_percival .+ con_percival) .== (obj_ipopt .+ con_ipopt))

costs = [
  df -> .!solved(df) * Inf + df.elapsed_time,
  df -> .!solved(df) * Inf + df.neval_obj + df.neval_cons,
]
costnames = ["Time", "Evaluations of obj + cons"]
p = profile_solvers(stats, costs, costnames, width=400, height=400)
plot!(xticks = ([0.0, 5.0, 10.0, 15.0], ["2⁰", "2⁵", "2¹⁰", "2¹⁵"]))
plot!(xtickfontsize=9)
png(p, "ipopt_percival_82")
