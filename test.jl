using NLPModels, LinearAlgebra, Krylov
using Ipopt, NLPModelsIpopt

include("ALModel.jl")
#include("ipopt.jl")
include("al.jl")

x0 = [-1.2; 1.0]
lvar = [0.0; 0.0]
uvar = [1.0; 1.0]
restr(x) = [x[1] + x[2] - 1]
lcon = [0.0]
ucon = [0.0]
nlp = ADNLPModel(x->(x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2, x0, lvar=lvar, uvar=uvar, c=restr, lcon=lcon, ucon=ucon)
x, fx, normgL, normc, k = al(nlp)
