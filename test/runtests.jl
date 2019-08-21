using AugLag

using Test

using NLPModels
using SolverTools
#using CUTEst
using NLPModelsIpopt

function test()
    
    @testset "Cutest problems" begin

        # NullLogger
        #pnames = CUTEst.select(max_var = 2, max_con = 2, only_equ_con = true)
        #problems = (CUTEstModel(p) for p in pnames[41:41])

        #solvers = Dict(:auglag => al, :ipopt => ipopt)
        #stats = bmark_solvers(solvers, problems)

        @test true
    end

end

test()
