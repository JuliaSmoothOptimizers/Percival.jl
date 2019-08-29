using AugLag

using Test, Logging

using NLPModels, SolverTools
using CUTEst

function test()

    @testset "cutest small problems" begin

        pnames = CUTEst.select(max_var = 2, min_con = 1, max_con = 2, only_equ_con = true)

        for i = 1:length(pnames)
            nlp = CUTEstModel(pnames[i])
            output = with_logger(NullLogger()) do
                al(nlp)
            end
            finalize(nlp)
            @test output.status == :first_order
        end

        #nlp = ADNLPModel(x->(x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2, [-1.2; 1.0],
        #        lvar = [0.0; 0.0], uvar = [1.0; 1.0], c = x->[x[1] + x[2] - 1], lcon = [0.0], ucon = [0.0])
        #output = with_logger(NullLogger()) do
        #    al(nlp)
        #end
        #@test output.status == :first_order
        #finalize(nlp)
    end
end

test()
