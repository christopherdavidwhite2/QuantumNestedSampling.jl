
@testitem "nested sampling" begin
    dim = 10
    nlives = 3 ∪ ( 2 .^ 2:4)
    for nlive = nlives
        for _ = 1:10
            steps = 3*nlive
            H = GOE(dim)
            Estar, θstar = nested_sampling(H; steps, nlive)
            @assert all( ( Estar[1:end-1] - Estar[2:end] ) .> 0 )
        end
    end
end