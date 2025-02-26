
@testitem "Householder" begin
    dim = 10
    for _ = 1:10
        u = randn(ComplexF64,dim)
        v = randn(ComplexF64,dim)
        ur = householder_reflection(v,u)
        @test abs(v'*ur + v'*u ) < 1e-10
    end
end

@testitem "project_out!" begin
    dim = 10
    for _ = 1:10
        u = randn(ComplexF64,dim)
        v = randn(ComplexF64,dim)
        v = project_out!(u,v)
        @test abs(v'*u) < 1e-10
    end
end