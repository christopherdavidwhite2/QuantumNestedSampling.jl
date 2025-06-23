@testitem "int types" begin
    @test QuantumNestedSampling.uint_type_for_max(255) == UInt8
    @test QuantumNestedSampling.uint_type_for_max(256) == UInt16
end

@testitem "sanity checking" begin
    L = 8
    for r = 1:10
        a = rand(0:2^L - 1)
        @test a == evalpoly(2, digits(a, base=2, pad=L))
    end
end