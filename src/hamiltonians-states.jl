export GOE
export sample_haar
export sample_haar_perp

function sample_haar(dim :: Int)
    ψ = randn(ComplexF64,dim) / sqrt(dim)
    ψ /= norm(ψ)
    return ψ
end

function sample_haar_perp(u :: Vector)
    u /= norm(u)

    v = u |> length |> sample_haar
    v -= u*(u'*v)
    v /= norm(v)

    return v
end

function GOE(dim :: Int) 
    H = randn(dim,dim)
    H += H'
    H .-= tr(H)/dim
    H /= sqrt(dim)
    return H
end
