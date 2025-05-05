export GOE,GREM,ising_1d,make_hamiltonian
export sample_haar, sample_haar!
export sample_haar_perp, sample_haar_perp!

function sample_haar(::Type{T}, dim :: Int) where T <: Number
    ψ = randn(T,dim) / sqrt(dim)
    ψ /= norm(ψ)
    return ψ
end

function sample_haar(dim :: Int)
    ψ = randn(ComplexF64,dim) / sqrt(dim)
    ψ /= norm(ψ)
    return ψ
end

function sample_haar!(v :: Vector)
    @. v = randn(eltype(v))
    nv = norm(v)
    v ./= nv
    return nothing
end

function sample_haar_perp(u :: Vector)
    u /= norm(u)

    v = sample_haar(eltype(u), length(u))
    v -= u*(u'*v)
    v /= norm(v)

    return v
end

function sample_haar_perp!(u :: Vector{T}, v :: Vector{T}) where T <: Number
    @cassert abs(norm(u) - 1) < 1e-10
    sample_haar!(v)
    ovlp = u'*v
    @. v -= u*ovlp
    nv = norm(v)
    @. v /= nv
    return nothing
end

function GOE(dim :: Int) 
    H = randn(dim,dim)
    H += H'
    H .-= tr(H)/dim
    H /= (2*sqrt(dim))
    return H
end

function GREM(dim :: Int) :: Diagonal{Float64,Vector{Float64}}
    d = dim |> randn |> sort
    offset = mean(d)
    d .-= offset
    return Diagonal(d)
end

function GREM_od(dim :: Int) 
    d = dim |> randn |> sort
    offset = mean(d)
    d .-= offset
    return SymTridiagonal(d, fill(0.1, dim-1))
end
 
# TODO sparse
function ising_1d(dim; hz = 0.9045, hx = 1.4, pbc=true)
    Lfloat = log(dim) / log(2)
    @assert min(Lfloat % 1, abs( (Lfloat - 1) % 1)) < 1e-10
    L = Lfloat |> round |> Int
    z = [1 0; 0 -1]
    x = [0 1; 1 0]

    X = tembed(x,L)
    Z = tembed(z,L)

    H = sum(Z[j] * Z[j+1] for j = 1:L-1)
    H += sum(hz * Z[j] for j = 1:L)
    H += sum(hx * X[j] for j = 1:L)

    if pbc H += Z[1]*Z[L] end

    return H
end

# has to be a better way: this just feels dumb
function make_hamiltonian(type :: Symbol, dim :: Int)
    @assert dim >= 1
    if type == :GOE
        return GOE(dim)
    elseif type == :GREM
        return GREM(dim)
    elseif type == :GREM_od
        return GREM_od(dim)
    elseif type == :ising_1d
        return ising_1d(dim)
    else
        error("Unknown Hamiltonian type $type")
    end
end