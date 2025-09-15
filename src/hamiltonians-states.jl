export nqubit_from_dim
export GOE,GREM,ising_1d,heisenberg_1d,paramagnet,make_hamiltonian
export heisenberg_1d_gs
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
    H -= LinearAlgebra.I * tr(H)/dim
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
function ising_1d_embed(dim; hz = 0.9045, hx = 1.4, pbc=true)
    Lfloat = log(dim) / log(2)
    @assert min(Lfloat % 1, abs( (Lfloat - 1) % 1)) < 1e-10
    L = Lfloat |> round |> Int
    z = [1 0; 0 -1] |> sparse
    x = [0 1; 1 0] |> sparse

    X = tembed(x,L)
    Z = tembed(z,L)

    H = sum(Z[j] * Z[j+1] for j = 1:L-1)
    H += sum(hz * Z[j] for j = 1:L)
    H += sum(hx * X[j] for j = 1:L)

    if pbc H += Z[1]*Z[L] end

    return H
end


function ising_1d(dim; hz = 0.9045, hx = 1.4, pbc=true)
    L = nqubit_from_dim(dim)
    @assert L >= 2

    N_entry = 2^L * (L+1)
    T = uint_type_for_max(N_entry)
    I = zeros(T, N_entry)
    J = zeros(T, N_entry)
    V = zeros(Float64, N_entry)

    ctr = 1
    for b = 0:2^L-1
        b_bits = digits(b, base=2, pad=L) .|> Bool
        σb     = 2 * b_bits .- 1

        # zz

        diag_val = 0
        for j = 1:L-1
            diag_val += σb[j]*σb[j+1]
        end

        if pbc diag_val += σb[1]*σb[L] end

        diag_val -= hz * sum(σb)

        J[ctr] = b+1
        I[ctr] = b+1
        V[ctr] = diag_val
        ctr += 1

        for j = 1:L
            a_bits = deepcopy(b_bits)
            a_bits[j] = !a_bits[j]

            a = evalpoly(2, a_bits)

            J[ctr] = a+1
            I[ctr] = b+1
            V[ctr] = hx
            ctr += 1
        end
    end

    return sparse(I,J,V)
end

function nqubit_from_dim(dim :: Real)
    Lfloat = log(dim) / log(2)
    @assert min(Lfloat % 1, abs( (Lfloat - 1) % 1)) < 1e-10
    L = Lfloat |> round |> Int
    return L
end

function heisenberg_1d(dim :: Integer;
                       hx :: Real = 0,
                       hy :: Real = 0,
                       hz :: Real = 0,
                       pbc = true)
    L = nqubit_from_dim(dim)
    return heisenberg_1d(dim :: Integer,
                         fill(hx, L),
                         fill(hy, L),
                         fill(hz, L),
                         pbc
                         )
end

function heisenberg_1d_embed(dim,
                       hx :: Array{<:Real},
                       hy :: Array{<:Real},
                       hz :: Array{<:Real},
                       pbc = true)

                       
    L = nqubit_from_dim(dim)
    @assert L == length(hx)
    z = [1 0; 0 -1] |> sparse
    x = [0 1; 1 0] |> sparse
    y = [0 -im; im 0] |> sparse

    X = tembed(x,L)
    Y = tembed(y,L)
    Z = tembed(z,L)

    H  = -sum(X[j] * X[j+1] for j = 1:L-1)
    H += -sum(Y[j] * Y[j+1] for j = 1:L-1)
    H += -sum(Z[j] * Z[j+1] for j = 1:L-1)

    H += -sum(hx[j] * X[j] for j = 1:L)
    H += -sum(hy[j] * Y[j] for j = 1:L)
    H += -sum(hz[j] * Z[j] for j = 1:L)

    if pbc
        H += -X[1]*X[L]
        H += -Y[1]*Y[L]
        H += -Z[1]*Z[L]
    end
    
    return H
end

# Not type-stable.
# If hy = 0, the resulting Hamiltonian is in fact real;
# in that case I construct it as real to start out with 
# to avoid paying the memory cost for ComplexF64
function heisenberg_1d(dim,
                       hx :: Array{<:Real},
                       hy :: Array{<:Real},
                       hz :: Array{<:Real},
                       pbc = true)
    L = nqubit_from_dim(dim)
    @assert L >= 2

    # diagonal + (x,y,z,hop) * L
    # Overestimate: doesn't take into account conservation
    # (not every state/bond has a nontrivial hop)
    N_entry = 2^L * (4*L+1)
    T = uint_type_for_max(N_entry)
    I = zeros(T, N_entry)
    J = zeros(T, N_entry)


    # return complex values if f64
    has_y_field = norm(hy) > 1e-10
    if has_y_field
        V = zeros(ComplexF64, N_entry)
    else
        V = zeros(Float64, N_entry)
    end

    ctr = 1
    for b = 0:2^L-1
        b_bits = digits(b, base=2, pad=L) .|> Bool
        σb     = 2 * b_bits .- 1

        ###### diagonal part: zz + hz*z

        diag_val = 0
        for j = 1:L-1
            diag_val -= σb[j]*σb[j+1]
        end

        if pbc diag_val -= σb[1]*σb[L] end
        diag_val += sum(reverse(hz) .* σb)

        J[ctr] = b+1
        I[ctr] = b+1
        V[ctr] = diag_val
        ctr += 1

        ###### s+s-
        for j = 1:L
            k = (j % L) + 1 # one-indexing
            if ! ( b_bits[j] && !b_bits[k] )
                continue
            end

            a_bits = deepcopy(b_bits)
            a_bits[j] = false
            a_bits[k] = true

            # sanity check: conserve σz
            @assert sum(a_bits) == sum(b_bits)

            a = evalpoly(2, a_bits)

            @assert a != b
            # hop one direction
            J[ctr] = a+1
            I[ctr] = b+1
            V[ctr] = -2
            ctr += 1

            # hop other direction
            I[ctr] = a+1
            J[ctr] = b+1
            V[ctr] = -2
            ctr += 1
        end

        ###### hx σx + hy σy
        for j = 1:L
            a_bits = deepcopy(b_bits)
            a_bits[j] = !a_bits[j]

            a = evalpoly(2, a_bits)

            @assert a != b

            J[ctr] = a+1
            I[ctr] = b+1
            if has_y_field
                V[ctr] = -hx[L-j+1] - hy[L-j+1]*im * (σb[j])
            else
                V[ctr] = -hx[L-j+1] 
            end
            ctr += 1
        end
    end

    I = I[V.!= 0]
    J = J[V.!= 0]
    V = V[V.!= 0]

    return sparse(I,J,V,2^L, 2^L)
end


""" Analytical ground state projector for L = lg(dim)-site isotropic Heisenberg.
If
    ``` gs = heisenberg_1d_gs(2^L)
    ```
then `gs` is `(2^L, L+1)` and `gs[:,j]` has σz expectation value `j - 1 - L/2`

It's tempting (for the sake of API consistency) to supply an analog for ising_1d,
but that would require pulling in Arpack. (Could/should for paramagnet, I suppose.)
"""
function heisenberg_1d_gs_embed(dim)
    L = nqubit_from_dim(dim)

    x = [0 1; 1 0] |> sparse
    y = [0 -im; im 0] |> sparse

    X = tembed(x,L)
    Y = tembed(y,L)

    totalX = sum(X)
    totalY = sum(Y)

    # deallocate when finished
    # (I hope)
    # (idk, maybe the compiler is smart enough to do this early anyways)
    # (but I want to make sure)
    x = nothing
    y = nothing
    totalSM = totalX - im*totalY

    totalX = spzeros(dim,dim); GC.gc()
    totalY = spzeros(dim,dim); GC.gc()

    gs =zeros(2^L, L+1)
    gs[1,1] = 1
    for j = 2:L+1
        gs[:,j] = totalSM * gs[:,j-1]
        gs[:,j] /= norm(gs[:,j])
    end

    return gs
end

function heisenberg_1d_gs(dim)
    L = nqubit_from_dim(dim)

    gs =zeros(2^L, L+1)
    gs[1,1] = 1
    for j = 2:L+1
        for a = 1:2^L
            if gs[a,j-1] != 0
                ψ = gs[a,j-1]
                a_bits = digits(a-1, base=2, pad=L) .|> Bool
                for i = 1:L
                    if !a_bits[i]
                        b_bits = deepcopy(a_bits)
                        b_bits[i] = !b_bits[i]
                        b = evalpoly(2, b_bits)
                        gs[b+1,j] = ψ
                    end
                end
            end
        end
        gs[:,j] /= norm(gs[:,j])
    end

    return gs
end

# slow / memory intensive: do not use
function paramagnet_embed(dim)
    L = nqubit_from_dim(dim)
    z = [1 0; 0 -1] |> sparse
    Z = tembed(z,L)
    return - sum(Z)
end


function paramagnet(dim)
    L = nqubit_from_dim(dim)

    T = uint_type_for_bits(L)
    I = 1:2^L |> Array{T}
    J = 1:2^L |> Array{T}

    @assert L <= 127 # else we might run into trouble with the int8
    V = [Int8(sum(2 * digits(a-1,base=2, pad=L) .- 1 )) for a in I]

    return sparse(I,J,V)
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
    elseif type == :paramagnet
        return paramagnet(dim)
    elseif type == :ising_1d
        return ising_1d(dim)
    elseif type == :heisenberg_1d
        return heisenberg_1d(dim)
    else
        error("Unknown Hamiltonian type $type")
    end
end