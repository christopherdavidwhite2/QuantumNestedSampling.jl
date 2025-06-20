export energy
export energy_gradient
export fast_adjoint_mul!
export Constraint
export SubspaceConstraint
export apply_cx_gradient
export energy_constraint
export quart_constraint
export effective
export quart 
export quart_expanded
export fast_effective
export fast_quart

# I've chosen not to store the constraint *value* here with the constraint *function*.
# Reason is that that's somehow conceptually separate: it's going to change a whole bunch.
# But! That's a bug waiting to happen!
# If I'm keeping track of the list of constraint values seprately from the constraint functions,
# with the link provided only by the list index,
# I'm liable to accidentally swap the order.

struct Constraint
    constraint_function :: Function
    gradient_function   :: Function
    subspace_constraint_function :: Function
    data                :: Any
    subspace_data_reduction :: Function
end

struct SubspaceConstraint
    subspace_constraint_function :: Function
    subspace_data                :: Any   
end

effective(H, u,v) = SMatrix{2,2}([u'*H*u u'*H*v ; v'*H*u v'*H*v])

"""
    Quickly compute and overwrite
        c = α A' b + β C .
    Problem I'm trying to solve here is that some datastructures have faster left multiplication and some have faster right multiplication:
     - `Array{2}`: `mul!(s',H,u')` is much faster than `mul(s,H,u)` because Julia's array memory layout makes columns contiguous
     - SparseMatrixCSR is much faster to the right, and indeed ThreadedSparseCSR doesn't define `mul!()` with the adjoints that make Array{2} convenient.
    So I need something that Does The Right Thing.

    Question of adjoints is delicate.
    I'm choosing to make this function Weird But Correct,
    in the sense that it really does multiply by the adjoint.
    That's a balance of two considerations:
    1. In the applications I anticipate, A is Hermitian so it doesn't actually matter & multiplying by adjoint is simpler, but
    2. I don't want to sweep that fact too far under the rug.

    It's tempting to go ahead and define this for multithreaded sparse matrix multiplication here, but the state of that in Julia is parlous.
    ThreadedSparseArrays.jl (https://github.com/jagot/ThreadedSparseArrays.jl) appeared not to give much speedup;
    ThreadedSparseCSR.jl (https://github.com/BacAmorim/ThreadedSparseCSR.jl) did, but because it's unmaintained, getting it to install & compile is not so easy.
    (I plan to go with ThreadedSparseCSR for my use.)
"""
fast_adjoint_mul!(c :: Vector, A :: AbstractMatrix, b :: Vector, α, β) = mul!(c', b',A,α',β')
fast_adjoint_mul!(c :: Vector, A :: AbstractMatrix, b :: Vector)  = fast_adjoint_mul!(c, A,b,1,0)

#assuming hermiticity would save me one inner product, but that doesn't seem worth it
function fast_effective(H,u,v,s)
    fast_adjoint_mul!(s,H,u) # mul!(s',u',H)
    uHu = u'*s
    vHu = v'*s
    fast_adjoint_mul!(s,H,v) # mul!(s',v',H)
    uHv = u'*s
    vHv = v'*s
    return SMatrix{2,2}(uHu, vHu, uHv,vHv)
end

# apply Wirtinger derivative of f : C^n \to R as dual of a gradient
# allocs
# for testing
# cf notes quantum chemistry 2 2025-02-15
function apply_cx_gradient(grad, h)
    2*real(grad'*h)
end

energy(H,u) = real(u'*H*u)
    
function energy(H) 
    dim = size(H,1)
    Hu = zeros(ComplexF64, dim)
    function energy_inner(u)
        fast_adjoint_mul!(Hu,H,u)
        E = u'*Hu |> real
        return E 
    end
    return energy_inner
end

function energy_gradient(H)
    energy_gradient_inner!(grad, u) = fast_adjoint_mul!(grad, H, u)
    return energy_gradient_inner!
end

function energy_constraint(H)
    constraint_function = energy(H)
    gradient_function = energy_gradient(H) 
    subspace_constraint_function = (data,u) -> energy(data,u) 
    data = H
    subspace_data_reduction = (H,u,v) -> effective(H,u,v) 
    Constraint(constraint_function,gradient_function, subspace_constraint_function, data,subspace_data_reduction)
end

# this not quite the kurtosis---it's missing the normalization by variance
function quart(H,u)
    @assert norm(u) ≈ 1
    E = u'*H*u 
    real(u'*(H-E*LinearAlgebra.I)^4*u)
end

# could probably make this faster by playing adjoint games
function fast_quart(H)
    dim = size(H,1)
    Hu = zeros(ComplexF64, dim)
    s  = zeros(ComplexF64, dim)
    s2 = zeros(ComplexF64, dim)

    function fast_quart_inner(u)
        fast_adjoint_mul!(Hu,H,u)
        E = u'*Hu
        HmEu!(s, H,u,E) # (H - E)u
        HmEu!(s2, H,s,E) # (H - E)^2 u
        HmEu!(s, H,s2,E) # (H - E)^3 u
        HmEu!(s2, H,s,E) # (H - E)^4 u

        return real(u'*s2)
    end
end

function gradquart(H)
    function gradquart_inner(u)
        dim = length(u)
        E = u'*H*u * LinearAlgebra.I(dim)
        HmE = H - E
        return  ( 2*(u'*HmE^4 - (4 *u'*HmE^3*u) * u'*H) )'
    end
    return gradquart_inner
end

function HmEu!(s, H,u,E)
    s .= u
    fast_adjoint_mul!(s,H,u,1,-E)
end

function fast_gradquart(H)
    dim = size(H,1)
    s     = zeros(ComplexF64,dim)
    s2    = zeros(ComplexF64,dim)
    Hu    = zeros(ComplexF64,dim)
    HmE3u = zeros(ComplexF64,dim)

    function gradquart_inner!(HmE4u, u)
        fast_adjoint_mul!(Hu,H,u)
        E = u'*Hu

        HmEu!(s,    H,u,    E) # (H-E)u
        HmEu!(s2,   H,s,    E) # (H-E)^2 u
        HmEu!(HmE3u,H,s2,   E) # (H-E)^3 u
        HmEu!(HmE4u,H,HmE3u,E) # (H-E)^4 u

        uHmE3u_4 = (4*(u'*HmE3u)) :: ComplexF64
        Hu .*= uHmE3u_4

        HmE4u .-= Hu
        return HmE4u
    end
    return gradquart_inner!
end

function quart_expanded(Hpowers, u)
    H, H2, H3, H4 = Hpowers
    E = u'*H*u
    return real(u'*H4*u - 4*u'*H3*u*E + 6*u'*H2*u*E^2 - 4*u'*H*u*E^3 + E^4)
end

function quart_constraint(H)
    dim = size(H,1)
    s   = Array{ComplexF64}(undef, dim)
    # OK this is really playing with fire 
    # I'm assuming the list comprehension using fast_effective doesn't get parallelized
    # because all the calls are using the same scratch space s
    Constraint(fast_quart(H),fast_gradquart(H), quart_expanded, [H,H^2,H^3,H^4], (data,u,v) -> [fast_effective(Hn, u,v,s) for Hn in data])
end