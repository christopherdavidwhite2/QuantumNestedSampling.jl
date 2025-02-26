export householder_reflection
export cx_gradient_householder_reflection!
export project_out!

# Householder reflection of u about v
# Better to use 
function householder_reflection(v,u)
    v /= norm(v)
    return u - 2*v*(v'*u)
end

function cx_gradient_householder_reflection!(v :: Array{ComplexF64}, u :: Array{ComplexF64}, vRI :: Array{Float64}, uRI :: Array{Float64})
    dim = length(v)
    @cassert length(vRI) == 2*dim
    @cassert length(uRI) == 2*dim
    @cassert length(u) == dim 

    @. vRI[1:dim] = real(v)
    @. vRI[dim+1:end] = imag(v)
    @. uRI[1:dim] = real(u)
    @. uRI[dim+1:end] = imag(u)
    urefl_RI = householder_reflection(vRI,uRI)
    return urefl_RI[1:dim] + im*urefl_RI[dim+1:end]
end

function project_out!(u,v)
    ovlp = u'*v/norm(u)^2
    @. v = v - u*ovlp
end

