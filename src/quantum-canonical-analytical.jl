export cdf
function cdf(a :: Vector{Float64})
    N = length(a)
    function cdf_given_a(x)
        F = 0.5
        for k = 1:N 
            denominator = prod(a[j] - a[k] for j = 1:N if j != k)
            term = sign(x - a[k]) * (x - a[k])^(N-1) / denominator
            F += 0.5*term 
        end
        return F
    end
    return cdf_given_a
end