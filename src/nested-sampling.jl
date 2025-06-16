export nested_sampling
export Xlinstep, Xlogstep, partition, heatcapacity, energy, energy_variance

function nested_sampling(EC :: Constraint, cutoffs :: Array{Float64}, θ :: Vector, resampler :: Function;steps :: Int=Int(1e3),verbose=false )
    nlive = length(θ)
    E = EC.constraint_function.(θ)
    Estar = zeros(steps)
    θstar = Array{eltype(θ)}(undef, steps)

    @showprogress for i = 1:steps
        jstar = argmax(E)
        Estar[i] = E[jstar]
        θstar[i] = θ[jstar]

        if verbose @show i,Estar[i]; flush(stdout) end

        new_start = rand( setdiff(1:nlive, jstar) )
        θ[jstar] = resampler(Estar[i:i] ∪ cutoffs, θ[new_start] ) 
        E[jstar] = EC.constraint_function(θ[jstar])
    end

    return Estar, θstar, θ
end

function nested_sampling(H; nlive = 32, steps = 100, resampler_kwargs...)
    dim = size(H,1)
    @assert size(H,1) == size(H,2)
    cutoffs = Float64[]
    EC = energy_constraint(H)

    θ = [sample_haar(dim) for _ = 1:nlive]
    resampler = gmc(constraints = [EC];resampler_kwargs...)
    return nested_sampling(EC,cutoffs,θ,resampler;steps)
end

function nested_sampling(H, Emin :: Float64;  nlive = 32, quiet=true, resampler_kwargs...)
    dim = size(H,1)
    @assert size(H,1) == size(H,2)
    cutoffs = Float64[]
    EC = energy_constraint(H)

    θ = [sample_haar(dim) for _ = 1:nlive]
    resampler = gmc(constraints = [EC];resampler_kwargs...)
    Estar = Float64[]
    θstar = Array{Complex{Float64}}[]
    while true
        batch_Estar, batch_θstar, θ = nested_sampling(EC,cutoffs,θ,resampler;steps=100*nlive)
        Estar = vcat(Estar, batch_Estar)
        θstar = vcat(θstar, batch_θstar)
        if !quiet
            @show batch_Estar[end]
        end

        if batch_Estar[end] <= Emin
            return Estar, θstar, θ
        end

    end
end


#############################################################################
############## Postprocessing 

# `t`` of the review
Xlogstep(nlive) = 1 - 1/nlive
# `w` of the review
function Xlinstep(nlive,i)
    t = Xlogstep(nlive)
    return t^(i-1)*(1- t^2)/2
end

function partition(Estar, nlive)
    w = Xlinstep.(nlive, 1:length(Estar))
    β -> sum(w .* exp.(-β*Estar))
end

function energy(Estar , Z :: Function, nlive :: Int)
    w = Xlinstep.(nlive, 1:length(Estar))
    β -> sum(w .* Estar.*exp.(-β*Estar)) / Z(β)
end

function heatcapacity(Estar , Z :: Function, E :: Function, nlive :: Int)
    w = Xlinstep.(nlive, 1:length(Estar))
    β -> β^2*( (sum(w .* Estar.^2 .*exp.(-β*Estar)) / Z(β)) .- E(β)^2)
end

function energy_variance(Estar , Z :: Function, E :: Function, nlive :: Int)
    w = Xlinstep.(nlive, 1:length(Estar))
    β -> ( (sum(w .* Estar.^2 .*exp.(-β*Estar)) / Z(β)) .- E(β)^2)
end



#=
function energyfunction(Estar, nlive)
    logZ = logpartition(Estar, nlive)
    return β -> -logZ'(β)
end
=#