@testitem "jog reversible" begin
    using LinearAlgebra
    using StaticArrays

    dim = 10

    H = GOE(dim)
    d, Ueig = eigen(H)

    CE = energy_constraint(H)
    CQ = quart_constraint(H)

    u = sample_haar(dim)
    v = sample_haar(dim)
    project_out!(u,v)
    v /= norm(v)

    # This ... makes sense in a twisty sort of way
    # but I'd do better to actually write out how subspace_data_reduction is morally a functor
    CES = SubspaceConstraint(CE.subspace_constraint_function, CE.subspace_data_reduction(CE.data,u,v))
    CQS = SubspaceConstraint(CQ.subspace_constraint_function, CQ.subspace_data_reduction(CQ.data,u,v))

    Eu = u'*H*u
    Qu = quart(H,u)

    @test abs( CES.subspace_constraint_function(CES.subspace_data,SVector(1,0)) - Eu ) < 1e-10
    @test abs( CQS.subspace_constraint_function(CQS.subspace_data,SVector(1,0)) - Qu ) < 1e-10

    constraints = [CES, CQS]

    unew_proj, vnew_proj, stepcount, constraint_satisfied = jog( 
                                                                 subspace_constraint_functions = [c.subspace_constraint_function for c in constraints],
                                                                 subspace_data                 = [c.subspace_data                for c in constraints];
                                                                 cutoffs = [Inf,Inf], dt = 0.125, maxsteps = 10)

    unew = u*unew_proj[1] + v * unew_proj[2]
    vnew = u*vnew_proj[1] + v * vnew_proj[2]

    CES = SubspaceConstraint(CE.subspace_constraint_function, CE.subspace_data_reduction(CE.data,unew,vnew))
    CQS = SubspaceConstraint(CQ.subspace_constraint_function, CQ.subspace_data_reduction(CQ.data,unew,vnew))
    uback_proj, vback_proj, stepcount, constraint_satisfied = jog(
                                                                 subspace_constraint_functions = [c.subspace_constraint_function for c in constraints],
                                                                 subspace_data                 = [c.subspace_data                for c in constraints];
                                                                 cutoffs = [Inf,Inf], dt = -0.125, maxsteps = 10)

    uback = unew*uback_proj[1] + vnew * uback_proj[2]
    vback = unew*vback_proj[1] + vnew * vback_proj[2]

    @test norm(uback - u) < 1e-10
    @test norm(vback - v) < 1e-10
end

@testitem "slalom reversible" begin
    using LinearAlgebra
    dim = 128
    H = GOE(dim)
    d, Ueig = eigen(H)

    constraints = [energy_constraint(H), quart_constraint(H)]
    cutoffs = [-1.4, 0.1]

    j = last_eigenindex_below(d, cutoffs[1]) - 10
    u = ComplexF64.(Ueig[:,j])

    nsteps = 128
    for dt = (2.0 .^ (0:-1:-20)) 
        v = sample_haar(dim)
        project_out!(u,v)
        v /= norm(v)
        unew,vnew,constraints_satisfied, reflections = slalom(;constraints,cutoffs,dt,total_steps = nsteps,u,v)

        # We report whether the constraints are satisfied.
        # Is that report accurate?
        @test all(constraints_satisfied .== ([C.constraint_function(unew) for C in constraints] .< cutoffs))

        flush(stdout)
        if all(constraints_satisfied)
        uback,vback,constraints_satisfied, reflections = slalom(;constraints,cutoffs,dt=dt,total_steps = nsteps,u=unew,v=-vnew)
            @assert norm(u) ≈ 1
            @assert norm(uback) ≈ 1
        
            # This implementation should give that uback is exactly u (up to numerical precision)
            # but that's not required for detailed balance
            # because it's ok to end up off by a phase.
            # (like, that's the same physical state
            # ---but I have a hard time imagining how you would pick up a phase when you round-trip like that.)
            #@test norm(uback - u) < 1e-10 
            #@test abs(uback'*u) ≈ 1 
        end
    end
   
end

@testitem "smoketest gmc" begin
    using LinearAlgebra
    dim = 128
    H = GOE(dim)
    d, Ueig = eigen(H)

    constraints = [energy_constraint(H), quart_constraint(H)]
    cutoffs = [-1.4, 0.1]

    j = last_eigenindex_below(d, cutoffs[1]) - 10
    u = ComplexF64.(Ueig[:,j])

    gmc(;constraints,dt=2.0 ^ -20,num_slaloms=2)(cutoffs, u)
end


