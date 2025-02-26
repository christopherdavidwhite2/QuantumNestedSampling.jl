export jog
export slalom
export gmc

struct EffortLimitException <: Exception end 

function jog(; constraints :: Array{SubspaceConstraint}, cutoffs :: Array{<:Real}, dt, maxsteps)
    Nconstraint = length(constraints)
    @cassert length(cutoffs) == Nconstraint
    @cassert dt < π

    local x
    local t = dt
    local stepcount = 0
    local constraint_satisfied = Vector{Bool}(undef, Nconstraint)

    x = MVector{2,Float64}([1;0])

    cc = zip(constraints, cutoffs)
    while true
        t :: Float64
        x = SVector(cos(t),sin(t))
        stepcount += 1

        for j = 1:Nconstraint
            c = constraints[j]
            cutoff = cutoffs[j]
            constraint_satisfied[j] = c.subspace_constraint_function(c.subspace_data, x) < cutoff 
            #@show c.subspace_constraint_function(c.subspace_data, x), constraint_satisfied[j]
        end
        
        # should probably just turn this into a normal while loop...
        if !all(constraint_satisfied) break end
        if stepcount >= maxsteps      break end
        if abs(t) > π - dt                break end

        t += dt
    end

    unew = SVector(cos(t), sin(t))
    v_parallel_transport = SVector{2}(-sin(t), cos(t))

    @cassert abs(v_parallel_transport'*unew) < 1e-10

    return unew, v_parallel_transport, stepcount, constraint_satisfied
end

# H,u,v,dt,cutoff,total_steps
function slalom(;constraints :: Array{Constraint},
                 cutoffs     :: Array{<:Real},
                 dt,total_steps,
                 u :: Array{ComplexF64},
                 v :: Array{ComplexF64},
                 )

    @cassert all([C.constraint_function(u) for C in constraints] .< cutoffs)

    # It's tempting to just *not* copy
    # but that would be bad behavior
    # worth the allocation to be polite to my future self 
    # (this allocation is not so far in: second-from-outermost loop)
    # (it is not remotely a big deal)
    u = copy(u) 
    v = copy(v)

    dim = length(u)
    @cassert length(v) == dim
    @cassert abs(u'*v) < 1e-10
    @cassert norm(u) ≈ 1
    @cassert norm(v) ≈ 1

    stepcount = 0

    local constraints_satisfied

    local unew  = zeros(ComplexF64, dim)
    local vnew  = zeros(ComplexF64, dim)
    local grad  = zeros(ComplexF64, dim)
    local uRI   = zeros(Float64,2*dim)
    local vRI   = zeros(Float64,2*dim)

    local reflections = 0
    local already_reflected_once = false

    while stepcount < total_steps
        @cassert abs(u'*v) < 1e-10

        # tighter tolerance here
        # because if I don't force this get unstable
        # vide infra
        @cassert abs(norm(u) - 1) < 1e-14 
        @cassert norm(v) ≈ 1

        # Are all these constructors a performance problem?
        # This much abstraction is actually kind of a bad UI decision
        # but it's also maybe bad for performance
        # My guess is it's ok: just, like an extra pointer dereference or something
        # so not bad compert to all the gemv's
        # but maybe it'll get in the compiler's way?
        subspace_constraints = [SubspaceConstraint(C.subspace_constraint_function, C.subspace_data_reduction(C.data,u,v)) for C in constraints]
        u2e, v2e, jog_stepcount, constraints_satisfied = jog(constraints = subspace_constraints; cutoffs, maxsteps = total_steps - stepcount, dt)

        @. unew = u2e[1]*u + u2e[2]*v
        @. vnew = v2e[1]*u + v2e[2]*v

        @. u = unew
        @. v = vnew

        # this is required for numerical stability.
        # The fact that I need to do this makes me a bit nervous.
        # What am I missing? Is there another problem hiding here? Is there a better fix?
        normalize!(u)
        normalize!(v)

        stepcount += jog_stepcount
        if stepcount >= total_steps break end

        if !all(constraints_satisfied)
            if already_reflected_once break end

            already_reflected_once = true
            reflections += 1
        else
            println("sat: $constraints_satisfied")
            already_reflected_once = false
        end

        for C in constraints[(!).(constraints_satisfied)]
            C.gradient_function(grad,u)
            project_out!(u,grad)

            if norm(grad) < 1e-5 error("gradquart too close to u for numerical stability") end

            @cassert abs(v'*u) < 1e-10
            @cassert abs(grad'*u) < 1e-10
            v = cx_gradient_householder_reflection!(grad, v,uRI,vRI)
            @cassert norm(v) ≈ 1
            @cassert abs(v'*u) < 1e-10
        end

        @cassert abs(v'*u) < 1e-10
        @cassert stepcount <= total_steps
        @cassert norm(v) ≈ 1
    end

    @cassert abs(v'*u) < 1e-10
    return u, v, constraints_satisfied, reflections
end

function gmc(;
    constraints :: Array{Constraint},
    cutoffs :: Array{<:Real},
    dt      = 2.0 ^ -10,
    num_slaloms = 8,
    stepcount_per_slalom = 128,
    effort_limit = 10^5
    )

    function gmc_step(u)
        u = ComplexF64.(u)
        dim = length(u)
        total_attempt_count = 0
        for slalom_count = 1:num_slaloms
            attempt_count = 0
            while true

                v = sample_haar(dim)
                @assert norm(u) ≈ 1
                project_out!(u,v)
                @assert abs(u'*v) < 1e-10
                v /= norm(v)

                uproposed,vproposed, constraints_satisfied, reflections = slalom(;constraints,cutoffs,dt,total_steps=stepcount_per_slalom,u,v,)

                attempt_count += 1
                total_attempt_count += 1

                if all(constraints_satisfied)
                    u = uproposed
                    break
                end

                if attempt_count > effort_limit
                    throw(EffortLimitException())
                end
            end
        end
        return u
    end
    return gmc_step
end
