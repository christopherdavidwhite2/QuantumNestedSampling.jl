export @cassert
export last_eigenindex_below
export unzip

# conditional assert
# https://discourse.julialang.org/t/assert-alternatives/24775/14?u=cdwhite1
asserting() = true #making this a function results in code being invalidated and recompiled when this gets changed
macro cassert(test)
  esc(:(if $(@__MODULE__).asserting()
    @assert($test)
   end))
end

last_eigenindex_below(d,cutoff) = (@show cutoff; findlast(d .< cutoff))

function normalize!(u)
  nu = norm(u)
  u ./= nu
end

scalar(A) = (if length(A) != 1; error("scalar: argument has wrong length A = $A"); else A[1]; end)
unzip(A :: Array{Tuple{S,T}} where {T,S}) = ([a[1] for a in A], [a[2] for a in A])
unzip(A :: Array{Tuple{R,S,T}} where {R,S,T}) = ([a[1] for a in A], [a[2] for a in A], [a[3] for a in A],)

