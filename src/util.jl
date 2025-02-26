export @cassert
export last_eigenindex_below

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