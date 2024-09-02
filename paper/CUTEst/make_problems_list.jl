using Pkg
path = dirname(@__FILE__)
Pkg.activate(path)
using CUTEst

nmax = 10
_pnames = CUTEst.select(
  max_var = nmax,
  min_con = 1,
  max_con = nmax,
  only_free_var = true,
  objtype = 3:6,
)

#Remove all the problems ending by NE as Ipopt cannot handle them.
pnamesNE = _pnames[findall(x -> occursin(r"NE\b", x), _pnames)]
pnames = setdiff(_pnames, pnamesNE)

open("list_problems_$nmax.dat", "w") do io
  for name in pnames
    write(io, name * "\n")
  end
end
