using Documenter, Percival

makedocs(
  sitename = "Percival.jl",
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)

deploydocs(repo = "github.com/JuliaSmoothOptimizers/Percival.jl.git", devbranch = "main")
