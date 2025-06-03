ENV["GKSwstype"] = "100"
using Documenter, Percival

pages = ["Introduction" => "index.md", "Benchmark" => "benchmark.md", "Reference" => "reference.md"]

makedocs(
  sitename = "Percival.jl",
  format = Documenter.HTML(ansicolor = true, prettyurls = get(ENV, "CI", nothing) == "true"),
  modules = [Percival],
  pages = pages,
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/Percival.jl.git",
  push_preview = true,
  devbranch = "main",
)
