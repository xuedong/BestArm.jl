push!(LOAD_PATH, "../src/")
using Documenter, BestArm

makedocs(
    modules = [BestArm],
    format = :html,
    sitename = "BestArm.jl",
    pages = Any[
        "BestArm" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/xuedong/BestArm.jl",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
