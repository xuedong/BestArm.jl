if Sys.KERNEL == :Darwin
    include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
    include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end

mu = rand(4)
dist = "Bernoulli"

print(gamma_beta(mu, dist))
