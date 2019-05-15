if Sys.KERNEL == :Darwin
	include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end

mu = [0.5, 0.4, 0.3, 0.2]
dist = "Bernoulli"

# x = BestArm.inverse(0.01, 2, mu, dist, 0.5, 1e-11)
x = BestArm.target(0, mu, dist, 0.5, 1e-11)

print(x)
