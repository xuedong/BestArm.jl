if Sys.KERNEL == :Darwin
	include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end

# mus = [[0.5, 0.4, 0.35, 0.3],
# 		[0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
# 		[0.5, 0.42, 0.42, 0.42, 0.42, 0.42, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38],
# 		[0.5, 0.48125839, 0.3631, 0.449347],
# 		[0.5, 0.42, 0.4, 0.4, 0.35, 0.35],
# 		[0.5, 0.45, 0.425, 0.4, 0.375, 0.35, 0.325, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125],
# 		[0.5, 0.48, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37],
# 		[0.5, 0.45, 0.45, 0.45, 0.45, 0.45, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38]]
random_lengths = rand(3:10, 100)
mus = [sort(rand(random_lengths[i]), rev=true) for i in 1:length(random_lengths)]
dist = "Bernoulli"
len = length(mus)

# x = BestArm.inverse(0.01, 2, mu, dist, 0.5, 1e-11)
# x = BestArm.target(0, mu, dist, 0.5, 1e-11)
gammas = zeros(1, len)
for i in 1:len
	gamma = BestArm.gamma_beta(mus[i], dist, 0.5, 1e-11)
	gammas[i] = gamma
end

io = open("gammas.txt", "w")
for i in 1:length(gammas)
	gamma = gammas[i]
	write(io, "$gamma\n")
end
close(io)
