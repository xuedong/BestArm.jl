using PyPlot
using ProgressMeter
using HDF5
using Distributed

addprocs(3)
if Sys.KERNEL == :Darwin
	@everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	@everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end
@everywhere using DistributedArrays

# settings = ["setting0", "setting1", "setting2", "setting3", "setting4", "setting5", "setting6", "setting7", "setting8"]
# policies = [uniform, ucbe, succ_reject, seq_halving_ref, ttts, ts, at_lucb]
# names = ["Uniform Sampling", "UCB-E", "Successive Reject", "Sequential Halving with Refresh", "Top-Two Thompson Sampling", "Thompson Sampling", "AT-LUCB"]
# abrevs = ["uniform", "ucbe", "succ_reject", "seq_halving_ref", "ttts", "ts", "at_lucb"]
settings = ["setting0", "setting1", "setting2", "setting3", "setting4", "setting5", "setting6", "setting7"]
dist = "Bernoulli"
# mus = [[0.5, 0.4, 0.35, 0.3],
# 		[0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
# 		[0.5, 0.42, 0.42, 0.42, 0.42, 0.42, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38],
# 		[0.5, 0.3631, 0.449347, 0.48125839],
# 		[0.5, 0.42, 0.4, 0.4, 0.35, 0.35],
# 		[0.5, 0.45, 0.425, 0.4, 0.375, 0.35, 0.325, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125],
# 		[0.5, 0.48, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37],
# 		[0.5, 0.45, 0.45, 0.45, 0.45, 0.45, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38]]
random_lengths = rand(3:10, 100)
len = length(random_lengths)
mus = [sort(rand(random_lengths[i]), rev=true) for i in 1:length(random_lengths)]
#budgets = [5000, 10000, 10000, 10000, 3000, 20000, 30000, 30000]
budgets = [1000 for _ in 1:len]
mcmcs = [10]

policies = [BestArm.ttts]
policy_names = ["TTTS"]
abrevs = ["ttts"]
lp = length(policies)
SAVE = false
PLOT = false

io = open("mus.txt", "w")
for i in 1:len
	for j in 1:length(mus[i])
		current = mus[i][j]
		write(io, "$current ")
	end
	write(io, "\n")
end
close(io)

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

ends = zeros(1, len)


# Tests
for i in 1:len
	# setting = settings[i]
	mu = mus[i]
	budget = budgets[i]
	mcmc = mcmcs[1]

	if PLOT
		fig = figure()
	end
	X = 1:budget

	# running tests
	for imeth in 1:lp
		policy = policies[imeth]
		if policy_names[imeth] == "TTTS"
			hits = zeros(1, budget)
			@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
				_, _, _, recs = policy(mu, budget, dist, 0.5, true, false)
				for j in 1:budget
					if recs[j] == 1
						hits[j] += 1
					end
				end
			end
			if PLOT
				plot(X, reshape(1 .- hits/mcmc, budget, 1), linestyle="--", label=string(policy_names[imeth]))
			end
			ends[i] = -1/budget*log(1 .- hits[budget]/mcmc)
			if SAVE
				if Sys.KERNEL == :Darwin
					h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/ttts/", setting, "_hits.h5"), "w") do file
				    	write(file, abrevs[imeth], hits)
					end
				elseif Sys.KERNEL == :Linux
					h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/ttts/", setting, "_hits.h5"), "w") do file
			    		write(file, abrevs[imeth], hits)
					end
				end
			end
		end

		# subplot(121)
		# plot(X, transpose(regrets/mcmc), label = names[imeth])
		# subplot(122)
		# plot(log10.(X), -log10.(transpose(regrets/mcmc) ./ X), label = names[imeth])
	end

	if PLOT
		xlabel("Allocation budget")
		ylabel("Expectation of the simple regret")
		grid("on")
		legend(loc=2)
		if Sys.KERNEL == :Darwin
			savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/results/ttts/", setting, ".pdf"))
		elseif Sys.KERNEL == :Linux
			savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/results/ttts/", setting, ".pdf"))
		end
		close(fig)
	end
end


io = open("ends.txt", "w")
for i in 1:length(ends)
	current = ends[i]
	write(io, "$current\n")
end
close(io)
