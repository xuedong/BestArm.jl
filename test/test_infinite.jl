using PyPlot
using ProgressMeter
using Distributed
using HDF5

addprocs(3)
if Sys.KERNEL == :Darwin
	@everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	@everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end
# @everywhere using BestArm
@everywhere using DistributedArrays

# Problem setting
reservoir = "ShiftedBeta"
dist = "Bernoulli"
# alphas = [0.5, 1.0, 2.0, 3.0, 1.0]
# betas = [0.5, 1.0, 2.0, 1.0, 3.0]
alphas = [0.5]
betas = [0.5]
num = 1
nums_ttts = [81, 80, 78, 73, 54]
# 81, 80, 78, 73, 54
budget = 160
mcmc = 100
maxmus = [0.2, 0.4, 0.6, 0.8, 1.0]
limit = budget
shifts = [0.2, 0.4, 0.6, 0.8, 1.0]

# policies = [BestArm.ttts_dynamic]
# policy_names = ["Dynamic TTTS"]
# abrevs = ["dttts"]
policies = [BestArm.seq_halving_infinite, BestArm.ttts_dynamic]
policy_names = ["ISHA", "Dynamic TTTS"]
abrevs = ["isha", "dttts"]
lp = length(policies)


# Options
VERBOSE = true
SAVE = false


# Tests
for iparam in 1:length(alphas)
	for ishift in 1:5
		fig = figure()
		shift = shifts[ishift]
		maxmu = maxmus[ishift]
		num_ttts = nums_ttts[ishift]
		X = 1:budget
		for imeth in 1:lp
			policy = policies[imeth]
			if policy_names[imeth] == "ITTTS"
				regrets = zeros(1, budget)
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					_, _, _, recs, mu = policy(reservoir, num_ttts, budget, dist, 0.5, false, alphas[iparam], betas[iparam], false, shift)
					regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget, maxmu)
					regrets += regrets_current
				end
				plot(X, reshape(regrets/mcmc, budget, 1), linestyle="--", label=string(policy_names[imeth]))
				if SAVE
					h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], "_", shift, ".h5"), "w") do file
				    	write(file, abrevs[imeth], regrets)
					end
				end
			elseif policy_names[imeth] == "Dynamic TTTS"
				regrets = zeros(1, budget)
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					_, N, recs, mu = policy(reservoir, 1, limit, budget, dist, 0.5, false, alphas[iparam], betas[iparam], false, shift)

					regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget, maxmu)
					regrets += regrets_current
				end
				if SAVE
					h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], "_", shift, ".h5"), "w") do file
						write(file, abrevs[imeth], regrets)
					end
				end

				plot(X, reshape(regrets/mcmc, budget, 1), linestyle="-.", label=policy_names[imeth])
			elseif policy_names[imeth] == "SiRI"
				for beta in 1:3
					regrets = zeros(1, budget)
					@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
						_, _, _, recs, mu = policy(reservoir, budget, dist, 0.01, 1.0, beta, BestArm.mpa, alphas[iparam], betas[iparam], false)
						regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
						regrets += regrets_current
					end
					PyPlot.plot(X, reshape(regrets/mcmc, budget, 1), linestyle="-.", label=string(policy_names[imeth], beta))
				end
			else
				for i in 4:5
					regrets = zeros(1, budget)
					# num_arms = zeros(1, budget+1)
					@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
						_, N, _, recs, mu = policy(reservoir, Int(2^i), budget, dist, BestArm.eba, alphas[iparam], betas[iparam], false, shift)
						regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget, maxmu)
						regrets += regrets_current
						# for j in 1:(budget+1)
						# 	num_arms[j] += length(filter(x -> x==(j-1), N))
						# end
					end
					plot(X, reshape(regrets/mcmc, budget, 1), label=string(policy_names[imeth], 2^i))
					if SAVE
						h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], "_", shift, ".h5"), "w") do file
				    		write(file, abrevs[imeth], regrets)
						end
					end
					# num_arms /= mcmc
					# if Sys.KERNEL == :Darwin
					# 	h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], Int(2^i), "_N.h5"), "w") do file
					# 		write(file, abrevs[imeth], num_arms)
					# 	end
					# elseif Sys.KERNEL == :Linux
					# 	h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], Int(2^i), "_N.h5"), "w") do file
					# 		write(file, abrevs[imeth], num_arms)
					# 	end
					# end
				end
			end
		end

		xlabel("Allocation budget")
		ylabel("Expectation of the simple regret")
		grid("on")
		legend(loc=1)
		if Sys.KERNEL == :Darwin
			savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/test/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", shift, ".pdf"))
		elseif Sys.KERNEL == :Linux
			savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/test/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", shift, ".pdf"))
		end
		close(fig)
	end
end
