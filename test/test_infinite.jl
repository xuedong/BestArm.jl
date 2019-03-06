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
# reservoir = "ShiftedBeta"
reservoir = "ShiftedBeta"
dist = "Bernoulli"
alphas = [1.0, 2.0, 3.0, 1.0]
betas = [1.0, 2.0, 1.0, 3.0]
# alphas = [1.0, 3.0, 1.0, 0.5, 2.0, 5.0, 2.0, 0.3]
# betas = [1.0, 1.0, 3.0, 0.5, 5.0, 2.0, 2.0, 0.7]
num = 16
budget = 64
mcmc = 10
maxmu = 0.5
default = true

# policies = [BestArm.siri]
# policy_names = ["SiRI"]
# abrevs = ["siri"]
policies = [BestArm.seq_halving_infinite, BestArm.ttts_infinite, BestArm.ttts_dynamic]
policy_names = ["ISHA", "TTTS", "Dynamic TTTS"]
abrevs = ["isha", "ttts", "dttts"]
lp = length(policies)


# Options
VERBOSE = true
SAVE = false


# Tests
for iparam in 1:4
	fig = figure()
	X = 1:budget
	for imeth in 1:lp
		policy = policies[imeth]
		if policy_names[imeth] == "TTTS"
		  	#regrets_array = @DArray [BestArm.parallel_ttts(mu, budget, dist) for i = 1:mcmc]
			#for i in 1:mcmc
			#	regrets += regrets_array[i]
			#end
			for i in 4:5
				regrets = zeros(1, budget)
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					_, _, _, recs, mu = policy(reservoir, Int(2^i), budget, dist, 0.5, true, alphas[iparam], betas[iparam], false)
					regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget, maxmu)
					regrets += regrets_current
				end
				plot(X, reshape(regrets/mcmc, budget, 1), linestyle="--", label=string(policy_names[imeth], 2^i))
				if SAVE
					h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], Int(2^i), ".h5"), "w") do file
			    		write(file, abrevs[imeth], regrets)
					end
				end
			end
		elseif policy_names[imeth] == "Dynamic TTTS"
			regrets = zeros(1, budget)
			@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
				_, _, recs, mu = policy(reservoir, 1, num, budget, dist, 0.5, true, alphas[iparam], betas[iparam], false)
				regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget, maxmu)
				regrets += regrets_current
				if SAVE
					h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], "_mpa.h5"), "w") do file
				    	write(file, abrevs[imeth], regrets)
					end
				end
			end
			plot(X, reshape(regrets/mcmc, budget, 1), linestyle="-.", label=string(policy_names[imeth], " (MPA)"))
			if default
				regrets = zeros(1, budget)
				num_arms = zeros(1, budget+1)
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					_, N, recs, mu = policy(reservoir, 1, budget, budget, dist, 0.5, false, alphas[iparam], betas[iparam], false)
					# print(N)
					# num_arms1[k] = length(filter(x -> x>0, N))
					# num_arms2[k] = length(filter(x -> x>1, N))
					print(length(N))
					print(N)
					regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget, maxmu)
					regrets += regrets_current
					for j in 1:(budget+1)
						num_arms[j] += length(filter(x -> x==(j-1), N))
					end
					if SAVE
						h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], ".h5"), "w") do file
							write(file, abrevs[imeth], regrets)
						end
					end
				end
				num_arms /= mcmc
				print(num_arms)
				if Sys.KERNEL == :Darwin
					h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], "_N.h5"), "w") do file
						write(file, abrevs[imeth], num_arms)
					end
				elseif Sys.KERNEL == :Linux
					h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], "_N.h5"), "w") do file
						write(file, abrevs[imeth], num_arms)
					end
				end
				# print(num_arms1)
				# print(num_arms2)
				# h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")_arms.h5"), "w") do file
				# 	h5write(file, num_arms)
				# end
				plot(X, reshape(regrets/mcmc, budget, 1), linestyle="-.", label=policy_names[imeth])
			end
		elseif policy_names[imeth] == "SiRI"
			for beta in 1:3
				regrets = zeros(1, budget)
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					_, _, _, recs, mu = policy(reservoir, budget, dist, 0.01, 1.0, beta, BestArm.mpa, alphas[iparam], betas[iparam], false)
					regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
					regrets += regrets_current
				end
				plot(X, reshape(regrets/mcmc, budget, 1), linestyle="-.", label=string(policy_names[imeth], beta))
			end
		else
			for i in 3:4
				regrets = zeros(1, budget)
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					_, _, _, recs, mu = policy(reservoir, Int(2^i), budget, dist, BestArm.eba, alphas[iparam], betas[iparam], false)
					regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget, maxmu)
					regrets += regrets_current
				end
				plot(X, reshape(regrets/mcmc, budget, 1), label=string(policy_names[imeth], 2^i))
				if SAVE
					h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], Int(2^i), ".h5"), "w") do file
			    		write(file, abrevs[imeth], regrets)
					end
				end
			end
		end
	end

	xlabel("Allocation budget")
	ylabel("Expectation of the simple regret")
	grid("on")
	legend(loc=1)
	if Sys.KERNEL == :Darwin
		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/test/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", budget, ".pdf"))
	elseif Sys.KERNEL == :Linux
		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/test/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", budget, ".pdf"))
	end
	close(fig)
end
