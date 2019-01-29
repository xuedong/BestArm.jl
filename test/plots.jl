using PyPlot
# using ConfParser
using HDF5
using Seaborn

# if Sys.KERNEL == :Darwin
# 	conf = ConfParse("/Users/xuedong/Programming/PhD/BestArm.jl/test/configs.ini")
# elseif Sys.KERNEL == :Linux
# 	conf = ConfParse("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/test/configs.ini")
# end
# parse_conf!(conf)

# settings = ["setting0", "setting1", "setting2", "setting3", "setting4", "setting5", "setting6", "setting7", "setting8"]
# names = ["Uniform Sampling", "UCB-E", "Successive Reject", "Sequential Halving with Refresh", "Top-Two Thompson Sampling", "Thompson Sampling", "AT-LUCB"]
# abrevs = ["uniform", "ucbe", "succ_reject", "seq_halving_ref", "ttts", "ts", "at_lucb"]

mcmc = 1000
pace = 4
lb = 64
ub = 128
budgets = [Int(round(pace*i*log2(pace*i))) for i in Int(lb/pace):Int(ub/pace)]
narms = [pace*i for i in Int(lb/pace):Int(ub/pace)]
lbudget = length(budgets)
policy_names = ["ISHA", "TTTS", "Dynamic TTTS"]
abrevs = ["isha", "ttts", "dttts"]
settings = ["Beta(0.5,0.5)", "Beta(1.0,1.0)", "Beta(1.0,3.0)", "Beta(2.0,2.0)", "Beta(3.0,1.0)"]

lp = length(policy_names)


# Plots
# for setting in settings
# 	budget = retrieve(conf, setting, "budget")
# 	budget = parse(Int, budget)
# 	mcmc = retrieve(conf, setting, "mcmc")
# 	mcmc = parse(Int, mcmc)
#
# 	fig = figure()
# 	X = 1:budget
#
# 	# running tests
# 	for imeth in 1:lp
# 		regrets = zeros(1, budget)
#
# 		regrets = h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/log/ttts/", setting, "_", abrevs[imeth], ".h5"), "r") do file
#     		read(file, abrevs[imeth])
# 		end
#
# 		plot(X, transpose(regrets/mcmc), label = names[imeth])
# 	end
#
# 	xlabel("Allocation budget")
# 	ylabel("Expectation of the simple regret")
# 	grid("on")
# 	legend(loc=1)
# 	if Sys.KERNEL == :Darwin
# 		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/results/ttts/", setting, ".png"))
# 	elseif Sys.KERNEL == :Linux
# 		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/results/ttts/", setting, ".png"))
# 	end
# 	close(fig)
# end

# for setting in settings
# 	budget = retrieve(conf, setting, "budget")
# 	budget = p@everywhere arse(Int, budget)
# 	mcmc = retrieve(conf, setting, "mcmc")
# 	mcmc = parse(Int, mcmc)
#
# 	fig = figure()
# 	X = 1:budget
#
# 	# running tests
# 	for imeth in 1:lp
# 		regrets = zeros(1, budget)
#
# 		regrets = h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/log/ttts/", setting, "_", abrevs[imeth], ".h5"), "r") do file
#     		read(file, abrevs[imeth])
# 		end
#
# 		plot(log10.(X), -log10.(transpose(regrets/mcmc) ./ X), label = names[imeth])
# 	end
#
# 	xlabel("Allocation budget")
# 	ylabel("Expectation of the simple regret (log)")
# 	# xlim((2.0, 3.5))
# 	# ylim((3.0, 7.0))
# 	grid("on")
# 	legend(loc=4)
# 	if Sys.KERNEL == :Darwin
# 		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/results/ttts/", setting, "_log.png"))
# 	elseif Sys.KERNEL == :Linux
# 		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/results/ttts/", setting, "_log.png"))
# 	end
# 	close(fig)
# end

for setting in settings
	fig = figure()

	# running tests
	for imeth in 1:lp
		regrets = zeros(1, lbudget)

		regrets = h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/final/", setting, "_", abrevs[imeth], ".h5"), "r") do file
    		read(file, abrevs[imeth])
		end

		plot(budgets, reshape(regrets/mcmc, lbudget, 1), marker="*", label=policy_names[imeth])
	end

	xlabel("Allocation budget")
	ylabel("Expectation of the simple regret")
	# xlim((2.0, 3.5))
	# ylim((3.0, 7.0))
	grid("on")
	if Sys.KERNEL == :Darwin
		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/results/final/", setting, ".pdf"))
	elseif Sys.KERNEL == :Linux
		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/results/final/", setting, ".pdf"))
	end
	close(fig)
end
