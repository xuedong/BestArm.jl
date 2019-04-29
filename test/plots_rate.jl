using PyPlot
# using ConfParser
using HDF5
# using Seaborn

budgets = [1000, 2000, 2000, 2000, 600, 4000, 6000, 6000]
settings = ["setting0", "setting1", "setting2", "setting3", "setting4", "setting5", "setting6", "setting7"]
mcmc = 1000

for i in 1:length(settings)
	fig = figure()
	budget = budgets[i]
	setting = settings[i]
	# Seaborn.set(style="darkgrid")

	# running tests
	X = 1:budget
	if Sys.KERNEL == :Darwin
		hits = h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/ttts/", setting, "_hits.h5"), "r") do file
    		read(file, "ttts")
		end
	elseif Sys.KERNEL == :Linux
		hits = h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/ttts/", setting, "_hits.h5"), "r") do file
    		read(file, "ttts")
		end
	end

	Y = reshape(1 .- hits/mcmc, budget, 1)
	Y = [-1/i*log(Y[i]) for i in 1:budget]
	plot(X[2:end], Y[2:end], linestyle="--", label="TTTS")

	xlabel("Budget")
	ylabel("Rate")
	grid("on")
	# legend(loc=1)
	if Sys.KERNEL == :Darwin
		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/results/ttts/", setting, "_rate.pdf"))
	elseif Sys.KERNEL == :Linux
		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/results/ttts/", setting, "_rate.pdf"))
	end
	close(fig)
end
