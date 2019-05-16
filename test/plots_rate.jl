using PyPlot
# using ConfParser
using HDF5
using Seaborn

budgets = [1000, 2000, 2000, 2000, 600, 4000, 6000, 6000]
gamma_betas = [0.00407116, 0.000503566, 0.000540953, 0.000168461, 0.0018092, 0.000853012, 0.000179027, 0.000172676]
settings = ["setting0", "setting1", "setting2", "setting3", "setting4", "setting5", "setting6", "setting7"]
mcmc = 10

for i in 1:5
	fig = figure()
	budget = budgets[i]
	setting = settings[i]
	Seaborn.set(style="darkgrid")

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
	Y = [-1/j*log(Y[j]) for j in 1:budget]
	# println(Y[end])
	oracle = [gamma_betas[i] for j in 1:budget]
	plot(X[2:end], Y[2:end], linestyle="--", label="TTTS")
	plot(X[2:end], oracle[2:end], label="Gamma")

	xlabel("Budget")
	ylabel("Rate")
	grid("on")
	legend(loc=1)
	if Sys.KERNEL == :Darwin
		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/results/ttts/", setting, "_rate.pdf"))
	elseif Sys.KERNEL == :Linux
		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/results/ttts/", setting, "_rate.pdf"))
	end
	close(fig)
end
