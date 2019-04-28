using PyPlot
# using ConfParser
using HDF5
# using Seaborn

reservoir = "Beta"
alphas = [0.5, 1.0, 2.0, 3.0, 1.0]
betas = [0.5, 1.0, 2.0, 1.0, 3.0]
budget = 64


for iparam in 1:5
	fig = figure()
	# Seaborn.set(style="darkgrid")

	# running tests
<<<<<<< HEAD
	x = [0:1:budget;]
	if Sys.KERNEL == :Darwin
		num_arms = h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "dttts", "_N.h5"), "r") do file
	    	read(file, "dttts")
		end
	elseif Sys.KERNEL == :Linux
		num_arms = h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "dttts", "_N.h5"), "r") do file
    		read(file, "dttts")
		end
	end
	num_arms = reshape(num_arms', (budget+1,))
	print(sum(num_arms))
	bar(x, num_arms, color="#0f87bf", align="center", alpha=0.4) # Histogram
=======
	x = [-1:1:budget-1;]
	num_arms1 = h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "dttts", "_N.h5"), "r") do file
    	read(file, "dttts")
	end
	num_arms2 = h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "isha16", "_N.h5"), "r") do file
    	read(file, "isha")
	end
	num_arms1 = reshape(num_arms1', (budget+1,))
	num_arms2 = reshape(num_arms2', (budget+1,))
	# println(string(alphas[iparam]), ",", betas[iparam], ": ", sum(num_arms))

	bar(x, num_arms1, width=0.4, color="b", align="center", alpha=0.8) # Histogram
	bar(x, num_arms2, width=0.4, color="r", align="center", alpha=0.4)
>>>>>>> eb8a187618d6e572decdfe2ca19ee717e2a3ad52

	xlabel("Number of arm plays")
	ylabel("Expectation of number of arms")
	grid("on")
	# legend(loc=1)
	if Sys.KERNEL == :Darwin
		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/results/hist/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "dttts", "_N.pdf"))
	elseif Sys.KERNEL == :Linux
		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/results/hist/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "dttts", "_N.pdf"))
	end
	close(fig)
end
