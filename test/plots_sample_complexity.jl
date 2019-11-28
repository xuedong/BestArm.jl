# Visualize results that are saved in the results folder (names should match)

using PyPlot
using HDF5
using Statistics
using Seaborn

Seaborn.set()

# EPSILON BAI?
EpsilonBAI = "False"

# NAME AND POLICIES NAMES (should match saved data)
if Sys.KERNEL == :Darwin
    fname = "/Users/xuedong/Programming/PhD/BestArm.jl/misc/linear/xs"
elseif Sys.KERNEL == :Linux
    fname = "/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/linear/xs"
end
#namesPolicies = ["T3C", "TTTS", "TTEI", "BC", "D-Tracking", "Uniform", "UGapE"]
namesPolicies = ["TTTS", "L-T3S", "T3C", "L-T3C"]

# PARAMETERS
delta = 0.01
N = 10

clf()

# CHANGE xmax DEPENDING ON THE MAXIMAL RANGE!
xmax = 1000
NBins = 30

# position of the text on the graphs
xtxt = 0.6 * xmax
Bins = round.(Int, range(1, stop = xmax, length = NBins))
# arranging the graphs
xdim = length(namesPolicies)
ydim = 1


mu = h5read("$(fname)_$(namesPolicies[1])_delta_$(delta)_N_$(N).h5", "mu")
K = length(mu)
if EpsilonBAI == "True"
  epsilon = h5read("$(fname)_$(namesPolicies[1])_delta_$(delta)_N_$(N).h5", "epsilon")
end

clf()
title("mu = $(mu)")
Means = zeros(length(namesPolicies), 1)
Stds = zeros(length(namesPolicies), 1)

# for j in 1:length(namesPolicies)
#     name = "$(fname)_$(namesPolicies[j])_delta_$(delta)_N_$(N).h5"
#     FracNT = h5read(name, "FracNT")
#     Draws = h5read(name, "Draws")
#     Error = h5read(name, "Error")
#     FracReco = h5read(name, "FracReco")
#     subplot(xdim, ydim, j)
#     NbDraws = sum(Draws, dims = 2)
#     proportion = zeros(N, K)
#     for k in 1:N
#         if (sum(Draws[k,:]) != 0)
#             proportion[k,:] = Draws[k,:] / sum(Draws[k,:])
#         end
#     end
#     prop = sum(proportion, dims = 1) / N * (1 - FracNT)
#     MeanDraws = mean(NbDraws)
#     Means[j] = MeanDraws
#     StdDraws = std(NbDraws)
#     Stds[j] = StdDraws
#     histo = plt.hist(vec(NbDraws), Bins)
#     Mhisto = maximum(histo[1])
#     PyPlot.axis([0,xmax,0,Mhisto])
#     ytxt1 = 0.75 * Mhisto
#     ytxt2 = 0.6 * Mhisto
#     ytxt3 = 0.5 * Mhisto
#     ytxt4 = 0.4 * Mhisto
#     EmpError = round.(Int, 10000 * Error) / 10000
#     axvline(MeanDraws, color = "black", linewidth = 2.5)
#     PyPlot.text(xtxt, ytxt1, "mean = $(round(Int, MeanDraws)) (std=$(round(Int, StdDraws)))")
#     PyPlot.text(xtxt, ytxt2, "delta = $(delta)")
#     PyPlot.text(xtxt, ytxt3, "fraction no term = $(FracNT)")
#     PyPlot.text(xtxt, ytxt4, "emp. recommendation made = $(FracReco)")
#     if (j == 1)
#         if (EpsilonBAI == "True")
#             title("mu=$(mu), epsilon=$(epsilon), $(namesPolicies[j])")
#         else
#             title("mu=$(mu), $(namesPolicies[j])")
#         end
#     else
#         title("$(namesPolicies[j])")
#     end
#     print("Results for $(namesPolicies[j]), average on $(N) runs\n")
#     print("proportion of runs that did not terminate: $(FracNT)\n")
#     print("average number of draws: $(MeanDraws)\n")
#     print("average proportion of draws: \n $(prop)\n")
#     print("proportion of errors: $(EmpError)\n")
#     print("proportion of recommendation made when termination: $(FracReco)\n\n")
# end
# savefig("test1.pdf")


results = Array{Any}(undef, length(namesPolicies))
for imeth = 1:length(namesPolicies)
  namePol = namesPolicies[imeth]
  name = "$(fname)_$(namesPolicies[imeth])_delta_$(delta)_N_$(N).h5"
  draws = h5read(name, "Draws")
  results[imeth] = vec(sum(draws, dims = 2))
end

meanpointprops = Dict(
  "marker" => :o,
  "markersize" => 6,
  "markeredgecolor" => :black,
  "markerfacecolor" => :black,
)
fig, ax = subplots()#figure("pyplot_boxplot",figsize=(10,10))
ax[:boxplot](
  results,
  labels = namesPolicies,
  showfliers = false,
  showmeans = true,
  meanprops = meanpointprops,
  patch_artist = true,
  notch = true,
)
ax[:set_title](L"$\delta = 0.01, \sigma = 1$")
ax[:tick_params]()
ax[:yaxis][:grid](true)

savefig("test.pdf")


# fig = subplots()
# x_pos = 1:length(namesPolicies)
# Means = reshape(Means, length(namesPolicies))
# Stds = reshape(Stds, length(namesPolicies))
# bar(x_pos, Means, yerr = Stds, tick_label=namesPolicies, align = "center", alpha = 0.5, ecolor = "black", capsize = 5)
# ylabel("Number of draws")
# #xticks(x_pos)
# title("$(mu), delta = 0.01")
#
# # Save the figure and show
# tight_layout()
# savefig("test2.pdf")
# show()
