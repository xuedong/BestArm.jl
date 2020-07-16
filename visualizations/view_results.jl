# Visualize results that are saved in the results folder (names should match)

using PyPlot
using HDF5
using Statistics
using Seaborn
using LaTeXStrings
# using Plots

Seaborn.set()

ϵ_bai = false

# NAME AND POLICIES NAMES (should match saved data)
fname = "/home/xuedong/Downloads/t3c/results/xs"
names = ["T3C", "TTTS", "TTEI", "BC", "D-Tracking", "Uniform", "UGapE"]
#names = ["BC2", "T3C"]

# PARAMETERS
delta = 0.01
N = 1000

clf()

# CHANGE xmax DEPENDING ON THE MAXIMAL RANGE!
xmax = 1000
NBins = 30

# position of the text on the graphs
xtxt = 0.6 * xmax
Bins = round.(Int, range(1, stop = xmax, length = NBins))
# arranging the graphs
xdim = length(names)
ydim = 1


mu = h5read("$(fname)_$(names[1])_delta_$(delta)_N_$(N).h5", "mu")
K = length(mu)
if ϵ_bai == "True"
  ϵ = h5read("$(fname)_$(names[1])_delta_$(delta)_N_$(N).h5", "epsilon")
end

clf()
title("mu = $(mu)")
Means = zeros(length(names), 1)
Stds = zeros(length(names), 1)

results = Array{Any}(undef, length(names))
for imeth = 1:length(names)
  namePol = names[imeth]
  name = "$(fname)_$(names[imeth])_delta_$(delta)_N_$(N).h5"
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
  labels = names,
  showfliers = false,
  showmeans = true,
  meanprops = meanpointprops,
  patch_artist = true,
  notch = true,
)
ax[:axhline](422.37, ls = "--", color = "r", label = L"$N^\star=422.37$")
ax[:axhline](466.38, ls = "-.", color = "g", label = L"$N_{0.5}^\star=466.38$")
# labels = [L"$N^\star$", L"$N_{0.5}^\star$"]
# handles, _ = ax[:get_legend_handles_labels]()
# ax[:legend](handles, labels = labels)
ax[:legend]()
ax[:set_title]("Problem 1, Gaussian bandits, " * L"$\delta=0.01, \sigma=1$")
ax[:tick_params]()
ax[:yaxis][:grid](true)

savefig("test.pdf")


# fig = subplots()
# x_pos = 1:length(names)
# Means = reshape(Means, length(names))
# Stds = reshape(Stds, length(names))
# bar(x_pos, Means, yerr = Stds, tick_label=names, align = "center", alpha = 0.5, ecolor = "black", capsize = 5)
# ylabel("Number of draws")
# #xticks(x_pos)
# title("$(mu), delta = 0.01")
#
# # Save the figure and show
# tight_layout()
# savefig("test2.pdf")
# show()
