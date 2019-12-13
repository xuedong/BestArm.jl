using HDF5
using Statistics
using Distributed
using Plots
using Seaborn
using LaTeXStrings

if Sys.KERNEL == :Darwin
    @everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
    @everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end

# TYPE OF DISTRIBUTION
@everywhere distribution = "Bernoulli"

# CHANGE NAME (save mode)
if Sys.KERNEL == :Darwin
    fname = "/Users/xuedong/Programming/PhD/BestArm.jl/misc/linear/xs"
elseif Sys.KERNEL == :Linux
    fname = "/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/linear/xs"
end

# BANDIT PROBLEM
# make sure that the first element of the array is the maximum
@everywhere mu = [0.8 0.4 0.39 0.38]
@everywhere best = findall(x -> x == maximum(mu), mu)[1][2]
K = length(mu)

# RISK LEVEL
deltas = [1 / 10^k for k = 1:12]

# Variance for Gaussian Bandits
#sigma=1

# NUMBER OF SIMULATIONS
N = 100

# OPTIMAL SOLUTION
@everywhere v, optimal_weights = BestArm.optimal_weights(mu, distribution)
@everywhere gamma_optimal = optimal_weights[best]
@everywhere gamma_beta = BestArm.gamma_beta(mu, distribution)
@everywhere beta_weights = BestArm.beta_weights(mu, distribution, gamma_beta)
println("mu = $(mu)")
#println("Theoretical number of samples: $(v*log(1/delta))")
println("Optimal weights: $(optimal_weights)")
println("Beta-optimal weights: $(beta_weights)")
println()

# POLICIES

@everywhere policies = [BestArm.t3c_greedy, BestArm.d_tracking]
@everywhere names = ["T3C Greedy", "D-Tracking"]
#@everywhere policies = [BestArm.d_tracking]
#@everywhere names = ["D-Tracking"]

# EXPLORATION RATES
@everywhere explo(t, delta) = log((log(t) + 1) / delta)

rates = [explo for i = 1:lP]


# RUN EXPERIMENTS

function MCexp(mu, delta, N, imeth)
    Draws = zeros(N, K)
    policy = policies[imeth]
    rate = rates[imeth]
    startTime = time()
    Reco, Draws = @distributed ((x, y) -> (vcat(x[1], y[1]), vcat(x[2], y[2]))) for n = 1:N
        rec, dra = policy(mu, delta, rate, distribution)
        rec, dra
    end
    Error = collect([(r == best) ? 0 : 1 for r in Reco])
    FracNT = sum([r == 0 for r in Reco]) / N
    FracReco = zeros(K)
    proportion = zeros(K)
    for k = 1:K
        FracReco[k] = sum([(r == k) ? 1 : 0 for r in Reco]) / (N * (1 - FracNT))
    end
    for n = 1:N
        if (Reco[n] != 0)
            proportion += Draws[n, :] / sum(Draws[n, :])
        end
    end
    proportion = proportion / (N * (1 - FracNT))
    print("Results for $(policy), average on $(N) runs\n")
    print("proportion of runs that did not terminate: $(FracNT)\n")
    print("average number of draws: $(sum(Draws)/(N*(1-FracNT)))\n")
    print("average proportions of draws: $(proportion)\n")
    print("proportion of errors: $(sum(Error)/(float(N*(1-FracNT))))\n")
    print("proportion of recommendation made when termination: $(FracReco)\n")
    print("elapsed time: $(time()-startTime)\n\n")
    return sum(Draws) / (N * (1 - FracNT))
end


draws = zeros(3, length(deltas))
for imeth = 1:length(policies)
    for k = 1:length(deltas)
        draws[imeth, k] = MCexp(mu, deltas[k], N, imeth)
    end
end

f(x) = 1 / v * x
g(x) = 1 / gamma_beta * x

Seaborn.set()
Plots.default(overwrite_figure = false)
Plots.plot(f, 0, 30, label = L"$\texttt{Oracle}: 1/\Gamma^{\star}$", leg = :topleft, title = latexstring("\\texttt{$(mu)}"))
Plots.plot!(g, 0, 30, label = L"$\beta\texttt{-Oracle}: 1/\Gamma_{0.5}^{\star}$", linestyle = :dot)
shapes = [:diamond, :circle, :hexagon]
for imeth in 1:length(policies)
    Plots.scatter!(
        [log(1 / delta) for delta in deltas],
        draws[imeth, :],
        label = latexstring("\$\\texttt{$(names[imeth])}\$"),
        markershape = shapes[imeth],
        markersize = 2,
    )
end
Plots.xlabel!(L"$\log(1/\delta)$")
Plots.ylabel!(L"\texttt{Empirical stopping time}")
Plots.savefig("test2.pdf")
