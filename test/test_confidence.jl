using HDF5
using Statistics
using Distributed

if Sys.KERNEL == :Darwin
    @everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
    @everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end

# DO YOU WANT TO SAVE RESULTS?
typeExp = "NoSave"

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
@everywhere mu = [0.9 0.51 0.5 0.49]
@everywhere best = findall(x -> x == maximum(mu), mu)[1][2]
K = length(mu)

# RISK LEVEL
delta = 1e-11

# Variance for Gaussian Bandits
#sigma=1

# NUMBER OF SIMULATIONS
N = 10

# OPTIMAL SOLUTION
@everywhere v, optimal_weights = BestArm.optimal_weights(mu, distribution)
@everywhere gamma_optimal = optimal_weights[best]
@everywhere gamma_beta = BestArm.gamma_beta(mu, distribution)
@everywhere beta_weights = BestArm.beta_weights(mu, distribution, gamma_beta)
println("mu = $(mu)")
println("Theoretical number of samples: $(v*log(1/delta))")
println("Optimal weights: $(optimal_weights)")
println("Beta-optimal weights: $(beta_weights)")
println()

# POLICIES

@everywhere policies = [BestArm.t3c_optimal, BestArm.t3c, BestArm.best_challenger]
@everywhere namesPolicies = ["T3C2", "T3C", "BC"]
# @everywhere policies = [
#     BestArm.best_challenger,
#     BestArm.kl_lucb,
#     BestArm.racing,
#     BestArm.t3c,
#     BestArm.target,
#     BestArm.d_tracking,
#     BestArm.ttei,
#     BestArm.ttts_c,
#     BestArm.ugape_c,
#     BestArm.uniform_c,
# ]
# @everywhere namesPolicies = [
#     "BC",
#     "KL-LUCB",
#     "Racing",
#     "T3C",
#     "Target",
#     "D-Tracking",
#     "TTEI",
#     "TTTS",
#     "UGapE",
#     "Uniform",
# ]

# EXPLORATION RATES
@everywhere explo(t, delta) = log((log(t) + 1) / delta)

lP = length(policies)
rates = [explo for i = 1:lP]


# RUN EXPERIMENTS

function MCexp(mu, delta, N)
    for imeth = 1:lP
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
    end
end


function SaveData(mu, delta, N)
    K = length(mu)
    for imeth = 1:lP
        Draws = zeros(N, K)
        policy = policies[imeth]
        rate = rates[imeth]
        namePol = namesPolicies[imeth]
        startTime = time()
        Reco, Draws = @distributed ((x, y) -> (vcat(x[1], y[1]), vcat(x[2], y[2]))) for n = 1:N
            reco, draws = policy(mu, delta, rate, distribution)
            reco, draws
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
        print("elapsed time: $(time()-startTime)\n")
        print("one step time: $((time()-startTime)/((sum(Draws)/(N*(1-FracNT)))))\n\n")
        name = "$(fname)_$(namePol)_delta_$(delta)_N_$(N).h5"
        h5write(name, "mu", mu)
        h5write(name, "delta", delta)
        h5write(name, "FracNT", collect(FracNT))
        h5write(name, "FracReco", FracReco)
        h5write(name, "Draws", Draws)
        h5write(name, "Error", mean(Error))
    end
end


if (typeExp == "Save")
    SaveData(mu, delta, N)
else
    MCexp(mu, delta, N)
end
