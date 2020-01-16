using HDF5
using Distributed
using Statistics
using LinearAlgebra

if Sys.KERNEL == :Darwin
    @everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
    @everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end

# DO YOU WANT TO SAVE RESULTS?
typeExp = "NoSave"

# TYPE OF DISTRIBUTION
@everywhere distribution = "Gaussian"

# CHANGE NAME (save mode)
if Sys.KERNEL == :Darwin
    fname = "/Users/xuedong/Programming/PhD/BestArm.jl/misc/linear/xs"
elseif Sys.KERNEL == :Linux
    fname = "/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/linear/xs"
end

# BANDIT PROBLEM
# @everywhere c1 = [1, 0]
# @everywhere c2 = [cos(3/4*pi), sin(3/4*pi)]
# @everywhere c3 = [cos(pi/4+randn()/0.3), sin(pi/4+randn()/0.3)]
# @everywhere c4 = [cos(pi/4+randn()/0.3), sin(pi/4+randn()/0.3)]
# @everywhere c5 = [cos(pi/4+randn()/0.3), sin(pi/4+randn()/0.3)]
# @everywhere c6 = [cos(pi/4+randn()/0.3), sin(pi/4+randn()/0.3)]
# @everywhere contexts = [c1, c2, c3, c4, c5, c6]
# @everywhere true_theta = [1, 0]
# println(c3)
# println(c4)
# println(c5)
# println(c6)
@everywhere c1 = [1, 0, 0]
@everywhere c2 = [0, 1, 0]
@everywhere c3 = [0, 0, 1]
@everywhere c4 = [cos(pi/6), sin(pi/6), 0]
@everywhere contexts = [c1, c2, c3, c4]
@everywhere true_theta = [1, 0, 0]
@everywhere mu = [dot(c, true_theta) for c in contexts]
@everywhere best = findall(x -> x == maximum(mu), mu)[1]
K = length(mu)

# RISK LEVEL
delta = 0.01
d = 3

# NUMBER OF SIMULATIONS
N = 100

print("mu = $(mu)\n")

# POLICIES
@everywhere l_t3c_greedy(contexts, true_theta, delta, rate, dist) = BestArm.l_t3c(contexts, true_theta, delta, rate, dist, true)
@everywhere l_t3c_original(contexts, true_theta, delta, rate, dist) = BestArm.l_t3c(contexts, true_theta, delta, rate, dist, false)

@everywhere policies = [BestArm.lingape, l_t3c_greedy]
@everywhere namesPolicies = ["LinGapE", "L-T3C-Greedy"]

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
            rec, dra = policy(contexts, true_theta, delta, rate, distribution)
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
        print("average number of draws per arm: $(sum(Draws, dims=1)/N)\n")
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
            reco, draws = policy(contexts, true_theta, delta, rate, distribution)
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
