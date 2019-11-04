using HDF5
using Distributed

if Sys.KERNEL == :Darwin
	@everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	@everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end

# DO YOU WANT TO SAVE RESULTS?
#typeExp = "Save"
typeExp = "NoSave"

# TYPE OF DISTRIBUTION
@everywhere distribution="Gaussian"

# CHANGE NAME (save mode)
fname="/home/xuedong/Downloads/t3c/results/xs"

# BANDIT PROBLEM
@everywhere mu=[1 0.8 0.75 0.7]
@everywhere best=findall(x->x==maximum(mu),mu)[1][2]
K=length(mu)

# RISK LEVEL
delta=0.01

# Variance for Gaussian Bandits
sigma=1

# NUMBER OF SIMULATIONS
N=1

# OPTIMAL SOLUTION
@everywhere v, optWeights = BestArm.optimal_weights(mu, distribution)
@everywhere gammaOpt=optWeights[best]
print("mu=$(mu)\n")
print("Theoretical number of samples: $(v*log(1/delta))\n")
print("Optimal weights: $(optWeights)\n\n")

# POLICIES

@everywhere ChernoffBCForcedExplo(mu,delta,explo)=ChernoffBC(mu,delta,explo,true)
@everywhere ChernoffBCTS(mu,delta,explo)=ChernoffBC(mu,delta,explo,false,true)

# @everywhere policies = [BestArm.ttts_c, BestArm.ttei, BestArm.best_challenger_ts, BestArm.d_tracking, BestArm.uniform_c, BestArm.ugape_c]
# @everywhere namesPolicies = ["TTTS", "TTEI", "BC", "D-Tracking", "Uniform", "UGapE"]
@everywhere policies = [BestArm.ttts_c, BestArm.ttei, BestArm.t3c]
@everywhere namesPolicies = ["TTTS", "TTEI", "T3C"]

# EXPLORATION RATES
@everywhere explo(t, delta)=log((log(t)+1)/delta)

lP=length(policies)
rates=[explo for i in 1:lP]


# RUN EXPERIMENTS

function MCexp(mu,delta,N)
	for imeth=1:lP
		Draws=zeros(N,K)
		policy=policies[imeth]
		beta=rates[imeth]
		startTime=time()
		Reco,Draws = @distributed ((x,y) -> (vcat(x[1],y[1]),vcat(x[2],y[2]))) for n in 1:N
				rec,dra = policy(mu, delta, beta, distribution)
				rec,dra
		end
		Error=collect([(r==best) ? 0 : 1 for r in Reco])
		FracNT=sum([r==0 for r in Reco])/N
		FracReco=zeros(K)
		proportion = zeros(K)
		for k in 1:K
			FracReco[k]=sum([(r==k) ? 1 : 0 for r in Reco])/(N*(1-FracNT))
		end
		for n in 1:N
			if (Reco[n]!=0)
			    proportion += Draws[n,:]/sum(Draws[n,:])
			end
		end
		proportion = proportion / (N*(1-FracNT))
		print("Results for $(policy), average on $(N) runs\n")
		print("proportion of runs that did not terminate: $(FracNT)\n")
		print("average number of draws: $(sum(Draws)/(N*(1-FracNT)))\n")
		print("average proportions of draws: $(proportion)\n")
		print("proportion of errors: $(sum(Error)/(float(N*(1-FracNT))))\n")
		print("proportion of recommendation made when termination: $(FracReco)\n")
		print("elapsed time: $(time()-startTime)\n\n")
	end
end


function SaveData(mu,delta,N)
	K=length(mu)
    for imeth=1:lP
        Draws=zeros(N,K)
        policy=policies[imeth]
		beta=rates[imeth]
        namePol=namesPolicies[imeth]
        startTime=time()
		Reco,Draws = @distributed ((x,y) -> (vcat(x[1],y[1]),vcat(x[2],y[2]))) for n in 1:N
	        reco,draws = policy(mu,delta,beta)
	        reco,draws
	    end
		Error=collect([(r==best) ? 0 : 1 for r in Reco])
        FracNT=sum([r==0 for r in Reco])/N
        FracReco=zeros(K)
		proportion = zeros(K)
        for k in 1:K
            FracReco[k]=sum([(r==k) ? 1 : 0 for r in Reco])/(N*(1-FracNT))
		end
		for n in 1:N
			if (Reco[n]!=0)
			   proportion += Draws[n,:]/sum(Draws[n,:])
			end
        end
		proportion = proportion /(N*(1-FracNT))
        print("Results for $(policy), average on $(N) runs\n")
	    print("proportion of runs that did not terminate: $(FracNT)\n")
	    print("average number of draws: $(sum(Draws)/(N*(1-FracNT)))\n")
		print("average proportions of draws: $(proportion)\n")
	    print("proportion of errors: $(sum(Error)/(float(N*(1-FracNT))))\n")
        print("proportion of recommendation made when termination: $(FracReco)\n")
        print("elapsed time: $(time()-startTime)\n")
		print("one step time: $((time()-startTime)/((sum(Draws)/(N*(1-FracNT)))))\n\n")
        name="$(fname)_$(namePol)_delta_$(delta)_N_$(N).h5"
        h5write(name,"mu",mu)
        h5write(name,"delta",delta)
        h5write(name,"FracNT",collect(FracNT))
        h5write(name,"FracReco",FracReco)
        h5write(name,"Draws",Draws)
        h5write(name,"Error",mean(Error))
	end
end


if (typeExp=="Save")
   SaveData(mu,delta,N)
else
   MCexp(mu,delta,N)
end
