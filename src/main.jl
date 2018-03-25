# Run Experiments, display results (and possibly save data) on a Bandit Problem to be specified

using HDF5

# DO YOU WANT TO SAVE RESULTS?
typeExp = "Save"
#typeExp = "NoSave"

# TYPE OF DISTRIBUTION
type_dist="Bernoulli"
include("BAIConfidence.jl")
include("BAIBudget.jl")
include("Arms.jl")
include("Utils.jl")

# CHANGE NAME (save mode)
fname="results/Experiment4arms"

# BANDIT PROBLEM
mu=vec([0.3 0.25 0.2 0.1])
best=find(mu.==maximum(mu))[1]
K=length(mu)

# RISK LEVEL
delta=0.1

# NUMBER OF SIMULATIONS
N=5


# OPTIMAL SOLUTION
@everywhere v,optWeights=OptimalWeights(mu)
@everywhere gammaOpt=optWeights[best]
print("mu=$(mu)\n")
print("Theoretical number of samples: $((1/v)*log(1/delta))\n")
print("Optimal weights: $(optWeights)\n\n")

# POLICIES

@everywhere ChernoffPTSHalf(x,y,z)=ChernoffPTS(x,y,z,0.5)
@everywhere ChernoffPTSOpt(x,y,z)=ChernoffPTS(x,y,z,gammaOpt)

policies=[TrackAndStop,ChernoffBC2,ChernoffPTSHalf,ChernoffPTSOpt,KLLUCB,UGapEC]
names=["TrackAndStop","ChernoffBC","ChernoffPTS","ChernoffPTSOpt","KLLUCB","UGapEC"]


# EXPLORATION RATES
@everywhere explo(t,n,delta)=log((log(t)+1)/delta)

lP=length(policies)
rates=[explo for i in 1:lP]



# RUN EXPERIMENTS

function MCexp(mu,delta,N)
	for imeth=1:lP
		Draws=zeros(N,K)
		policy=policies[imeth]
		beta=rates[imeth]
		startTime=time()
		Reco,Draws = @parallel ((x,y) -> (vcat(x[1],y[1]),vcat(x[2],y[2]))) for n in 1:N
			rec,dra = policy(mu,delta,beta)
			rec,dra
		end
		Error=collect([(r==best)?0:1 for r in Reco])
		FracNT=sum([r==0 for r in Reco])/N
		FracReco=zeros(K)
		proportion = zeros(K)
		for k in 1:K
			FracReco[k]=sum([(r==k)?1:0 for r in Reco])/(N*(1-FracNT))
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
		namePol=names[imeth]
		startTime=time()
		Reco,Draws = @parallel ((x,y) -> (vcat(x[1],y[1]),vcat(x[2],y[2]))) for n in 1:N
			reco,draws = policy(mu,delta,beta)
			reco,draws
		end
		Error=collect([(r==best)?0:1 for r in Reco])
		FracNT=sum([r==0 for r in Reco])/N
		FracReco=zeros(K)
		proportion = zeros(K)
		for k in 1:K
			FracReco[k]=sum([(r==k)?1:0 for r in Reco])/(N*(1-FracNT))
		end
		for n in 1:N
			if (Reco[n]!=0)
				proportion += Draws[n,:]/sum(Draws[n,:])
			end
		end
		proportion = proportion / N
		print("Results for $(policy), average on $(N) runs\n")
		print("proportion of runs that did not terminate: $(FracNT)\n")
		print("average number of draws: $(sum(Draws)/(N*(1-FracNT)))\n")
		print("average proportions of draws: $(proportion)\n")
		print("proportion of errors: $(sum(Error)/(float(N*(1-FracNT))))\n")
		print("proportion of recommendation made when termination: $(FracReco)\n")
		print("elapsed time: $(time()-startTime)\n\n")
		name="$(fname)_$(namePol)_delta_$(delta)_N_$(N).h5"
		h5write(name,"mu",mu)
		h5write(name,"delta",delta)
		h5write(name,"FracNT",collect(FracNT))
		h5write(name,"FracReco",FracReco)
		h5write(name,"Draws",Draws)
		h5write(name,"Error",Error)
	end
end


if (typeExp=="Save")
   SaveData(mu,delta,N)
else
   MCexp(mu,delta,N)
end
