function racing(mu::Array, delta::Real, rate::Function, dist::String,
	elimination::Symbol=:Chernoff)
    condition = true
  	K = length(mu)
  	N = zeros(1,K)
  	S = zeros(1,K)
  	# initialization
  	for a in 1:K
      	N[a] = 1
      	S[a] = sample_arm(mu[a], dist)
  	end
  	round=1
  	t=K
  	Remaining=collect(1:K)
  	while (length(Remaining)>1)
      	# Drawn all remaining arms
      	for a in Remaining
	      	S[a] += sample(mu[a], dist)
	      	N[a] += 1
      	end
      	round+=1
      	t+=length(Remaining)
      	# Check whether the worst should be removed
      	Mu=S./N
      	MuR=Mu[Remaining]
      	MuBest=maximum(MuR)
      	IndBest=randmax(MuR)
      	Best=Remaining[IndBest]
      	MuWorst=minimum(MuR)
      	IndWorst=randmax(-MuR)
	  	if elimination == "Chernoff"
      		if (round*(d(MuBest, (MuBest+MuWorst)/2, dist)+d(MuWorst, (MuBest+MuWorst)/2), dist) > rate(t,delta))
         		# remove Worst arm
         		deleteat!(Remaining, IndWorst)
      		end
		elseif elimination == "KL"
			kl_low = dlow(MuBest, rate(round, delta)/round, dist)
			kl_up = dup(MuWorst, rate(round, delta)/round, dist)
			if kl_low > kl_up
         		# remove Worst arm
         		deleteat!(Remaining, IndWorst)
			end
      	end
      	if (t>1000000)
	      	Remaining=[0]
         	N=zeros(1,K)
      	end
   	end
   	recommendation=Remaining[1]
   	return (recommendation,N)
end
