function t3c(mu::Array, delta::Real, rate::Function, dist::String,
    frac::Real = 0.5, alpha::Real = 1, beta::Real = 1,
	stopping::String = "chernoff")
   	# T3C with Chernoff stopping rule
   	continuing = true
   	K=length(mu)
   	N = zeros(1,K)
   	S = zeros(1,K)
   	# initialization
   	for a in 1:K
      	N[a] = 1
      	S[a] = sample_arm(mu[a], dist)
   	end
   	t=K
   	TrueBest=1
   	while continuing
      	Mu=S./N
      	# Empirical best arm
      	Best=randmax(Mu)
      	TrueBest=randmax(Mu)
      	# Compute the stopping statistic
      	NB=N[Best]
      	SB=S[Best]
      	MuB=SB/NB
      	MuMid=(SB.+S)./(NB.+N)
      	Score=minimum([NB*d(MuB, MuMid[i], dist)+N[i]*d(Mu[i], MuMid[i], dist) for i in 1:K if i!=Best])
      	# compute the best arm and the challenger
      	for a=1:K
         	if dist == "Gaussian"
            	Mu[a] = rand(Normal(S[a] / N[a], alpha / sqrt(N[a])), 1)[1]
         	elseif dist == "Bernoulli"
            	Mu[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
         	end
      	end
      	Best=randmax(Mu)
      	NB=N[Best]
      	MuB=Mu[Best]
      	MuMid=(NB*MuB.+N.*Mu)./(NB.+N)
      	Challenger=1
      	NewScore=Inf
      	for i=1:K
	      	if i!=Best
            	score=NB*d(MuB, MuMid[i], dist)+N[i]*d(Mu[i], MuMid[i], dist)
	         	if (score<NewScore)
		         	Challenger=i
		         	NewScore=score
	         	end
	      	end
      	end
      	I = 1
      	if (Score > rate(t,delta))
         	# stop
         	continuing=false
      	elseif (t >1e7)
         	# stop and return (0,0)
         	continuing=false
         	TrueBest=0
         	print(N)
         	print(S)
         	N=zeros(1,K)
      	else
         	# continue and sample an arm
	      	if (rand()>frac)
            	# TS sample
            	I=Best
         	else
            	# choose between the arm and its Challenger
            	I=Challenger
         	end
      	end
      	# draw the arm
      	t+=1
      	S[I] += sample_arm(mu[I], dist)
      	N[I] += 1
   	end
   	return TrueBest,N
end
