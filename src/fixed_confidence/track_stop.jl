"""

"""
function d_tracking(mu::Array, delta::Real, rate::Function, dist::String,
	stopping::String = "Chernoff")
   	# D-Tracking + Chernoff stopping
   	condition = true
   	K = length(mu)
   	N = zeros(1, K)
   	S = zeros(1, K)
   	# initialization
   	for a in 1:K
      	N[a] = 1
      	S[a] = sample(mu[a])
   	end
   	t = K
   	Best = 1
   	while (condition)
      	Mu = S./N
      	# Empirical best arm
      	IndMax = findall(x -> x==maximum(Mu),Mu)
      	Best = IndMax[floor(Int, length(IndMax)*rand())+1]
      	I = 1
      	if (length(IndMax) > 1)
		  	# if multiple maxima, draw one them at random
		  	I = Best
      	else
		  	# compute the stopping statistic
		  	NB = N[Best]
         	SB = S[Best]
         	muB = SB/NB
         	MuMid = (SB.+S)./(NB.+N)
         	Index = collect(1:K)
         	deleteat!(Index, Best[2])
         	Score = minimum([NB*d(muB, MuMid[i])+N[i]*d(Mu[i], MuMid[i]) for i in Index])
         	if (Score > rate(t, delta))
            	# stop
            	condition = false
         	elseif (t > 10000000)
            	# stop and outputs (0,0)
            	condition = false
            	Best = 0
            	print(N)
            	print(S)
            	N = zeros(1, K)
         	else
            	if (minimum(N) <= max(sqrt(t) - K/2,0))
               		# forced exploration
               		I = argmin(N)
            	else
               	 	# continue and sample an arm
	            	val, Dist = optimal_weights(Mu, dist, 1e-11)
               		# choice of the arm
               		I = argmax(Dist-N/t)
            	end
	      	end
      	end
      	# draw the arm
      	t += 1
      	S[I] += sample(mu[I])
      	N[I] += 1
   	end
   	recommendation = Best[2]
   	return (recommendation, N)
end


function c_tracking(mu::Array, delta::Real, rate::Function, dist::String)
    # Uses a tracking of the cummulated sum
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample_arm(mu[a], dist)
    end
    t=K
    Best=1
    SumWeights=ones(1,K)/K
    while (condition)
        Mu=S./N
        # Empirical best arm
        IndMax=find(Mu.==maximum(Mu))
        Best=IndMax[floor(Int,length(IndMax)*rand())+1]
        I=1
        # Compute the stopping statistic
        NB=N[Best]
        SB=S[Best]
        muB=SB/NB
        MuMid=(SB+S)./(NB+N)
        Index=collect(1:K)
        splice!(Index,Best)
        Score=minimum([NB*d(muB, MuMid[i], dist)+N[i]*d(Mu[i], MuMid[i], dist) for i in Index])
        if (Score > rate(t,0,delta))
            # stop
            condition=false
        elseif (t >1000000)
            # stop and output (0,0)
            condition=false
            Best=0
            print(N)
            print(S)
            N=zeros(1,K)
        else
            # continue and sample an arm
            val,Dist=OptimalWeights(Mu,1e-11)
            SumWeights=SumWeights+Dist
            # choice of the arm
            if (minimum(N) <= max(sqrt(t) - K/2,0))
                # forced exploration
                I=indmin(N)
            else
                I=indmax(SumWeights-N)
            end
        end
        # draw the arm
        t+=1
        S[I]+=sample_arm(mu[I], dist)
        N[I]+=1
    end
    recommendation=Best
    return (recommendation,N)
end
