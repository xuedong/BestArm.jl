function ChernoffBC(mu,delta,rate)
  # Chernoff stopping rule,  sampling based on the "best challenger"
  # described in experimental section of [Garivier and Kaufmann 2016]
  condition = true
  K=length(mu)
  N = zeros(1,K)
  S = zeros(1,K)
  # initialization
  for a in 1:K
      N[a]=1
      S[a]=sample_arm(mu[a], type_dist)
  end
  t=K
  Best=1
  while (condition)
       Mu=S./N
       # Empirical best arm
       IndMax=find(Mu.==maximum(Mu))
       Best=IndMax[floor(Int,length(IndMax)*rand())+1]
       I=1
       # Compute the stopping statistic
       NB=N[Best]
       SB=S[Best]
       MuB=SB/NB
       MuMid=(SB+S)./(NB+N)
       Challenger=1
       Score=Inf
       for i=1:K
	  if i!=Best
             score=NB*d(MuB, MuMid[i], type_dist)+N[i]*d(Mu[i], MuMid[i], type_dist)
	     if (score<Score)
		 Challenger=i
		 Score=score
	     end
	  end
       end
       if (Score > rate(t,0,delta))
            # stop
            condition=false
          elseif (t >1000000)
            # stop and return (0,0)
            condition=false
            Best=0
            print(N)
            print(S)
            N=zeros(1,K)
          else
            # continue and sample an arm
	    val,Dist=OptimalWeights(Mu,1e-11)
            if (minimum(N) <= max(sqrt(t) - K/2,0))
               # forced exploration
               I=indmin(N)
             else
               # choose between the arm and its Challenger
               I=(NB/(NB+N[Challenger]) < Dist[Best]/(Dist[Best]+Dist[Challenger]))?Best:Challenger
               #I=(d(MuB, MuMid[Challenger], type_dist)>d(Mu[Challenger], MuMid[Challenger], type_dist))?Best:Challenger
               #I=(N[Best]<N[Challenger])?Best:Challenger
            end
       end
       # draw the arm
       t+=1
       S[I]+=sample_arm(mu[I], type_dist)
       N[I]+=1
   end
   recommendation=Best
   return (recommendation,N)
end



function ChernoffBC2(mu,delta,rate)
  # Chernoff stopping rule + alternative choice between the empirical best and its "challenger"
  # Faster, requires no computation of Optimal Weights
  condition = true
  K=length(mu)
  N = zeros(1,K)
  S = zeros(1,K)
  # initialization
  for a in 1:K
      N[a]=1
      S[a]=sample_arm(mu[a], type_dist)
  end
  t=K
  Best=1
  while (condition)
       Mu=S./N
       # Empirical best arm
       IndMax=find(Mu.==maximum(Mu))
       Best=IndMax[floor(Int,length(IndMax)*rand())+1]
       I=1
       # Compute the stopping statistic
       NB=N[Best]
       SB=S[Best]
       MuB=SB/NB
       MuMid=(SB+S)./(NB+N)
       Challenger=1
       Score=Inf
       for i=1:K
	  if i!=Best
             score=NB*d(MuB, MuMid[i], type_dist)+N[i]*d(Mu[i], MuMid[i], type_dist)
	     if (score<Score)
		 Challenger=i
		 Score=score
	     end
	  end
       end
       if (Score > rate(t,0,delta))
            # stop
            condition=false
          elseif (t >1000000)
            # stop and return (0,0)
            condition=false
            Best=0
            print(N)
            print(S)
            N=zeros(1,K)
          else
            # continue and sample an arm
	    if (minimum(N) <= max(sqrt(t) - K/2,0))
               # forced exploration
               I=indmin(N)
             else
               # choose between the arm and its Challenger
               I=(N[Best]<N[Challenger])?Best:Challenger
               #I=(d(MuB, MuMid[Challenger], type_dist)>d(Mu[Challenger], MuMid[Challenger], type_dist))?Best:Challenger
            end
       end
       # draw the arm
       t+=1
       S[I]+=sample_arm(mu[I], type_dist)
       N[I]+=1
   end
   recommendation=Best
   return (recommendation,N)
end
