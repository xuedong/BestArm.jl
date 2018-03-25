# Algorithms for Best Arm Identification in Exponential Family Bandit Models in the Fixed-Confidence Setting

# The nature of the distribution should be precised by choosing a value for typeDistribution before including the current file

# All the algorithms take the following input
# mu : vector of arms means
# delta : risk level
# rate : the exploration rate (a function)

using Distributions
using PyPlot

include("Arms.jl")
include("KLFunctions.jl")

# OPTIMAL ALGORITHMS

function TrackAndStop(mu,delta,rate)
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
       if (length(IndMax)>1)
          # if multiple maxima, draw one them at random
          I = Best
       else
       	  # compute the stopping statistic
       	  NB=N[Best]
          SB=S[Best]
          muB=SB/NB
          MuMid=(SB+S)./(NB+N)
          Index=collect(1:K)
          splice!(Index,Best)
          Score=minimum([NB*d(muB, MuMid[i], type_dist)+N[i]*d(Mu[i], MuMid[i], type_dist) for i in Index])
          if (Score > rate(t,0,delta))
            # stop
            condition=false
          elseif (t >10000000)
            # stop and outputs (0,0)
            condition=false
            Best=0
            print(N)
            print(S)
            N=zeros(1,K)
          else
            if (minimum(N) <= max(sqrt(t) - K/2,0))
               # forced exploration
               I=indmin(N)
            else
               # continue and sample an arm
	       val,Dist=OptimalWeights(Mu,1e-11)
               # choice of the arm
               I=indmax(Dist-N/t)
            end
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


function TrackAndStop2(mu,delta,rate)
  # Uses a Tracking of the cummulated sum
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
       Score=minimum([NB*d(muB, MuMid[i], type_dist)+N[i]*d(Mu[i], MuMid[i], type_dist) for i in Index])
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
       S[I]+=sample_arm(mu[I], type_dist)
       N[I]+=1
   end
   recommendation=Best
   return (recommendation,N)
end


# Chernoff stopping rule coupled with different sampling rules


function ChernoffTarget(mu,delta,rate,Target=ones(1,length(mu))/length(mu))
  # sampling rule : choose arm maximizing (Target - empirical proportion)
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
       Ind=find(Mu.==maximum(Mu))
       # Empirical best arm
       Best=Ind[floor(Int,length(Ind)*rand())+1]
       # Compute the stopping statistic
       NB=N[Best]
       SB=S[Best]
       muB=SB/NB
       MuMid=(SB+S)./(NB+N)
       Index=collect(1:K)
       splice!(Index,Best)
       Score=minimum([NB*d(muB, MuMid[i], type_dist)+N[i]*d(Mu[i], MuMid[i], type_dist) for i in Index])
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
	    I=indmax(Target-N/t)
	    t+=1
	    S[I]+=sample_arm(mu[I], type_dist)
	    N[I]+=1
	end
   end
   recommendation=Best
   return (recommendation,N)
end



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


function ChernoffKLLUCB(mu,delta,rate)
  # Chernoff stopping rule, KL-LUCB sampling rule
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
       Ind=find(Mu.==maximum(Mu))
       # Empirical best arm
       Best=Ind[round(Int,floor(length(Ind)*rand())+1)]
       # Compute the stopping statistic
       NB=N[Best]
       SB=S[Best]
       muB=SB/NB
       MuMid=(SB+S)./(NB+N)
       Index=collect(1:K)
       splice!(Index,Best)
       Score=minimum([NB*d(muB, MuMid[i], type_dist)+N[i]*d(Mu[i], MuMid[i], type_dist) for i in Index])
       # Find the challenger
       UCB=zeros(1,K)
       LCB=dlow(Mu[Best], rate(t,0,delta)/N[Best], type_dist)
       for a in 1:K
	  if a!=Best
	     UCB[a]=dup(Mu[a], rate(t,0,delta)/N[a], type_dist)
          end
       end
       Ind=find(UCB.==maximum(UCB))
       Challenger=Ind[round(Int,floor(length(Ind)*rand())+1)]
       # draw both arms
       t=t+2
       S[Best]+=sample_arm(mu[Best], type_dist)
       N[Best]+=1
       S[Challenger]+=sample_arm(mu[Challenger], type_dist)
       N[Challenger]+=1
       # check stopping condition
       condition=(Score <= rate(t,0,delta))
       if (t>1000000)
	  condition=false
          Best=0
          N=zeros(1,K)
       end
   end
   recommendation=Best
   return (recommendation,N)
end





# KL-LUCB [Kaufmann and Kalyanakrishnan 2013]

function KLLUCB(mu,delta,rate)
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
       Ind=find(Mu.==maximum(Mu))
       # Empirical best arm
       Best=Ind[round(Int,floor(length(Ind)*rand())+1)]
       # Find the challenger
       UCB=zeros(1,K)
       LCB=dlow(Mu[Best], rate(t,N[Best],delta)/N[Best], type_dist)
       for a in 1:K
	  if a!=Best
	     UCB[a]=dup(Mu[a], rate(t,N[a],delta)/N[a], type_dist)
          end
       end
       Ind=find(UCB.==maximum(UCB))
       Challenger=Ind[round(Int,floor(length(Ind)*rand())+1)]
       # draw both arms
       t=t+2
       S[Best]+=sample_arm(mu[Best], type_dist)
       N[Best]+=1
       S[Challenger]+=sample_arm(mu[Challenger], type_dist)
       N[Challenger]+=1
       # check stopping condition
       condition=(LCB < UCB[Challenger])
       if (t>1000000)
	  condition=false
          Best=0
          N=zeros(1,K)
       end
   end
   recommendation=Best
   return (recommendation,N)
end





# Racing algorithms

function ChernoffRacing(mu,delta,rate)
  condition = true
  K=length(mu)
  N = zeros(1,K)
  S = zeros(1,K)
  # initialization
  for a in 1:K
      N[a]=1
      S[a]=sample_arm(mu[a], type_dist)
  end
  round=1
  t=K
  Best=1
  Remaining=collect(1:K)
  while (length(Remaining)>1)
       # Drawn all remaining arms
       for a in Remaining
	  S[a]+=sample_arm(mu[a], type_dist)
	  N[a]+=1
       end
       round+=1
       t+=length(Remaining)
       # Check whether the worst should be removed
       Mu=S./N
       MuR=Mu[Remaining]
       MuBest=maximum(MuR)
       IndBest=find(MuR.==MuBest)[1]
       IndBest=IndBest[floor(Int,rand()*length(IndBest))+1]
       Best=Remaining[IndBest]
       MuWorst=minimum(MuR)
       IndWorst=find(MuR.==MuWorst)[1]
       IndWorst=IndWorst[floor(Int,rand()*length(IndWorst))+1]
       if (round*(d(MuBest, (MuBest+MuWorst)/2, type_dist)+d(MuWorst, (MuBest+MuWorst)/2, type_dist)) > rate(t, 0, delta))
          # remove Worst arm
          splice!(Remaining,IndWorst)
       end
       if (t>1000000)
	  Remaining=[]
          Best=0
          N=zeros(1,K)
       end
   end
   recommendation=Best
   return (recommendation,N)
end



function KLRacing(mu,delta,rate)
  condition = true
  K=length(mu)
  N = zeros(1,K)
  S = zeros(1,K)
  # initialization
  for a in 1:K
      N[a]=1
      S[a]=sample_arm(mu[a], type_dist)
  end
  round=1
  t=K
  Best=1
  Remaining=collect(1:K)
  while (length(Remaining)>1)
       # Drawn all remaining arms
       for a in Remaining
	  S[a]+=sample_arm(mu[a], type_dist)
	  N[a]+=1
       end
       round+=1
       t+=length(Remaining)
       # Check whether the worst should be removed
       Mu=S./N
       MuR=Mu[Remaining]
       MuBest=maximum(MuR)
       IndBest=find(MuR.==MuBest)[1]
       Best=IndBest[floor(Int,rand()*length(IndBest))+1]
       Best=Remaining[Best]
       MuWorst=minimum(MuR)
       IndWorst=find(MuR.==MuWorst)[1]
       IndWorst=IndWorst[floor(Int,rand()*length(IndWorst))+1]
       if (dlow(MuBest, rate(t,round,delta)/round, type_dist) > dup(MuWorst, rate(t,round,delta)/round, type_dist))
          # remove Worst arm
          splice!(Remaining,IndWorst)
       end
       if (t>1000000)
	  Remaining=[]
          Best=0
          N=zeros(1,K)
       end
   end
   recommendation=Best
   return (recommendation,N)
end


# UGapE [Gabillon et al., 2012]

function UGapEC(mu,delta,rate)
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
       Ind=find(Mu.==maximum(Mu))
       # Empirical best arm
       Best=Ind[round(Int,floor(length(Ind)*rand())+1)]
       # Find the challenger
       UCB=zeros(1,K)
       LCB=zeros(1,K)
       for a in 1:K
          UCB[a]=dup(Mu[a], rate(t,N[a],delta)/N[a], type_dist)
          LCB[a]=dlow(Mu[a], rate(t,N[a],delta)/N[a], type_dist)
       end
       B=zeros(1,K)
       for a in 1:K
          Index=collect(1:K)
          splice!(Index,a)
          B[a] = maximum(UCB[Index])-LCB[a]
       end
       Value=minimum(B)
       Best=indmin(B)
       UCB[Best]=0
       Challenger=indmax(UCB)
       # choose which arm to draw
       t=t+1
       I=(N[Best]<N[Challenger])?Best:Challenger
       S[I]+=sample_arm(mu[I], type_dist)
       N[I]+=1
       # check stopping condition
       condition=(Value > 0)
       if (t>1000000)
	  condition=false
          Best=0
          N=zeros(1,K)
       end
   end
   recommendation=Best
   return (recommendation,N)
end


# Pure-Exploration Thompson Sampling [Russo, 2016] + Chernoff Stopping rule

function ChernoffPTS(mu,delta,rate,frac,alpha=1,beta=1)
  # Chernoff stopping rule combined with the PTS sampling rule
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
       Ind=find(Mu.==maximum(Mu))
       # Empirical best arm
       Best=Ind[floor(Int,length(Ind)*rand())+1]
       # Compute the stopping statistic
       NB=N[Best]
       SB=S[Best]
       muB=SB/NB
       MuMid=(SB+S)./(NB+N)
       Index=collect(1:K)
       splice!(Index,Best)
       Score=minimum([NB*d(muB, MuMid[i], type_dist)+N[i]*d(Mu[i], MuMid[i], type_dist) for i in Index])
       if (Score > rate(t,0,delta))
            # stop
            condition=false
       elseif (t >1000000)
            condition=false
            Best=0
            print(N)
            print(S)
            N=zeros(1,K)
       else
            TS=zeros(K)
            for a=1:K
				TS[a]=rand(Beta(alpha+S[a], beta+N[a]-S[a]), 1)[1]
            end
            I = indmax(TS)
            if (rand()>frac)
               	J=I
               	condition=true
               	while (I==J)
                  	TS=zeros(K)
                  	for a=1:K
	             		TS[a]=rand(Beta(alpha+S[a], beta+N[a]-S[a]), 1)[1]
                  	end
                  	J = indmax(TS)
               	end
               	I=J
            end
            # draw arm I
	    	t+=1
	    	S[I]+=sample_arm(mu[I], type_dist)
	    	N[I]+=1
		end
   	end
   	recommendation=Best
   	return (recommendation,N)
end
