function chernoff_bc(mu::Array, delta::Real, rate::Function, dist::String)
    # Chernoff stopping rule, sampling based on the "best challenger"
    # described in experimental section of Garivier and Kaufmann [2016]
    condition = true
    K = length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a] = 1
        S[a] = sample_arm(mu[a], dist)
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
                score=NB*d(MuB, MuMid[i], dist)+N[i]*d(Mu[i], MuMid[i], dist)
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
                I=(NB/(NB+N[Challenger]) < Dist[Best]/(Dist[Best]+Dist[Challenger])) ? Best : Challenger
                #I=(d(MuB, MuMid[Challenger], dist)>d(Mu[Challenger], MuMid[Challenger], dist))?Best:Challenger
                #I=(N[Best]<N[Challenger])?Best:Challenger
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


function chernoff_bc2(mu::Array, delta::Real, rate::Function, dist::String)
    # Chernoff stopping rule + alternative choice between the empirical best and its "challenger"
    # Faster, requires no computation of Optimal Weights
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
    while (condition)
        Mu=S./N
        # Empirical best arm
        IndMax = find(Mu .== maximum(Mu))
        Best = IndMax[floor(Int, length(IndMax)*rand())+1]
        I = 1
        # Compute the stopping statistic
        NB = N[Best]
        SB = S[Best]
        MuB = SB/NB
        MuMid = (SB+S)./(NB+N)
        Challenger = 1
        Score = Inf
        for i = 1:K
            if i != Best
                score = NB * d(MuB, MuMid[i], dist) + N[i] * d(Mu[i], MuMid[i], dist)
                if (score < Score)
                    Challenger = i
                    Score = score
                end
            end
        end
        if (Score > rate(t, 0, delta))
            # stop
            condition=false
        elseif (t > 1000000)
            # stop and return (0,0)
            condition = false
            Best = 0
            print(N)
            print(S)
            N = zeros(1,K)
        else
            # continue and sample an arm
            if (minimum(N) <= max(sqrt(t) - K/2,0))
                # forced exploration
                I=indmin(N)
            else
                # choose between the arm and its Challenger
                I = (N[Best]<N[Challenger]) ? Best : Challenger
                #I=(d(MuB, MuMid[Challenger], dist)>d(Mu[Challenger], MuMid[Challenger], dist))?Best:Challenger
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


function ChernoffBC(mu,delta,rate,forced_explo=false,TS=false,alpha=1,beta=1)
   # Chernoff stopping rule,  sampling based on the "best challenger"
   # (different tie breaking rule compared to the one described in [Garivier and Kaufmann 2016])
   continuing = true
   K=length(mu)
   N = zeros(1,K)
   S = zeros(1,K)
   # initialization
   for a in 1:K
      N[a]=1
      S[a]=sample(mu[a])
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
      Score=minimum([NB*d(MuB,MuMid[i])+N[i]*d(Mu[i],MuMid[i]) for i in 1:K if i!=Best])
      # compute the best arm and the challenger
      if TS
         for a=1:K
            if typeDistribution == "Gaussian"
               Mu[a] = rand(Normal(S[a] / N[a], sigma / sqrt(N[a])), 1)[1]
            elseif typeDistribution == "Bernoulli"
               Mu[a]=rand(Beta(alpha+S[a], beta+N[a]-S[a]), 1)[1]
            end
         end
         Best=randmax(Mu)
         NB=N[Best]
         MuB=Mu[Best]
         MuMid=(NB*MuB.+N.*Mu)./(NB.+N)
      end
      Challenger=1
      NewScore=Inf
      for i=1:K
	      if i!=Best
            score=NB*d(MuB,MuMid[i])+N[i]*d(Mu[i],MuMid[i])
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
	      if (forced_explo)&&(minimum(N) <= max(sqrt(t) - K/2,0))
            # forced exploration
            I=randmax(-N)
         else
            # choose between the arm and its Challenger
            I=(d(MuB,MuMid[Challenger])>d(Mu[Challenger],MuMid[Challenger])) ? Best : Challenger
         end
      end
      # draw the arm
      t+=1
      S[I]+=sample(mu[I])
      N[I]+=1
   end
   return TrueBest,N
end
