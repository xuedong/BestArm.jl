function uniform_c(mu,delta,rate,alpha=1,beta=1)
   # Uniform sampling rule
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
   Best = 1
   while continuing
      Mu=S./N
      # Empirical best arm
      Best=randmax(Mu)
      # Compute the stopping statistic
      NB=N[Best]
      SB=S[Best]
      MuB=SB/NB
      MuMid=(SB.+S)./(NB.+N)
      Score=minimum([NB*d(MuB,MuMid[i])+N[i]*d(Mu[i],MuMid[i]) for i in 1:K if i!=Best])
      I=1
      if (Score > rate(t,delta))
         # stop
         continuing=false
      elseif (t >1e7)
         # stop and return (0,0)
         continuing=false
         Best=0
         print(N)
         print(S)
         N=zeros(1,K)
      else
         # continue and sample an arm
	      I = rand(1:K)
      end
      # draw the arm
      t+=1
      S[I]+=sample(mu[I])
      N[I]+=1
   end
   return Best,N
end
