"""
    kl_lucb(mu::Array, delta::Real, rate::Function, dist::String,
        stopping::Symbol=:Chernoff)

KL-LUCB sampling rule by Kaufmann and Kalyanakrishnan [2013],
with Chernoff stopping rule and LUCB stopping rule.
"""
function kl_lucb(mu::Array, delta::Real, rate::Function, dist::String,
    stopping::Symbol=:Chernoff)
    condition = true
   	K=length(mu)
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
      	Best=randmax(Mu)
      	# Find the challenger
      	UCB = zeros(1,K)
      	LCB = dlow(Mu[Best], rate(t,delta)/N[Best], dist)
      	for a in 1:K
	      	if a!=Best
	         	UCB[a]=dup(Mu[a], rate(t,delta)/N[a], dist)
         	end
      	end
      	Challenger=randmax(UCB)
      	# draw both arms
      	t=t+2
      	S[Best] += sample_arm(mu[Best], dist)
      	N[Best] += 1
      	S[Challenger] += sample_arm(mu[Challenger], dist)
      	N[Challenger] += 1
      	# check stopping condition
		if stopping == "Chernoff"
			condition = (Score <= rate(t,delta))
		elseif stopping == "LUCB"
      		condition = (LCB < UCB[Challenger])
		end
      	if (t>1000000)
	      	condition=false
         	Best=0
         	N=zeros(1,K)
      	end
   	end
   	recommendation=Best
   	return (recommendation,N)
end
