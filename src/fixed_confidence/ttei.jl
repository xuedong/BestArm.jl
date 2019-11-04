function ttei(mu::Array, delta::Real, rate::Function, dist::String,
	frac::Real = 0.5, alpha::Real = 1, beta::Real = 1, stopping::String = "chernoff")
	# Chernoff stopping rule combined with the TTTS sampling rule of [Russo, 2016]
	condition = true
   	K = length(mu)
   	N = zeros(1,K)
   	S = zeros(1,K)
   	# initialization
   	for a in 1:K
      	N[a]=1
      	S[a]=sample(mu[a], dist)
   	end
   	t=K
   	Best=1
   	while (condition)
      	Mu=S./N
      	# Empirical best arm
      	Best=randmax(Mu)
      	# Compute the stopping statistic
      	NB=N[Best]
      	SB=S[Best]
      	muB=SB/NB
      	MuMid=(SB.+S)./(NB.+N)
      	Index=collect(1:K)
      	deleteat!(Index,Best)
      	Score=minimum([NB*d(muB, MuMid[i], dist)+N[i]*d(Mu[i], MuMid[i], dist) for i in Index])
      	if (Score > rate(t,delta))
         	# stop
         	condition=false
      	elseif (t >1000000)
         	condition=false
         	Best=0
         	print(N)
         	print(S)
         	N=zeros(1,K)
      	else
         	EI=zeros(K)
         	for a=1:K
	         	#TS[a]=rand(Beta(alpha+S[a], beta+N[a]-S[a]), 1)[1]
            	SigmaI = sqrt(sigma^2/N[a])
            	x = (Mu[a] - muB)/SigmaI
            	EI[a] = SigmaI * f(x)
         	end
         	#println(EI)
         	I = argmax(EI)
         	if (rand()>frac)
            	EII=zeros(K)
            	condition=true
            	for b=1:K
               		if b == I
                  		EII[b] = 0
               		else
                  		SigmaI = sqrt(sigma^2/N[I]+sigma^2/N[b])
                  		x = (Mu[b] - Mu[I])/SigmaI
                  		EII[b] = SigmaI * f(x)
               		end
            	end
            	#println(EII)
            	I = argmax(EII)
         	end
         	# draw arm I
	      	t+=1
	      	S[I]+=sample(mu[I], dist)
	      	N[I]+=1
	   	end
   	end
   	recommendation=Best
   	return (recommendation,N)
end
