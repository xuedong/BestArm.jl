function chernoff_kl_lucb(mu::Array, delta::Real, rate::Function)
    # Chernoff stopping rule, KL-LUCB sampling rule
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
        Score=minimum([NB*d(muB, MuMid[i], dist)+N[i]*d(Mu[i], MuMid[i], dist) for i in Index])
        # Find the challenger
        UCB=zeros(1,K)
        LCB=dlow(Mu[Best], rate(t,0,delta)/N[Best], dist)
        for a in 1:K
            if a!=Best
                UCB[a]=dup(Mu[a], rate(t,0,delta)/N[a], dist)
            end
        end
        Ind=find(UCB.==maximum(UCB))
        Challenger=Ind[round(Int,floor(length(Ind)*rand())+1)]
        # draw both arms
        t=t+2
        S[Best]+=sample_arm(mu[Best], dist)
        N[Best]+=1
        S[Challenger]+=sample_arm(mu[Challenger], dist)
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
function kl_lucb(mu::Array, delta::Real, rate::Function)
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
        Ind=find(Mu.==maximum(Mu))
        # Empirical best arm
        Best=Ind[round(Int,floor(length(Ind)*rand())+1)]
        # Find the challenger
        UCB=zeros(1,K)
        LCB=dlow(Mu[Best], rate(t,N[Best],delta)/N[Best], dist)
        for a in 1:K
            if a!=Best
                UCB[a]=dup(Mu[a], rate(t,N[a],delta)/N[a], dist)
            end
        end
        Ind=find(UCB.==maximum(UCB))
        Challenger=Ind[round(Int,floor(length(Ind)*rand())+1)]
        # draw both arms
        t=t+2
        S[Best]+=sample_arm(mu[Best], dist)
        N[Best]+=1
        S[Challenger]+=sample_arm(mu[Challenger], dist)
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
