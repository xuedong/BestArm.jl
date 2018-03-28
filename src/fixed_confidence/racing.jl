function chernoff_racing(mu::Array, delta::Real, rate::Function)
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample_arm(mu[a], dist)
    end
    round=1
    t=K
    Best=1
    Remaining=collect(1:K)
    while (length(Remaining)>1)
        # Drawn all remaining arms
        for a in Remaining
            S[a]+=sample_arm(mu[a], dist)
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
        if (round*(d(MuBest, (MuBest+MuWorst)/2, dist)+d(MuWorst, (MuBest+MuWorst)/2, dist)) > rate(t, 0, delta))
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

function kl_racing(mu::Array, delta::Real, rate::Function)
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample_arm(mu[a], dist)
    end
    round=1
    t=K
    Best=1
    Remaining=collect(1:K)
    while (length(Remaining)>1)
        # Drawn all remaining arms
        for a in Remaining
            S[a]+=sample_arm(mu[a], dist)
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
        if (dlow(MuBest, rate(t,round,delta)/round, dist) > dup(MuWorst, rate(t,round,delta)/round, dist))
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
