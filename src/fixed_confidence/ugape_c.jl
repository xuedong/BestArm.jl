function ugape_c(mu::Array, delta::Real, rate::Function)
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
        LCB=zeros(1,K)
        for a in 1:K
            UCB[a]=dup(Mu[a], rate(t,N[a],delta)/N[a], dist)
            LCB[a]=dlow(Mu[a], rate(t,N[a],delta)/N[a], dist)
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
        S[I]+=sample_arm(mu[I], dist)
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
