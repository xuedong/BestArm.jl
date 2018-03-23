# R, η = 27, 3
# smax = log(R) / log(η)
# B = (smax + 1) * R

module Hyperband

using Logging

function budget(maxresource, reduction = 3)
  smax = floor(Int, log(maxresource) / log(reduction))
  B = (smax + 1) * maxresource
  @parallel (+) for s in smax:-1:0
      n = ceil(Int, (B / maxresource) * (reduction^s / (s + 1)))
      r = maxresource / reduction^s
      n * r * s
  end
end

function resource(maxbudget, reduction = 3)
  indmin(abs(maxbudget - budget(i, reduction)) for i in 1:100000)
end

function halving(getconfig, getloss, n, r, reduction,  s)
    best = (Inf, nothing, nothing)
    T = [ getconfig() for i in 1:n ]
    for i in 0:s-1
        ni = floor(Int, n / reduction^i)
        ri = Int(r * reduction^i)
        L = [ getloss(t, ri) for t in T ]
        l = sortperm(L); l1 = l[1]
        L[l1] < best[1] && (best = (L[l1], ri, T[l1]))
        T = T[l[1:floor(Int, ni / reduction)]]
        report(best) |> debug
    end
    report(best) |> info
    return best
end

export hyperband
function hyperband(getconfig, getloss, maxbudget = 27, reduction = 3)
    maxresource = resource(maxbudget, reduction)
    info("max_budget = ", budget(maxresource), "  max_resource = ", maxresource)

    smax = floor(Int, log(maxresource) / log(reduction))
    B = (smax + 1) * maxresource
    best = (Inf,)
    for s in smax:-1:0
        n = ceil(Int, (B / maxresource) * (reduction^s / (s + 1)))
        r = maxresource / reduction^s
        curr = halving(getconfig, getloss, n, r, reduction, s)
        if curr[1] < best[1]; (best = curr); end
    end
    info("\n", "Hyperband is completed", "\n", report(best))
    return best
end

function report(best)
  loss, epoch, config = best
  string("loss = ", loss, " epoch = ", epoch)
end

end
