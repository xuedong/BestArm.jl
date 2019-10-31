module BestArm
    using Distributions
    using Documenter
    using HCubature
    using StatsFuns
    using ProgressMeter

    export edp, eba, mpa
    export compute_regrets, compute_regrets_reservoir
    # export memuse

    export optimal_weights
    export gamma_beta

    export seq_halving_ref, seq_halving_no_ref, seq_halving_infinite, hyperband
    export succ_reject
    export uniform
    export ucbe, ucbe_adaptive
    export ugape_b, ugape_b_adaptive

    export best_challenger, best_challenger_bis, best_challenger_ts
    export ttts_c
    export ttei
    export t3c
    export racing, kl_racing
    export target
    export kl_lucb
    export d_tracking, c_tracking
    export ugape_c

    export ts
    export ttts, parallel_ttts, ttts_infinite, ttts_dynamic
    export ttps, parallel_ttps
    export at_lucb

    export siri

    include("arms.jl")
    include("utils.jl")
    include("kl_functions.jl")
    include("reservoirs.jl")
    include("weights.jl")

    include("fixed_budget/ucbe.jl")
    include("fixed_budget/seq_halv.jl")
    include("fixed_budget/succ_rej.jl")
    include("fixed_budget/ugape_b.jl")
    include("fixed_budget/uniform.jl")

    include("fixed_confidence/kl_ucb.jl")
    include("fixed_confidence/racing.jl")
    include("fixed_confidence/target.jl")
    include("fixed_confidence/best_challenger.jl")
    include("fixed_confidence/ttts_c.jl")
    include("fixed_confidence/track_stop.jl")
    include("fixed_confidence/ugape_c.jl")

    include("anytime/ttps.jl")
    include("anytime/ttts.jl")
    include("anytime/ttvs.jl")
    include("anytime/ts.jl")
    include("anytime/at_lucb.jl")

    include("infinite/siri.jl")
end
