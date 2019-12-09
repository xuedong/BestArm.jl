module BestArm
    using Distributions
    using LinearAlgebra
    using Documenter
    using HCubature
    using StatsFuns
    using ProgressMeter

    export edp, eba, mpa
    export compute_regrets, compute_regrets_reservoir
    # export memuse
    export compute_observation, compute_transportation
    export update_design_inverse, update_square_root

    export optimal_weights
    export gamma_beta

    export at_lucb
    export ts
    export ttts, parallel_ttts, ttts_infinite, ttts_dynamic
    export ttps, parallel_ttps

    export seq_halving_ref, seq_halving_no_ref, seq_halving_infinite, hyperband
    export succ_reject
    export ucbe, ucbe_adaptive
    export ugape_b, ugape_b_adaptive
    export uniform

    export best_challenger
    export kl_lucb
    export racing
    export t3c, t3c_optimal
    export target
    export d_tracking, c_tracking
    export ttei
    export ttts_c
    export ugape_c
    export uniform_c

    export siri

    export l_t3c
    export l_t3s
    export lingape

    include("arms.jl")
    include("utils.jl")
    include("linear/utils_linear.jl")
    include("kl_functions.jl")
    include("reservoirs.jl")
    include("weights.jl")

    include("anytime/at_lucb.jl")
    include("anytime/ts.jl")
    include("anytime/ttps.jl")
    include("anytime/ttts.jl")
    include("anytime/ttvs.jl")

    include("fixed_budget/seq_halv.jl")
    include("fixed_budget/succ_rej.jl")
    include("fixed_budget/ucbe.jl")
    include("fixed_budget/ugape_b.jl")
    include("fixed_budget/uniform.jl")

    include("fixed_confidence/best_challenger.jl")
    include("fixed_confidence/kl_ucb.jl")
    include("fixed_confidence/racing.jl")
    include("fixed_confidence/t3c.jl")
    include("fixed_confidence/target.jl")
    include("fixed_confidence/track_stop.jl")
    include("fixed_confidence/ttei.jl")
    include("fixed_confidence/ttts_c.jl")
    include("fixed_confidence/ugape_c.jl")
    include("fixed_confidence/uniform_c.jl")

    include("infinite/siri.jl")

    include("linear/l_t3c.jl")
    include("linear/l_t3s.jl")
    include("linear/lingape.jl")
    include("linear/xy_adaptive.jl")
    include("linear/xy_static.jl")
end
