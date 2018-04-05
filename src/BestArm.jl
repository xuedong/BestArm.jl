module BestArm
    using Distributions
    using PyPlot

    export edp, eba, mpa
    export compute_regrets

    export seq_halving_ref, seq_halving_no_ref
    export succ_reject
    export uniform
    export ucbe, ucbe_adaptive
    export ugape_b, ugape_b_adaptive

    export chernoff_bc, chernoff_bc2
    export chernoff_ttts
    export chernoff_racing, kl_racing
    export chernoff_target
    export chernoff_kl_lucb, kl_lucb
    export track_stop, track_stop2
    export ugape_c

    include("arms.jl")
    include("utils.jl")
    include("kl_functions.jl")

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
end