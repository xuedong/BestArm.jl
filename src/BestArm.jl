module BestArm
    using Distributions
    using PyPlot

    export edp, eba, mpa
    export seq_halving_ref, seq_halving_no_ref
    export succ_reject

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
    include("fixed_confidence/ttps_c.jl")
    include("fixed_confidence/track_stop.jl")
    include("fixed_confidence/ugape_c.jl")
end
