using Knet
include("../Hyperband.jl")

function hyperband_demo()
    include(Knet.dir("examples","fashion-mnist","fashion-mnist.jl"))
    best = (Inf,)
    neval = 0

    function getloss(config,epochs)
        neval += 1
        winit,lr,hidden = config
        epochs = round(Int,epochs)
        w = FashionMNIST.main("--winit $winit --lr $lr --hidden $hidden --epochs $epochs --seed 1 --fast")
        corr,loss = FashionMNIST.accuracy(w,FashionMNIST.dtst)
        println((:epochs,epochs,:corr,corr,:loss,loss,:winit,winit,:lr,lr,:hidden,hidden))
        if loss < best[1]
            best = (loss, config, epochs)
        end
        return loss
    end

    function getconfig()
        winit = 0.001^rand()
        lr = 0.001^rand()
        hidden = 16 + floor(Int, 10000^rand())
        return (winit,lr,hidden)
    end

    Hyperband.hyperband(getconfig, getloss)
    println((:neval,neval,:minloss,best[1],:epochs,best[3],:winit,best[2][1],:lr,best[2][2],:hidden,best[2][3]))
end
