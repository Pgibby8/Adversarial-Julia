module adv
using Distances
using Knet: Knet, KnetArray, gpu, minibatch
using StatsBase
using Optim
using Knet: SGD, train!, nll, zeroone, relu, dropout
using AutoGrad
import ProgressMeter
using Statistics
using ForwardDiff
# https://arxiv.org/abs/1607.07892
using IterTools

function inf_dist(x, y)
    # Doesn't work with methods requiring gradient descent
    maximum([abs(v) for v in (x - y)])^2
end

function L2_dist(x, y)
    ret = sum((x - y).^2)
    ret
end

function L0_dist(x, y)
    # Doesn't work with methods requiring gradient descent
    count = 0
    for (i, entry) in x
        if entry == y[i]
            count += 1
        end
    end
    return count / length(x)
end

function DSSIM(x, y; bits=8, k₁=0.01, k₂=0.03)
    # Note: when using DSSIM with adv_LBFGS, max_c needs to be much higher than for L2
    μₓ = mean(x)
    μ_y = mean(y)
    σₓ2 = var(x)
    σ_y2 = var(y)
    σₓy = cov(x, y)
    L = 2^bits - 1
    c1 = (k₁*L)^2
    c2 = (k₂*L)^2
    SSIM = (2*μₓ*μ_y + c1)*(2*σₓy + c2) / ((μₓ^2 + μ_y^2 + c1)*(σₓ2 + σ_y2 + c2))
    return (1 - SSIM) / 2.0
end


function predict_class(model, x)
    findmax(model(x, smax=true))[2]
end

function near_search(x0, model, target)
    target == 1 ? off = 2 : off = 1
    best_x = nothing
    best_score = 0
    for (i, x) in enumerate(x0)
        if x == 0
            new_x = x0
            new_x[i] = 1 - x
            pred = model(new_x, smax=true)
            if pred[target] > pred[off] && pred[target] > best_score
                best_x = new_x
                best_score = pred[target]
            end
        end
    end
    return best_x
end

function saliency_map_alt(G, N, target)
    ret = zeros(N)
    for i in 1:N
        T = G[target, i]
        if G[target, i] < 0 || sum(G[:, i]) > T
            ret[i] = 0
        else
            ret[i] = T * abs(sum(G[:, i]) - T)
        end
    end
    ret
end

function saliency_map(G, Γ, target; inc=true)
    M = 0
    p1, p2 = nothing, nothing
    for (p, q) in subsets(Γ, 2)
        α = G[target, p] + G[target, q]
        β = sum(G[:, p]) + sum(G[:, q]) - α
        if inc
            cond = α > 0 && β < 0 && -α*β > M
        else
            cond = α < 0 && β > 0 && -α*β > M
        end
        if cond
            p1, p2 = p, q
            M = -α*β
        end
    end
    p1, p2
end
    
function JSMA(x0, model, target; γ=100, θ=1, smax=true)
#     https://arxiv.org/abs/1511.07528
    inc = θ > 0
    classes = length(target)
    G(x) = ForwardDiff.jacobian(model, x)
    N = length(x0)
    xp = copy(x0)
    max_iter = round(length(x0) * γ / (200 * abs(θ)))
    iter = 0
    Γ = [i for i in 1:length(x0)]
    inc ? filt = 1 : filt = 0
    while predict_class(model, xp) != target && iter < max_iter && length(Γ) > 0 
        p1, p2 = saliency_map(G(xp), Γ, target; inc=inc)
        if p1 == nothing
            iter = max_iter
        else
            xp[p1] = max(min(xp[p1] + θ, 1), 0)
            xp[p2] = max(min(xp[p2] + θ, 1), 0)
            filter!(e->(xp[e]!=filt), Γ)
            iter += 1
        end
    end
    xp
end

function binary_search(f; depth=5, lower=0, upper=100, best=nothing)
    curr = (lower + upper) / 2
    val1, val2 = f(curr)
    if best == nothing || (val2 && val1 < best[2])
        best = (curr, val1, val2)
    end
    if depth == 0
        return best
    else
        if !val2
            return binary_search(f; depth=depth - 1, lower=curr, upper=upper, best=best)
        else
            return binary_search(f; depth=depth - 1, lower=lower, upper=curr, best=best)
        end
    end
end

function adv_LBFGS(x0, model, target; max_c=1, dist=L2_dist)
    """
    Perturbs x0 such that the model will classify it as the target
    Uses an L-BFGS similar to as described by Szegedy et al.
    
    max_c is the maximum value of the c parameter which weights the
    relative importance of the distance term. If the resulting example
    is too far from the original, consider increasing max_c
    
    dist is the distance metric used. Commonly used distance metrics are
    L2 norm, L_infinity norm, and DSSIM (all implemented above). Note that
    different distance metrics require different values of max_c
    """
    NumDimensions = length(x0)
    function error(xp; model=model, c=1, x0=x0, target=target)
        """
        Computes a value to be minimized when generating adversarial examples
        model is the model to fool
        c is a parameter to tun how much to weigh distance from the original
        x0 is the original
        target is the target class

        See "Intriguing properties of neural networks" Szegedy et al (2013)
        """
        for x in 1:length(xp)
            if xp[x] < 0
                xp[x] = 0
            elseif xp[x] > 1
                xp[x] = 1
            end
        end
        c * dist(x0, xp) - log(model(xp, smax=true)[target])
    end
    
    function f(c)
        """
        Given c solves the optimization problem and returns the difference
        from x0 if the produced example is classified as the target
        
        ret1 is the distance from the original image
        ret2 is whether the produced example is classified as the target
        """
        curr_err(x) = error(x; c=c)
        results = optimize(curr_err, x0, LBFGS(); autodiff = :forward)
        x1 = results.minimizer
        ret1 = dist(x0, x1)
        ret2 = findmax(model(x1, smax=true))[2] == target
        return (ret1, ret2)
    end
    
    c = binary_search(f; upper=max_c)[1]
    curr_err(x) = error(x; c=c)
    x1 = optimize(curr_err, copy(x0), LBFGS(); autodiff = :forward).minimizer
    return x1
end

function adv_fast_LBFGS(x0, model, target; c=1, dist=L2_dist)
    """
    Perturbs x0 such that the model will classify it as the target
    Uses an L-BFGS similar to as described by Szegedy et al.
    
    max_c is the maximum value of the c parameter which weights the
    relative importance of the distance term. If the resulting example
    is too far from the original, consider increasing max_c
    
    dist is the distance metric used. Commonly used distance metrics are
    L2 norm, L_infinity norm, and DSSIM (all implemented above). Note that
    different distance metrics require different values of max_c
    """
    NumDimensions = length(x0)
    function error(xp; model=model, c=1, x0=x0, target=target)
        """
        Computes a value to be minimized when generating adversarial examples
        model is the model to fool
        c is a parameter to tun how much to weigh distance from the original
        x0 is the original
        target is the target class

        See "Intriguing properties of neural networks" Szegedy et al (2013)
        """
        for x in 1:length(xp)
            if xp[x] < 0
                xp[x] = 0
            elseif xp[x] > 1
                xp[x] = 1
            end
        end
        c * dist(x0, xp) - log(model(xp, smax=true)[target])
    end
    curr_err(x) = error(x; c=c)
    x1 = optimize(curr_err, copy(x0), LBFGS(); autodiff = :forward).minimizer
    return x1
end

all_sign(x) = [(i >= 0 ? 1 : -1) for i in x]

function adv_fast_gradient_sign(x0, model, target; ϵ=0.1)
    """
    Implements the fast gradient sign method for finding adversarial examples
    Optimized for L_∞ norm, and is fast rather than finding close examples
    """        
    loss(xp) = -log(model(xp, smax=true)[target])
    x1 = x0 - ϵ*all_sign(grad(loss)(x0))
end


function adv_iterative_gradient_sign(x0, model, target; ϵ=10, dist=L2_dist, α=.01)
    """
    Similar to fast gradient sign, just with taking multiple steps of length α
    α = ϵ/steps
    A little slower, but with better results
    """
    x1 = copy(x0)
    loss(xp) = -log(model(xp, smax=true)[target])
    i = 1
    found = false
    δ = dist(x1, x0)
    while δ < ϵ && !found
        x1 = x1 - α*all_sign(grad(loss)(x1))
        δ = dist(x1, x0)
        if predict_class(model, x1) == target
            found = true
        end
    end
    x1
end

function f6(x, model, target; κ=0)
    # A term in the error function described in the Carlini et al. paper
    Z = model(x)
    targetted_val = Z[target]
    untargetted_val = -Inf
    for (i, val) in enumerate(Z)
        if i != target && val > untargetted_val
            untargetted_val = val
        end
    end
    return max(untargetted_val - targetted_val,-κ)
end

function adv_carlini_wagner(x0, model, target; dist=L2_dist, f=f6, c=.1)
    """
    The adversarial generative algorithm described in Carlini et al. 2017
    """
    function error(wp; model=model, c=1, x0=x0, target=target)
        xp = max.(min.(wp, 1), 0)
        return c*dist(x0, xp) + f(xp, model, target)
    end
    w0 = copy(x0)
    curr_err(w) = error(w; c=c)
    w1 = optimize(curr_err, w0, GradientDescent(); autodiff = :forward).minimizer
    x1 = w1 #.5 * (tanh.(w1) .+ 1)
    return x1
end


function softmax(x::Any, T=1)
    ret = exp.(x / T)
    ret / sum(ret)
end

function softmax(x::Array{Float64, 1}, T=1)
    ret = exp.(x / T)
    ret / sum(ret)
end

function softmax(x::Array{Float32, 2}; dims=1, T=1)
    f(x) = softmax(x; dims=dims, T=T)
    mapslices(f, x; dims=dims)
end

function untargetted_attack(model, attack, x, y; N=10)
    for i in 1:N
        if i != y
            xp = attack(x, model, i)
            if predict_class(model, xp) != y
                return xp
            end
        end
    end
    return x
end

function trainresults(model,train; adv_model=nothing, adv_gen=adv_iterative_gradient_sign, o...)
    N = length(model.layers[end].b)
    hit = 0
    miss = 0
    if adv_model != nothing
        # If given a previous model, it will generate adversarial examples and
        # add them to the training set
        dtrn = []
        for (x,y) in train
            new_x = nothing
            new_y = nothing
            for j in 1:length(y)
                for target in 1:N
                    x1 = adv_gen(x[:, j], model, target)
                    if target != y[j] && adv.predict_class(model, x1) == y[j]
                        # Only use valid samples
                        miss += 1
                        x1 = x[:, j]
                    else
                        hit += 1
                    end
                    if new_x == nothing
                        new_x = x1
                        new_y = [y[j]]
                    else
                        new_x = hcat(new_x, x1)
                        new_y = vcat(new_y, y[j])
                    end
                end
            end
            new_x = convert(Array{Float32}, new_x)
            new_y = convert(Array{UInt8}, new_y)
            if size(new_x)[2] != length(new_y)
                s1 = size(new_x)
                s2 = size(new_y)
                println("$s1 $s2")
                throw(DomainError("x and y aren't the same size!"))
            end
            push!(dtrn, (new_x, new_y))
        end
        println("Finished generating adversarial examples")
        println("$hit hits / $miss misses")
    else
        dtrn = train
    end

    train!(model, dtrn; optimizer=SGD(lr=0.1), o...)
end

function cross_entropy(x,target::AbstractArray{<:Integer}; dims=1, average=true, N=10)
    y = zeros(Float32, N, length(target))
    for (i, ind) in enumerate(target)
        y[ind, i] = 1
    end
    lp = logp(x, dims=dims) .* y
    average ? -sum(lp)/length(target) : -sum(lp)
end

function cross_entropy(x,y::AbstractArray{<:Number, 2}; dims=1, average=true, N=10)
    lp = logp(x, dims=dims) .* y
    average ? -sum(lp)/100 : -sum(lp)
end

function cross_entropy(model, data; dims=1, average=true, N=10, o...)
    sum = cnt = 0
    for (x,y) in data
        sum += cross_entropy(model(x; o...), y; dims=dims, average=false)
        cnt += length(y)
    end
    average ? sum / cnt : sum
end


cross_entropy(f, x, y; dims=1, average=true, N=10, o...)=cross_entropy(f(x; o...), y; dims=dims, average=average, N=N)

function trainresults_cross(model, train; o...)
    N = length(model.layers[end].b)
    dtrn = []
    for (x, y) in train
        if length(size(y)) == 1
            push!(dtrn, (x, reshape_y(y, N)))
        else
            push!(dtrn, (x, y))
        end
    end
    train!(model, dtrn; loss=cross_entropy, optimizer=SGD(lr=0.1), o...)
end

struct Linear; w; b; end
(f::Linear)(x) = (f.w * x .+ f.b)

Linear(inputsize::Int,outputsize::Int) = Linear(param(outputsize,inputsize),param0(outputsize))
param(d...; init=xavier, atype=atype())=Param(atype(init(d...)))
param0(d...; atype=atype())=param(d...; init=zeros, atype=atype)
xavier(o,i) = (s = sqrt(2/(i+o)); 2s .* rand(o,i) .- s)
atype()=(gpu() >= 0 ? KnetArray{Float32} : Array{Float32})

struct MLP; layers; end
MLP(h::Int...)=MLP(Linear.(h[1:end-1], h[2:end]))

function (m::MLP)(x; pdrop=0, smax=0, layers=nothing)
    if layers == nothing || layers > length(m.layers)
        range = enumerate(m.layers)
    else
        range = enumerate(m.layers[1:layers])
    end
    for (i, layer) in range
        p = (i <= length(pdrop) ? pdrop[i] : pdrop[end])
        x = dropout(x, p)
        x = layer(x)
        x = (layer == m.layers[end] ? x : relu.(x))
    end
    if smax > 0
        return softmax(x, smax)
    else
        return x
    end
end

function accuracy(model, dtst)
    correct = 0.0
    wrong = 0.0
    for (x,y) in dtst
        pred = model(x; smax=true)
        inds = argmax(pred, dims=1)
        for (i, ind) in enumerate(inds)
            if ind[1] == y[i]
                correct += 1
            else
                wrong += 1
            end
        end
    end
    correct / (correct + wrong)
end

function near_search(x0, model, target)
    best_x = nothing
    best_score = 0
    for (i, x) in enumerate(x0)
        if x != target
            new_x = x0
            new_x[i] = 1 - x
            pred = model(new_x, smax=true)
            if pred[1] > pred[2] && pred[1] > best_score
                best_x = new_x
                best_score = pred[1]
            end
        end
    end
    if best_x == nothing
        return x0
    else
        return best_x
    end
end

function predict_class(model, x0)
    findmax(model(x0, smax=true))[2]
end

function targetted_test(model, examples; alg=adv_fast_gradient_sign)
    successes = 0.0
    total = 0.0
    for example in examples
        x0 = example[1][:, 1]
        y0 = example[2][1]
        target = rand(1:10)
        while target == y0
            target = rand(1:10)
        end
        if findmax(model(x0, smax=true))[2] == y0
            x1 = alg(x0, model, target)
            if predict_class(model, x1) == target
                successes += 1
            end
            total += 1
        end
    end
    successes / total
end

function untargetted_test(model, examples; alg=adv_fast_gradient_sign)
    successes = 0.0
    total = 0.0
    for example in examples
        x0 = example[1][:, 1]
        y0 = example[2][1]
        if findmax(model(x0, smax=true))[2] == y0
            target = 1
            found = false
            while target < 11 && !found
                if target != y0
                    x1 = alg(x0, model, target)
                    if predict_class(model, x1) == target 
                        successes += 1
                        found = true
                    end
                end
                target += 1
            end
            total += 1
        end
    end
    return successes / total
end

end