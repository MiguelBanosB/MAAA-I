function _y_to_int(y)
    unique_classes = levels(y)
    idx = Dict{eltype(unique_classes), Int}()
    for (i, class) in enumerate(unique_classes)
        idx[class] = i
    end
    classes = Int[]
    for v in y
        push!(classes, idx[v])
    end
    return classes
end

function subject_folds(df::DataFrame; k::Int=5, seed::Int=SEED)
    subjects = unique(df.subject)
    Random.seed!(seed)
    shuffle!(subjects)
    fold_id = Dict{eltype(subjects), Int}()
    for (i, s) in enumerate(subjects)
        fold_id[s] = 1 + (i - 1) % k
    end
    folds = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, k)
    for fold in 1:k
        train_idx = Int[]
        val_idx   = Int[]
        for (i, row) in enumerate(eachrow(df))
            if fold_id[row.subject] == fold
                push!(val_idx, i)
            else
                push!(train_idx, i)
            end
        end
        folds[fold] = (train_idx, val_idx)
    end
    return folds
end

function _anova_f_score(x::AbstractVector{<:Real}, y)
    n = length(x)
    classes = levels(y)
    C = length(classes)
    if C < 2 return 0.0 end
    global_mean = mean(x)
    SST = 0.0
    SSE = 0.0
    for class in classes
        mask = y .== class
        xi = x[mask]
        ni = length(xi)
        if ni == 0 continue end
        m = mean(xi)
        SST += ni * (m - global_mean)^2 
        SSE += sum((xi .- m) .^ 2)
    end
    F = (SST / (C - 1)) / (SSE / (n - C))
    return F
end

function _mutual_info(x::AbstractVector{<:Real}, y_int::Vector{Int}; nbins::Int=10)
    n = length(x)
    if n == 0 return 0.0 end
    xmin, xmax = minimum(x), maximum(x)
    if xmin == xmax return 0.0 end
    
    # Discretización
    xbin = Vector{Int}(undef, n)
    for i in 1:n
        v = x[i]
        if v == xmax
            xbin[i] = nbins
        else
            frac = (v - xmin) / (xmax - xmin)
            b = floor(Int, frac * nbins) + 1
            xbin[i] = clamp(b, 1, nbins)
        end
    end
    
    # Conteo
    num_classes = maximum(y_int)
    counts = zeros(Float64, nbins, num_classes)
    for i in 1:n
        counts[xbin[i], y_int[i]] += 1
    end
    
    # Cálculo MI
    pxy = counts ./ n
    px = vec(sum(pxy, dims=2))
    py = vec(sum(pxy, dims=1))
    mi = 0.0
    for i in 1:nbins
        for j in 1:num_classes
            pij = pxy[i, j]
            if pij > 0.0
                mi += pij * log(pij / (px[i] * py[j]))
            end
        end
    end
    return mi
end

function _select_top_features(scores::AbstractVector{<:Real}, n_features::Int)
    p = length(scores)
    n_sel = min(n_features, p)
    idx_sorted = sortperm(scores, rev=true)
    return idx_sorted[1:n_sel]
end