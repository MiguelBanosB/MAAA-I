# ==============================================================================
# 1. SCALER
# ==============================================================================
struct MyMinMaxScaler <: MMI.Unsupervised end

function MMI.fit(model::MyMinMaxScaler, verbosity::Int, X)
    Xmat = MMI.matrix(X)
    mins = mapslices(minimum, Xmat; dims=1)[1, :]
    maxs = mapslices(maximum, Xmat; dims=1)[1, :]
    return (mins=mins, maxs=maxs), nothing
end

function MMI.transform(model::MyMinMaxScaler, cache, X)
    Xmat = MMI.matrix(X)
    ranges = cache.maxs .- cache.mins
    ranges[ranges .== 0] .= 1
    return (Xmat .- cache.mins) ./ ranges
end

# ==============================================================================
# 2. WRAPPERS DE FILTRADOS
# ==============================================================================

# -- ANOVA --
struct MyANOVAFilter <: MMI.Supervised; n_features::Int; end
MyANOVAFilter(; n_features::Int=100) = MyANOVAFilter(n_features)

function MMI.fit(model::MyANOVAFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_cat = coerce(y, Multiclass)
    scores = [_anova_f_score(view(Xmat, :, j), y_cat) for j in 1:size(Xmat, 2)]
    selected = _select_top_features(scores, model.n_features)
    return (selected=selected, scores=scores), nothing
end

function MMI.transform(model::MyANOVAFilter, cache, X)
    Xmat = MMI.matrix(X)
    return Xmat[:, cache.selected]
end

# -- PEARSON --
struct MyPearsonFilter <: MMI.Supervised; n_features::Int; end
MyPearsonFilter(; n_features::Int=100) = MyPearsonFilter(n_features)

function MMI.fit(model::MyPearsonFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_float = Float64.(_y_to_int(coerce(y, Multiclass)))
    scores = [abs(cor(view(Xmat, :, j), y_float)) for j in 1:size(Xmat, 2)]
    selected = _select_top_features(scores, model.n_features)
    return (selected=selected, scores=scores), nothing
end

function MMI.transform(model::MyPearsonFilter, cache, X)
    return MMI.matrix(X)[:, cache.selected]
end

# -- SPEARMAN --
struct MySpearmanFilter <: MMI.Supervised; n_features::Int; end
MySpearmanFilter(; n_features::Int=100) = MySpearmanFilter(n_features)

function MMI.fit(model::MySpearmanFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_float = Float64.(_y_to_int(coerce(y, Multiclass)))
    scores = [abs(corspearman(view(Xmat, :, j), y_float)) for j in 1:size(Xmat, 2)]
    selected = _select_top_features(scores, model.n_features)
    return (selected=selected, scores=scores), nothing
end
MMI.transform(m::MySpearmanFilter, c, X) = MMI.matrix(X)[:, c.selected]

# -- KENDALL --
struct MyKendallFilter <: MMI.Supervised; n_features::Int; end
MyKendallFilter(; n_features::Int=100) = MyKendallFilter(n_features)

function MMI.fit(model::MyKendallFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_float = Float64.(_y_to_int(coerce(y, Multiclass)))
    scores = [abs(corkendall(view(Xmat, :, j), y_float)) for j in 1:size(Xmat, 2)]
    selected = _select_top_features(scores, model.n_features)
    return (selected=selected, scores=scores), nothing
end
MMI.transform(m::MyKendallFilter, c, X) = MMI.matrix(X)[:, c.selected]

# -- MI --
struct MyMIFilter <: MMI.Supervised; n_features::Int; end
MyMIFilter(; n_features::Int=100) = MyMIFilter(n_features)

function MMI.fit(model::MyMIFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_int = _y_to_int(coerce(y, Multiclass))
    scores = [_mutual_info(view(Xmat, :, j), y_int) for j in 1:size(Xmat, 2)]
    selected = _select_top_features(scores, model.n_features)
    return (selected=selected, scores=scores), nothing
end
MMI.transform(m::MyMIFilter, c, X) = MMI.matrix(X)[:, c.selected]

# -- RFE --
struct MyRFEFilter <: MMI.Supervised; n_features::Int; end
MyRFEFilter(; n_features::Int=100) = MyRFEFilter(n_features)

function MMI.fit(model::MyRFEFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_float = Float64.(_y_to_int(coerce(y, Multiclass)))
    n, p = size(Xmat)
    original_indices = collect(1:p)
    
    while length(original_indices) > model.n_features
        Xi = Xmat[:, original_indices]
        w = Xi \ y_float # Regresión Lineal por Mínimos Cuadrados
        importance = abs.(w)
        
        n_remaining = length(original_indices)
        n_to_remove = max(1, ceil(Int, n_remaining * 0.5)) # Eliminar 50%
        
        # Eliminar los menos importantes
        idx_sorted = sortperm(importance)
        deleteat!(original_indices, idx_sorted[1:n_to_remove])
    end
    return (selected=original_indices, scores=zeros(p)), nothing
end
MMI.transform(m::MyRFEFilter, c, X) = MMI.matrix(X)[:, c.selected]

# ==============================================================================
# 3. DATOS DE LOS FILTROS (Evitamos errores de verificación de tipos)
# ==============================================================================
for T in [MyANOVAFilter, MyPearsonFilter, MySpearmanFilter, MyKendallFilter, MyMIFilter, MyRFEFilter]
    MMI.input_scitype(::Type{<:T}) = MMI.Table(MMI.ScientificTypes.Continuous) # Tabla de valores continuos como entrada
    MMI.target_scitype(::Type{<:T}) = AbstractVector{<:MMI.Finite} # Vector de etiquetas finitas como objetivo
end

# ==============================================================================
# 4. MANUAL PIPELINE (SOLUCIÓN DEFINITIVA A METHODERROR)
# ==============================================================================
mutable struct ManualPipeline{F, R, M} <: MMI.Probabilistic
    scaler::MyMinMaxScaler
    filter::F
    reduction::R
    model::M
end

function MMI.fit(p::ManualPipeline, verbosity::Int, X, y)
    # 1. Scaler
    fr_s, _, _ = MMI.fit(p.scaler, verbosity, X) # Entrenamos el scaler 
    Xt = MMI.transform(p.scaler, fr_s, X) # Transformamos X
    
    # 2. Filter
    fr_f, _, _ = MMI.fit(p.filter, verbosity, Xt, y) # Entrenamos el filtro con la X escalada y la y (supervisado)
    Xt = MMI.transform(p.filter, fr_f, Xt)
    
    # 3. Reduction (Si la hay, reduce)
    fr_r = nothing
    if !isnothing(p.reduction)
        fr_r, _, _ = MMI.fit(p.reduction, verbosity, Xt)
        Xt = MMI.transform(p.reduction, fr_r, Xt)
    end
    
    # 4. Model
    fr_m, _, _ = MMI.fit(p.model, verbosity, Xt, y) # Entrenamos el clasificador
    
    return (fr_s, fr_f, fr_r, fr_m), nothing, nothing
end

function MMI.predict(p::ManualPipeline, fitresult, Xnew) # Toma datos nuevos y predice
    fr_s, fr_f, fr_r, fr_m = fitresult
    Xt = MMI.transform(p.scaler, fr_s, Xnew)
    Xt = MMI.transform(p.filter, fr_f, Xt)
    if !isnothing(p.reduction)
        Xt = MMI.transform(p.reduction, fr_r, Xt)
    end
    return MMI.predict(p.model, fr_m, Xt)
end

MMI.input_scitype(::Type{<:ManualPipeline}) = MMI.Table(MMI.ScientificTypes.Continuous)
MMI.target_scitype(::Type{<:ManualPipeline}) = AbstractVector{<:MMI.Finite}
MMI.prediction_type(::Type{<:ManualPipeline}) = :probabilistic