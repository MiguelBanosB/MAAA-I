# wrappers.jl CORREGIDO
println("Cargando wrappers y pipeline manual (Versión Robusta)...")

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
# 2. FILTROS (Sin cambios en la lógica, solo en la definición)
# ==============================================================================

# Helper para filtros: Fit y Transform genéricos
# (Mantenemos tu lógica pero aseguramos que devuelvan matrices limpias)

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
MMI.transform(m::MyANOVAFilter, c, X) = MMI.matrix(X)[:, c.selected]

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
MMI.transform(m::MyPearsonFilter, c, X) = MMI.matrix(X)[:, c.selected]

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
        w = Xi \ y_float 
        importance = abs.(w)
        n_remaining = length(original_indices)
        n_to_remove = max(1, ceil(Int, n_remaining * 0.5))
        idx_sorted = sortperm(importance)
        deleteat!(original_indices, idx_sorted[1:n_to_remove])
    end
    return (selected=original_indices, scores=zeros(p)), nothing
end
MMI.transform(m::MyRFEFilter, c, X) = MMI.matrix(X)[:, c.selected]

# ==============================================================================
# 3. TRAITS
# ==============================================================================
for T in [MyANOVAFilter, MyPearsonFilter, MySpearmanFilter, MyKendallFilter, MyMIFilter, MyRFEFilter]
    MMI.input_scitype(::Type{<:T}) = MMI.Table(MMI.ScientificTypes.Continuous)
    MMI.target_scitype(::Type{<:T}) = AbstractVector{<:MMI.Finite}
end

# ==============================================================================
# 4. MANUAL PIPELINE (VERSIÓN ROBUSTA: MATRIX -> TABLE -> DETERMINISTIC)
# ==============================================================================
# Cambiamos a Deterministic para asegurar que siempre devolvemos etiquetas para Accuracy/F1
mutable struct ManualPipeline{F, R, M} <: MMI.Deterministic 
    scaler::MyMinMaxScaler
    filter::F
    reduction::R
    model::M
end

function MMI.fit(p::ManualPipeline, verbosity::Int, X, y)
    # 1. Scaler
    fr_s, _, _ = MMI.fit(p.scaler, verbosity, X)
    Xt_mat = MMI.transform(p.scaler, fr_s, X)
    Xt = MMI.table(Xt_mat) # Convertimos Matriz -> Tabla para el siguiente paso
    
    # 2. Filter
    fr_f, _, _ = MMI.fit(p.filter, verbosity, Xt, y)
    Xt_mat = MMI.transform(p.filter, fr_f, Xt)
    Xt = MMI.table(Xt_mat) # Matriz -> Tabla
    
    # 3. Reduction
    fr_r = nothing
    if !isnothing(p.reduction)
        fr_r, _, _ = MMI.fit(p.reduction, verbosity, Xt)
        Xt_red = MMI.transform(p.reduction, fr_r, Xt)
        # Algunos reducciones devuelven tabla, otras matriz. Aseguramos Tabla.
        Xt = MMI.table(MMI.matrix(Xt_red)) 
    end
    
    # 4. Model
    fr_m, _, _ = MMI.fit(p.model, verbosity, Xt, y)
    
    return (fr_s, fr_f, fr_r, fr_m), nothing, nothing
end

function MMI.predict(p::ManualPipeline, fitresult, Xnew)
    fr_s, fr_f, fr_r, fr_m = fitresult
    
    # Aplicamos transformaciones asegurando formato Tabla
    Xt_mat = MMI.transform(p.scaler, fr_s, Xnew)
    Xt = MMI.table(Xt_mat)
    
    Xt_mat = MMI.transform(p.filter, fr_f, Xt)
    Xt = MMI.table(Xt_mat)
    
    if !isnothing(p.reduction)
        Xt_red = MMI.transform(p.reduction, fr_r, Xt)
        Xt = MMI.table(MMI.matrix(Xt_red))
    end
    
    # Predicción FINAL
    # Si el modelo es probabilístico (KNN, MLP), pedimos la MODA (Etiqueta)
    if MMI.prediction_type(p.model) == :probabilistic
        return MMI.predict_mode(p.model, fr_m, Xt)
    else
        # Si es determinista (SVM), pedimos la predicción directa
        return MMI.predict(p.model, fr_m, Xt)
    end
end

# Traits del Pipeline
MMI.input_scitype(::Type{<:ManualPipeline}) = MMI.Table(MMI.ScientificTypes.Continuous)
MMI.target_scitype(::Type{<:ManualPipeline}) = AbstractVector{<:MMI.Finite}