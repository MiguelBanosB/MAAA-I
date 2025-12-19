# ==============================================================================
# SCALER
# ==============================================================================
# Wrapper para escalado Min-Max (0 a 1)
struct MyMinMaxScaler <: MMI.Unsupervised end

function MMI.fit(model::MyMinMaxScaler, verbosity::Int, X)
    Xmat = MMI.matrix(X)
    mins = mapslices(minimum, Xmat; dims=1)[1, :]
    maxs = mapslices(maximum, Xmat; dims=1)[1, :]
    return (mins=mins, maxs=maxs), nothing, nothing
end

function MMI.transform(model::MyMinMaxScaler, cache, X)
    Xmat = MMI.matrix(X)
    ranges = cache.maxs .- cache.mins
    # Evitar división por cero en columnas constantes
    ranges[ranges .== 0] .= 1.0 
    X_scaled = (Xmat .- cache.mins') ./ ranges'
    return MMI.table(Float64.(X_scaled))
end

# ==============================================================================
# FILTROS DE SELECCIÓN DE CARACTERÍSTICAS
# ==============================================================================

# -- ANOVA (F-Test) --
struct MyANOVAFilter <: MMI.Supervised
    n_features::Int
end
MyANOVAFilter(; n_features::Int=100) = MyANOVAFilter(n_features)

function MMI.fit(model::MyANOVAFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_cat = coerce(y, Multiclass)
    scores = [_anova_f_score(view(Xmat, :, j), y_cat) for j in 1:size(Xmat, 2)]
    selected = _get_best_indices(scores, model.n_features)
    return (selected=selected, scores=scores), nothing, nothing
end

MMI.transform(m::MyANOVAFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

# -- PEARSON --
mutable struct MyPearsonFilter <: MMI.Supervised
    n_features::Int
    redundancy_threshold::Float64
end
MyPearsonFilter(; n_features::Int=100, redundancy_threshold=0.9) = MyPearsonFilter(n_features, redundancy_threshold)

function MMI.fit(model::MyPearsonFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    n_features = size(Xmat, 2)
    y_float = Float64.(_y_to_int(coerce(y, Multiclass)))
 
    # Relevancia (Correlación Feature-Target)
    corr_with_y = zeros(Float64, n_features)
    for j in 1:n_features
        c = abs(cor(view(Xmat, :, j), y_float))
        corr_with_y[j] = isnan(c) ? 0.0 : c
    end
    
    # Selección iterativa 
    order = sortperm(corr_with_y, rev=true)
    selected = Int[]
    
    for j in order
        if length(selected) >= model.n_features break end
        
        is_redundant = false
        for s in selected
            c_red = abs(cor(view(Xmat, :, j), view(Xmat, :, s)))
            if !isnan(c_red) && c_red > model.redundancy_threshold
                is_redundant = true
                break
            end
        end
        if !is_redundant push!(selected, j) end
    end
    return (selected=selected, scores=corr_with_y), nothing, nothing
end

MMI.transform(m::MyPearsonFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

# -- SPEARMAN --
struct MySpearmanFilter <: MMI.Supervised
    n_features::Int
end
MySpearmanFilter(; n_features::Int=100) = MySpearmanFilter(n_features)

function MMI.fit(model::MySpearmanFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_float = Float64.(_y_to_int(coerce(y, Multiclass)))
    
    scores = zeros(Float64, size(Xmat, 2))
    for j in 1:size(Xmat, 2)
        val = abs(corspearman(view(Xmat, :, j), y_float))
        scores[j] = isnan(val) ? 0.0 : val
    end
    
    selected = _get_best_indices(scores, model.n_features)
    return (selected=selected, scores=scores), nothing, nothing
end
MMI.transform(m::MySpearmanFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

# -- KENDALL TAU --
struct MyKendallFilter <: MMI.Supervised
    n_features::Int
end
MyKendallFilter(; n_features::Int=100) = MyKendallFilter(n_features)

function MMI.fit(model::MyKendallFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_float = Float64.(_y_to_int(coerce(y, Multiclass)))
    
    scores = zeros(Float64, size(Xmat, 2))
    for j in 1:size(Xmat, 2)
        val = abs(corkendall(view(Xmat, :, j), y_float))
        scores[j] = isnan(val) ? 0.0 : val
    end
    
    selected = _get_best_indices(scores, model.n_features)
    return (selected=selected, scores=scores), nothing, nothing
end
MMI.transform(m::MyKendallFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

# -- MUTUAL INFORMATION --
struct MyMIFilter <: MMI.Supervised
    n_features::Int
end
MyMIFilter(; n_features::Int=100) = MyMIFilter(n_features)

function MMI.fit(model::MyMIFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_int = _y_to_int(coerce(y, Multiclass))
    scores = [_mutual_info(view(Xmat, :, j), y_int) for j in 1:size(Xmat, 2)]
    selected = _get_best_indices(scores, model.n_features)
    return (selected=selected, scores=scores), nothing, nothing
end
MMI.transform(m::MyMIFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

# -- RFE (Recursive Feature Elimination) --
struct MyRFEFilter <: MMI.Supervised
    n_features::Int
end
MyRFEFilter(; n_features::Int=100) = MyRFEFilter(n_features)

function MMI.fit(model::MyRFEFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    n, p = size(Xmat)
    current_indices = collect(1:p)
    
    while length(current_indices) > model.n_features
        X_curr = Xmat[:, current_indices]
        clf = LogisticClassifier(lambda=0.1, fit_intercept=true) 
        mach = machine(clf, MMI.table(X_curr), y)
        MLJ.fit!(mach, verbosity=0)
        
        fp = fitted_params(mach)
        coefs = fp.coefs 
        
        # Calcular importancia
        if coefs isa Matrix
            importance = vec(sum(abs.(coefs), dims=1)) 
        elseif coefs isa Vector && length(coefs) > 0 && coefs[1] isa Pair
            importance = abs.([c.second for c in coefs])
        else
            importance = abs.(coefs)
        end
        
        # Ajuste de tamaño
        if length(importance) != length(current_indices)
            importance = importance[end-length(current_indices)+1:end]
        end

        # Eliminar 50% peores
        n_current = length(current_indices)
        n_keep = max(model.n_features, floor(Int, n_current * 0.5))
        
        top_k_local = sortperm(importance, rev=true)[1:n_keep]
        current_indices = current_indices[top_k_local]
    end
    
    return (selected=sort(current_indices), scores=zeros(p)), nothing, nothing
end
MMI.transform(m::MyRFEFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

# -- TRAITS --
for T in [MyANOVAFilter, MyPearsonFilter, MySpearmanFilter, MyKendallFilter, MyMIFilter, MyRFEFilter]
    MMI.input_scitype(::Type{<:T}) = MMI.Table(MMI.ScientificTypes.Continuous)
    MMI.target_scitype(::Type{<:T}) = AbstractVector{<:MMI.Finite}
end

# ==============================================================================
# 3. IDENTITY & PIPELINE
# ==============================================================================

# -- Transformador Identidad --
struct IdentityTransformer <: MMI.Unsupervised end
MMI.fit(model::IdentityTransformer, verbosity, X) = nothing, nothing, nothing
MMI.transform(model::IdentityTransformer, fitresult, X) = X
MMI.input_scitype(::Type{<:IdentityTransformer}) = MMI.Table
MMI.output_scitype(::Type{<:IdentityTransformer}) = MMI.Table

# -- Pipeline Personalizado --
mutable struct PersonalizedPipeline <: MLJBase.ProbabilisticNetworkComposite
    scaler::MMI.Model
    filter::Union{MMI.Model, Nothing}
    reduction::Union{MMI.Model, Nothing}
    clf::MMI.Model
end

function PersonalizedPipeline(; scaler=MyMinMaxScaler(), filter=nothing, reduction=nothing, clf=KNNClassifier())
    return PersonalizedPipeline(scaler, filter, reduction, clf)
end

function MLJBase.prefit(model::PersonalizedPipeline, verbosity, X, y)
    Xs = source(X)
    ys = source(y)

    # Normalización
    m_scaler = machine(:scaler, Xs)
    X_curr = MLJ.transform(m_scaler, Xs)

    # Filtrado
    if model.filter !== nothing
        m_filter = machine(:filter, X_curr, ys)
        X_curr = MLJ.transform(m_filter, X_curr)
    end

    # Reducción
    if model.reduction !== nothing
        if MLJ.is_supervised(model.reduction)
            m_red = machine(:reduction, X_curr, ys)
        else
            m_red = machine(:reduction, X_curr)
        end
        X_curr = MLJ.transform(m_red, X_curr)
    end

    # Clasificación
    m_clf = machine(:clf, X_curr, ys)
    yhat = MLJ.predict(m_clf, X_curr)

    return (predict=yhat,)
end

# -- TRAITS PIPELINE --
MLJBase.input_scitype(::Type{<:PersonalizedPipeline})  = MLJBase.Table(MLJBase.Continuous)
MLJBase.target_scitype(::Type{<:PersonalizedPipeline}) = AbstractVector{<:Finite}
MLJBase.is_pure_julia(::Type{<:PersonalizedPipeline})  = true

# ==============================================================================
# WRAPPER DE HARD VOTING
# ==============================================================================
mutable struct HardVotingClassifier <: MLJBase.Deterministic
    base_model::MLJBase.Probabilistic
    n_partitions::Int
    rng::Int
end

HardVotingClassifier(; base_model=ProbabilisticSVC(cost=0.5), n_partitions=3, rng=104) = HardVotingClassifier(base_model, n_partitions, rng)

function MLJBase.fit(m::HardVotingClassifier, verbosity::Int, X, y)
    rng = Random.MersenneTwister(m.rng)
    trained_models = Any[]
    n_rows = length(y)
    
    # Bagging simulado por particiones (Bootstraping)
    for i in 1:m.n_partitions
        indices = rand(rng, 1:n_rows, n_rows)
        X_subset = selectrows(X, indices)
        y_subset = selectrows(y, indices)
        
        mach = machine(m.base_model, X_subset, y_subset)
        fit!(mach, verbosity=0)
        push!(trained_models, mach)
    end
    
    return trained_models, nothing, nothing
end

function MLJBase.predict(m::HardVotingClassifier, fitresult, X_new)
    trained_models = fitresult
    all_votes = [MLJ.predict_mode(mach, X_new) for mach in trained_models]
    votes_matrix = hcat(all_votes...)
    
    # Moda de votos por fila
    final_predictions = [mode(row) for row in eachrow(votes_matrix)]
    return final_predictions
end

# ==============================================================================
# WRAPPERS VISUALIZACIÓN
# ==============================================================================

# -- Isomap --
mutable struct MyIsomap <: MMI.Unsupervised
    n_components::Int
    k::Int
end
MyIsomap(; n_components::Int=2, k::Int=12) = MyIsomap(n_components, k)

function MMI.fit(model::MyIsomap, verbosity::Int, X)
    # Fit vacío, transform aplica el algoritmo
    return (k=model.k, n=model.n_components), nothing, nothing
end

function MMI.transform(model::MyIsomap, cache, X)
    Xmat = MMI.matrix(X)
    isomap_result = ManifoldLearning.fit(ManifoldLearning.Isomap, Xmat', maxoutdim=cache.n, k=cache.k)
    return MMI.table(ManifoldLearning.predict(isomap_result)')
end

# -- LLE --
mutable struct MyLLE <: MMI.Unsupervised
    n_components::Int
    k::Int
end
MyLLE(; n_components::Int=2, k::Int=12) = MyLLE(n_components, k)

function MMI.fit(model::MyLLE, verbosity::Int, X)
    return (k=model.k, n=model.n_components), nothing, nothing
end

function MMI.transform(model::MyLLE, cache, X)
    Xmat = MMI.matrix(X)
    lle_result = ManifoldLearning.fit(ManifoldLearning.LLE, Xmat', maxoutdim=cache.n, k=cache.k)
    return MMI.table(ManifoldLearning.predict(lle_result)')
end

# -- TRAITS VISUALIZACIÓN --
for T in [MyIsomap, MyLLE]
    MMI.input_scitype(::Type{<:T}) = MMI.Table(MMI.Continuous)
    MMI.output_scitype(::Type{<:T}) = MMI.Table(MMI.Continuous)
end