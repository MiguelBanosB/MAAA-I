include("setup.jl")
include("helpers.jl") 

# ==============================================================================
# 1. SCALER 
# ==============================================================================
# Wrapper para escalado Min-Max (0 a 1).
struct MyMinMaxScaler <: MMI.Unsupervised end

function MMI.fit(model::MyMinMaxScaler, verbosity::Int, X)
    Xmat = MMI.matrix(X)
    # Calcula mínimos y máximos por columna (feature)
    mins = mapslices(minimum, Xmat; dims=1)[1, :]
    maxs = mapslices(maximum, Xmat; dims=1)[1, :]
    return (mins=mins, maxs=maxs), nothing, nothing
end

function MMI.transform(model::MyMinMaxScaler, cache, X)
    Xmat = MMI.matrix(X)
    # Evita división por cero si max == min (columna constante)
    ranges = cache.maxs .- cache.mins
    ranges[ranges .== 0] .= 1.0 
    X_scaled = (Xmat .- cache.mins') ./ ranges'
    return MMI.table(Float64.(X_scaled))
end


# ==============================================================================
# 2. FILTROS DE SELECCIÓN DE CARACTERÍSTICAS
# ==============================================================================

# -- ANOVA (F-Test) --
# Selecciona variables basándose en la varianza entre medias de grupos vs varianza interna.
struct MyANOVAFilter <: MMI.Supervised
    n_features::Int
end
MyANOVAFilter(; n_features::Int=100) = MyANOVAFilter(n_features)

function MMI.fit(model::MyANOVAFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_cat = coerce(y, Multiclass)
    # Calcula F-score para cada columna usando el helper
    scores = [_anova_f_score(view(Xmat, :, j), y_cat) for j in 1:size(Xmat, 2)]
    selected = _get_best_indices(scores, model.n_features)
    return (selected=selected, scores=scores), nothing, nothing
end

# La transformación es común: recortar la matriz a las columnas seleccionadas
MMI.transform(m::MyANOVAFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])


# -- PEARSON (Correlación Lineal + Eliminación de Redundancia) --
# Selecciona variables correlacionadas con la clase, penalizando colinealidad entre ellas.
mutable struct MyPearsonFilter <: MMI.Supervised
    n_features::Int
    redundancy_threshold::Float64
end
MyPearsonFilter(; n_features::Int=100, redundancy_threshold=0.9) = MyPearsonFilter(n_features, redundancy_threshold)

function MMI.fit(model::MyPearsonFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    n_features = size(Xmat, 2)
    # Convertimos target a float para correlación (asumiendo ordinalidad o binario simple)
    y_float = Float64.(_y_to_int(coerce(y, Multiclass)))
 
    # 1. Calcular relevancia (Correlación Feature vs Target)
    corr_with_y = zeros(Float64, n_features)
    for j in 1:n_features
        c = abs(cor(view(Xmat, :, j), y_float))
        corr_with_y[j] = isnan(c) ? 0.0 : c
    end
    
    # 2. Selección iterativa con filtro de redundancia
    order = sortperm(corr_with_y, rev=true)
    selected = Int[]
    
    for j in order
        if length(selected) >= model.n_features break end
        
        is_redundant = false
        # Comparamos la candidata 'j' con las ya seleccionadas 's'
        for s in selected
            c_red = abs(cor(view(Xmat, :, j), view(Xmat, :, s)))
            if !isnan(c_red) && c_red > model.redundancy_threshold
                is_redundant = true
                break # Si es redundante con alguna, se descarta
            end
        end
        
        if !is_redundant push!(selected, j) end
    end
    return (selected=selected, scores=corr_with_y), nothing, nothing
end
MMI.transform(m::MyPearsonFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])


# -- SPEARMAN (Correlación de Rango) --
# Similar a Pearson pero captura relaciones monótonas no lineales.
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


# -- KENDALL (Correlación Tau) --
# Más robusta para datos pequeños o con muchos empates (ranks).
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


# -- MUTUAL INFORMATION (Información Mutua) --
# Captura cualquier tipo de dependencia (lineal o no) basada en entropía.
struct MyMIFilter <: MMI.Supervised
    n_features::Int
end
MyMIFilter(; n_features::Int=100) = MyMIFilter(n_features)

function MMI.fit(model::MyMIFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    y_int = _y_to_int(coerce(y, Multiclass))
    # Usa el helper _mutual_info discretizando variables continuas
    scores = [_mutual_info(view(Xmat, :, j), y_int) for j in 1:size(Xmat, 2)]
    selected = _get_best_indices(scores, model.n_features)
    return (selected=selected, scores=scores), nothing, nothing
end
MMI.transform(m::MyMIFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])


# -- RFE (Recursive Feature Elimination) --
# Elimina iterativamente las características menos importantes usando Regresión Logística.
struct MyRFEFilter <: MMI.Supervised
    n_features::Int
end
MyRFEFilter(; n_features::Int=100) = MyRFEFilter(n_features)

function MMI.fit(model::MyRFEFilter, verbosity::Int, X, y)
    Xmat = MMI.matrix(X)
    n, p = size(Xmat)
    current_indices = collect(1:p)
    
    # Bucle de eliminación recursiva
    while length(current_indices) > model.n_features
        X_curr = Xmat[:, current_indices]
        # Usamos LogisticRegression con regularización L2 (Ridge)
        clf = LogisticClassifier(lambda=0.1, fit_intercept=true) 
        mach = machine(clf, MMI.table(X_curr), y)
        MLJ.fit!(mach, verbosity=0)
        
        fp = fitted_params(mach)
        coefs = fp.coefs 
        
        # Calcular importancia absoluta de coeficientes
        if coefs isa Matrix
            # Caso Multiclase: Suma de valores absolutos por clase
            importance = vec(sum(abs.(coefs), dims=1)) 
        elseif coefs isa Vector && length(coefs) > 0 && coefs[1] isa Pair
            # Caso MLJLinearModels genérico
            importance = abs.([c.second for c in coefs])
        else
            # Caso Binario simple
            importance = abs.(coefs)
        end
        
        # Ajuste de tamaño si hay intercept (a veces devuelve vector +1 long)
        if length(importance) != length(current_indices)
            importance = importance[end-length(current_indices)+1:end]
        end

        # Eliminar el 50% de las peores en cada paso (para velocidad)
        n_current = length(current_indices)
        n_keep = max(model.n_features, floor(Int, n_current * 0.5))
        
        top_k_local = sortperm(importance, rev=true)[1:n_keep]
        current_indices = current_indices[top_k_local]
    end
    
    return (selected=sort(current_indices), scores=zeros(p)), nothing, nothing
end
MMI.transform(m::MyRFEFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])


# ==============================================================================
# 3. TRAITS (METADATA DE MODELOS)
# ==============================================================================
# Define qué tipos de datos aceptan nuestros filtros (Tabla continua -> Target finito)
for T in [MyANOVAFilter, MyPearsonFilter, MySpearmanFilter, MyKendallFilter, MyMIFilter, MyRFEFilter]
    MMI.input_scitype(::Type{<:T}) = MMI.Table(MMI.ScientificTypes.Continuous)
    MMI.target_scitype(::Type{<:T}) = AbstractVector{<:MMI.Finite}
end


# ==============================================================================
# 4. IDENTITY & PIPELINE
# ==============================================================================

# -- Transformador Identidad --
# Se usa para representar "Sin Reducción" o "Sin Filtro" en el pipeline dinámico.
struct IdentityTransformer <: MMI.Unsupervised end
MMI.fit(model::IdentityTransformer, verbosity, X) = nothing, nothing, nothing
MMI.transform(model::IdentityTransformer, fitresult, X) = X
MMI.input_scitype(::Type{<:IdentityTransformer}) = MMI.Table
MMI.output_scitype(::Type{<:IdentityTransformer}) = MMI.Table

# -- Pipeline Personalizado (Network Composite) --
# Permite encadenar Scaler -> Filter -> Reduction -> Classifier dinámicamente.
# Es necesario porque MLJ.Pipeline estándar es rígido con los tipos (Supervisado vs No Sup).
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
    Xs = source(X) # Nodo fuente de datos
    ys = source(y) # Nodo fuente de target

    # 1. Normalización
    m_scaler = machine(:scaler, Xs)
    X_curr = MLJ.transform(m_scaler, Xs)

    # 2. Filtrado (Opcional)
    if model.filter !== nothing
        m_filter = machine(:filter, X_curr, ys)
        X_curr = MLJ.transform(m_filter, X_curr)
    end

    # 3. Reducción de Dimensionalidad (Opcional)
    if model.reduction !== nothing
        # Detectamos si la reducción necesita el target (LDA) o no (PCA, ICA)
        if MLJ.is_supervised(model.reduction)
            m_red = machine(:reduction, X_curr, ys) # Supervisado (ej. LDA)
        else
            m_red = machine(:reduction, X_curr)     # No Supervisado (ej. PCA)
        end
        X_curr = MLJ.transform(m_red, X_curr)
    end

    # 4. Clasificación Final
    m_clf = machine(:clf, X_curr, ys)
    yhat = MLJ.predict(m_clf, X_curr)

    return (predict=yhat,)
end

# ==============================================================================
# 5. WRAPPER DE HARD VOTING
# ==============================================================================
mutable struct HardVotingClassifier <: MLJBase.Deterministic
    base_model::MLJBase.Probabilistic
    n_partitions::Int
    rng::Int
end

# Constructor externo
HardVotingClassifier(; base_model=ProbabilisticSVC(cost=0.5), n_partitions=3, rng=104) = HardVotingClassifier(base_model, n_partitions, rng)

function MLJBase.fit(m::HardVotingClassifier, verbosity::Int, X, y)
    rng = Random.MersenneTwister(m.rng)
    trained_models = Any[]
    
    n_rows = length(y)
    
    # Bucle de Bagging
    for i in 1:m.n_partitions
        indices = rand(rng, 1:n_rows, n_rows) # Bootstrapping
        
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

    # Cada modelo vota su etiqueta favorita (predict_mode)
    all_votes = [MLJ.predict_mode(mach, X_new) for mach in trained_models]
    
    votes_matrix = hcat(all_votes...)
    
    # Calculamos la moda
    final_predictions = [mode(row) for row in eachrow(votes_matrix)]
    
    return final_predictions
end

# Traits
MLJBase.input_scitype(::Type{<:PersonalizedPipeline})  = MLJBase.Table(MLJBase.Continuous)
MLJBase.target_scitype(::Type{<:PersonalizedPipeline}) = AbstractVector{<:Finite}
MLJBase.is_pure_julia(::Type{<:PersonalizedPipeline})  = true
MLJBase.supports_weights(::Type{<:PersonalizedPipeline}) = false


# ==============================================================================
# WRAPPER PARA ISOMAP
# ==============================================================================
mutable struct MyIsomap <: MMI.Unsupervised
    n_components::Int
    k::Int  # número de vecinos
end

MyIsomap(; n_components::Int=2, k::Int=12) = MyIsomap(n_components, k)

function MMI.fit(model::MyIsomap, verbosity::Int, X)
    # Isomap tampoco se "entrena" tradicionalmente
    # Guardamos parámetros
    cache = (k = model.k,)
    return cache, nothing, nothing
end

function MMI.transform(model::MyIsomap, cache, X)
    Xmat = MMI.matrix(X)
    
    # ManifoldLearning espera (features, samples)
    isomap_result = ManifoldLearning.fit(ManifoldLearning.Isomap, 
                                         Xmat', 
                                         maxoutdim=model.n_components, 
                                         k=cache.k)
    
    X_embedded = ManifoldLearning.predict(isomap_result)'
    
    return MMI.table(X_embedded)
end

MMI.input_scitype(::Type{<:MyIsomap}) = MMI.Table(Continuous)
MMI.output_scitype(::Type{<:MyIsomap}) = MMI.Table(Continuous)


# ==============================================================================
# WRAPPER PARA LLE (Locally Linear Embedding)
# ==============================================================================
mutable struct MyLLE <: MMI.Unsupervised
    n_components::Int
    k::Int  # número de vecinos
end

MyLLE(; n_components::Int=2, k::Int=12) = MyLLE(n_components, k)

function MMI.fit(model::MyLLE, verbosity::Int, X)
    cache = (k = model.k,)
    return cache, nothing, nothing
end

function MMI.transform(model::MyLLE, cache, X)
    Xmat = MMI.matrix(X)
    
    # ManifoldLearning espera (features, samples)
    lle_result = ManifoldLearning.fit(ManifoldLearning.LLE, 
                                      Xmat', 
                                      maxoutdim=model.n_components, 
                                      k=cache.k)
    
    X_embedded = ManifoldLearning.predict(lle_result)'
    
    return MMI.table(X_embedded)
end



MMI.input_scitype(::Type{<:MyLLE}) = MMI.Table(MMI.ScientificTypes.Continuous)
MMI.output_scitype(::Type{<:MyLLE}) = MMI.Table(MMI.ScientificTypes.Continuous)