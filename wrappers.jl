# wrappers.jl
include("dependencies.jl")

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
# 2. FILTROS
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
MMI.transform(m::MyANOVAFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

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
MMI.transform(m::MyPearsonFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

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
MMI.transform(m::MySpearmanFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

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
MMI.transform(m::MyKendallFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

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
MMI.transform(m::MyMIFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

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
MMI.transform(m::MyRFEFilter, c, X) = MMI.table(MMI.matrix(X)[:, c.selected])

# ==============================================================================
# 3. TRAITS DE NUESTROS FILTROS
# ==============================================================================
for T in [MyANOVAFilter, MyPearsonFilter, MySpearmanFilter, MyKendallFilter, MyMIFilter, MyRFEFilter]
    MMI.input_scitype(::Type{<:T}) = MMI.Table(MMI.ScientificTypes.Continuous) # Como entrada aceptamos tablas con números continuos
    MMI.target_scitype(::Type{<:T}) = AbstractVector{<:MMI.Finite} # Vector finito de etiquetas (Problema de clasificación)
end


# ==============================================================================
# 4. MODELO PARA EL CASO DE "SIN REDUCCIÓN"
# ==============================================================================
struct IdentityTransformer <: MMI.Unsupervised end
MMI.fit(model::IdentityTransformer, verbosity, X) = nothing, nothing, nothing # Al entrenar, no aprende ni devuelve nada
MMI.transform(model::IdentityTransformer, fitresult, X) = X # Al recibir X, devolvemos exactamente el mismo X

# Para que MLJ sepa que no cambia el tipo de datos
MMI.input_scitype(::Type{<:IdentityTransformer}) = MMI.Table
MMI.output_scitype(::Type{<:IdentityTransformer}) = MMI.Table


# ==============================================================================
# 5. WRAPPER LEARNING NETWORK 
# ==============================================================================
mutable struct PersonalizedPipeline <: MLJBase.ProbabilisticNetworkComposite
    scaler::MMI.Model
    filter::Union{MMI.Model, Nothing}
    reduction::Union{MMI.Model, Nothing}
    clf::MMI.Model
end

# Constructor externo para permitir keywords (scaler=..., clf=...)
function PersonalizedPipeline(; 
    scaler = MyMinMaxScaler(), 
    filter = nothing, 
    reduction = nothing, 
    clf = KNNClassifier()
)
    return PersonalizedPipeline(scaler, filter, reduction, clf)
end

function MLJBase.prefit(model::PersonalizedPipeline, verbosity, X, y)
    Xs = source(X)
    ys = source(y)

    # 1. Scaler
    m_scaler = machine(:scaler, Xs)
    X_curr = MLJ.transform(m_scaler, Xs)

    # 2. Filtro (Supervisado)
    if model.filter !== nothing
        m_filter = machine(:filter, X_curr, ys)
        X_curr = MLJ.transform(m_filter, X_curr)
    end

    # 3. Reducción
    if model.reduction !== nothing
        if MLJ.is_supervised(model.reduction)
            m_red = machine(:reduction, X_curr, ys) # LDA (Supervisada)
        else
            m_red = machine(:reduction, X_curr)     # PCA / ICA
        end
        X_curr = MLJ.transform(m_red, X_curr)
    end

    # 4. Clasificador
    m_clf = machine(:clf, X_curr, ys)
    yhat  = MLJ.predict(m_clf, X_curr)

    return (predict=yhat,)
end