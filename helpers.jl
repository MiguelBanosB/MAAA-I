# Este archivo define una serie de funciones auxiliares, que implementa lógica matemática, particionado de datos y demás utilidades


# --- 1. Gestión de Datos y Validación Cruzada ---
# Convierte un vector categórico (Strings/Symbols) a enteros (1..N). Es necesario para el cálculo de Mutual Information basado en arrays.
function _y_to_int(y)
    unique_classes = levels(y)
    idx_map = Dict(class => i for (i, class) in enumerate(unique_classes))
    return [idx_map[v] for v in y]
end


# Genera índices de CV respetando a los sujetos.
function subject_folds(df::DataFrame; k::Int=5, seed::Int=SEED)
    subjects = unique(df.subject)
    
    # Mezclamos sujetos aleatoriamente con la semilla fijada
    rng = MersenneTwister(seed)
    shuffle!(rng, subjects)
    
    # Asignamos cada sujeto a un fold (1 a k)
    fold_assignment = Dict(s => mod1(i, k) for (i, s) in enumerate(subjects))
    
    folds = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, k)
    
    # Construimos los índices para cada fold
    for k_i in 1:k
        train_idx = Int[]
        val_idx   = Int[]
        
        for (row_i, row) in enumerate(eachrow(df))
            if fold_assignment[row.subject] == k_i
                push!(val_idx, row_i)   # Sujeto pertenece al fold actual -> Test/Val
            else
                push!(train_idx, row_i) # Sujeto pertenece a otro fold -> Train
            end
        end
        folds[k_i] = (train_idx, val_idx)
    end
    return folds
end


# --- 2. Cálculos Estadísticos  ---
# Calcula el F-Score (ANOVA one-way) para una característica continua vs target categórico. Mide la varianza entre grupos vs la varianza dentro de los grupos.
function _anova_f_score(x::AbstractVector{<:Real}, y)
    n = length(x)
    classes = levels(y)
    num_classes = length(classes)
    
    if num_classes < 2 return 0.0 end
    
    global_mean = mean(x)
    SST = 0.0 # Suma de cuadrados entre grupos
    SSE = 0.0 # Suma de cuadrados del error
    
    for class in classes
        mask = y .== class
        xi = x[mask]
        ni = length(xi)
        
        if ni == 0 continue end
        
        mean_i = mean(xi)
        SST += ni * (mean_i - global_mean)^2
        SSE += sum((xi .- mean_i) .^ 2)
    end
    
    # Evitar división por cero
    if SSE == 0.0 return 0.0 end
    
    # Fórmula F-statistic
    return (SST / (num_classes - 1)) / (SSE / (n - num_classes))
end


# Calcula Mutual Information mediante discretización. Mide la dependencia no lineal entre variable continua y target.
function _mutual_info(x::AbstractVector{<:Real}, y_int::Vector{Int}; nbins::Int=10)
    n = length(x)
    if n == 0 return 0.0 end
    
    xmin, xmax = minimum(x), maximum(x)
    if xmin == xmax return 0.0 end
    
    # Discretización de variable continua en 'nbins'
    xbin = Vector{Int}(undef, n)
    scale = nbins / (xmax - xmin)
    for i in 1:n
        # Mapea valor a rango [1, nbins]
        b = floor(Int, (x[i] - xmin) * scale) + 1
        xbin[i] = clamp(b, 1, nbins)
    end
    
    # Matriz de contingencia
    num_classes = maximum(y_int)
    counts = zeros(Float64, nbins, num_classes)
    for i in 1:n
        counts[xbin[i], y_int[i]] += 1
    end
    
    # Cálculo de Entropía conjunta vs marginales
    pxy = counts ./ n             # Probabilidad conjunta
    px  = vec(sum(pxy, dims=2))   # Marginal X
    py  = vec(sum(pxy, dims=1))   # Marginal Y
    
    mi = 0.0
    for i in 1:nbins
        for j in 1:num_classes
            pij = pxy[i, j]
            if pij > 0.0
                # MI = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
                mi += pij * log(pij / (px[i] * py[j]))
            end
        end
    end
    return mi
end


# --- 3. Utilidades de Selección ---
# Devuelve los índices de las 'k' mejores características basándose en scores. Maneja NaNs convirtiéndolos a -1 para que queden al final.
function _get_best_indices(scores::Vector{<:Real}, k::Int)
    # Reemplazamos NaNs con -1.0 para que el sort no falle y queden al final
    scores_clean = [isnan(s) ? -1.0 : s for s in scores]
    
    p = length(scores)
    n_select = clamp(k, 1, p)
    
    # Orden descendente (mayor score es mejor)
    return sortperm(scores_clean, rev=true)[1:n_select]
end