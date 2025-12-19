# Este archivo define funciones auxiliares para validación, cálculo estadístico y visualización

# --- Gestión de Datos y Validación Cruzada ---

# Convierte vector categórico a enteros (necesario para Mutual Information)
function _y_to_int(y)
    unique_classes = levels(y)
    idx_map = Dict(class => i for (i, class) in enumerate(unique_classes))
    return [idx_map[v] for v in y]
end

# Genera índices de CV respetando a los sujetos
function subject_folds(df::DataFrame; k::Int=5, seed::Int=SEED)
    subjects = unique(df.subject)
    
    rng = MersenneTwister(seed)
    shuffle!(rng, subjects)
    
    # Asignar cada sujeto a un fold
    fold_assignment = Dict(s => mod1(i, k) for (i, s) in enumerate(subjects))
    
    folds = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, k)
    
    for k_i in 1:k
        train_idx = Int[]
        val_idx   = Int[]
        
        for (row_i, row) in enumerate(eachrow(df))
            if fold_assignment[row.subject] == k_i
                push!(val_idx, row_i)
            else
                push!(train_idx, row_i)
            end
        end
        folds[k_i] = (train_idx, val_idx)
    end
    return folds
end

# --- Cálculos Estadísticos ---

# Calcula F-Score (ANOVA one-way) para feature selection
function _anova_f_score(x::AbstractVector{<:Real}, y)
    n = length(x)
    classes = levels(y)
    num_classes = length(classes)
    
    if num_classes < 2 return 0.0 end
    
    global_mean = mean(x)
    SST = 0.0 
    SSE = 0.0 
    
    for class in classes
        mask = y .== class
        xi = x[mask]
        ni = length(xi)
        if ni == 0 continue end
        
        mean_i = mean(xi)
        SST += ni * (mean_i - global_mean)^2
        SSE += sum((xi .- mean_i) .^ 2)
    end
    
    if SSE == 0.0 return 0.0 end
    
    return (SST / (num_classes - 1)) / (SSE / (n - num_classes))
end

# Calcula Mutual Information discretizando la variable continua
function _mutual_info(x::AbstractVector{<:Real}, y_int::Vector{Int}; nbins::Int=10)
    n = length(x)
    if n == 0 return 0.0 end
    
    xmin, xmax = minimum(x), maximum(x)
    if xmin == xmax return 0.0 end
    
    # Discretización
    xbin = Vector{Int}(undef, n)
    scale = nbins / (xmax - xmin)
    for i in 1:n
        b = floor(Int, (x[i] - xmin) * scale) + 1
        xbin[i] = clamp(b, 1, nbins)
    end
    
    # Matriz de contingencia
    num_classes = maximum(y_int)
    counts = zeros(Float64, nbins, num_classes)
    for i in 1:n
        counts[xbin[i], y_int[i]] += 1
    end
    
    # Entropía conjunta vs marginales
    pxy = counts ./ n     
    px  = vec(sum(pxy, dims=2))   
    py  = vec(sum(pxy, dims=1))   
    
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

# Devuelve índices de las k mejores características
function _get_best_indices(scores::Vector{<:Real}, k::Int)
    scores_clean = [isnan(s) ? -1.0 : s for s in scores]
    p = length(scores)
    n_select = clamp(k, 1, p)
    return sortperm(scores_clean, rev=true)[1:n_select]
end

# --- Utilidades de Visualización ---

# Muestra matrices de confusión
function show_cm(dic_matrices; classes=levels(y_trainval))
    
    for (nombre_modelo, cm_obj) in dic_matrices
        # Extraer matriz numérica
        matriz = hasproperty(cm_obj, :mat) ? cm_obj.mat : cm_obj
        
        # Generar anotaciones
        annotations = vec([
            (j, i, text(string(Int(matriz[i, j])), :center, :vcenter, 10, :black)) 
            for i in 1:size(matriz, 1), j in 1:size(matriz, 2)
        ])
        
        titulo = "Matriz de Confusión: $nombre_modelo"
    
        p = heatmap(
            classes,    
            classes,    
            matriz,   
            title = titulo,
            xlabel = "Predicción",
            ylabel = "Real",
            color = :blues, 
            aspect_ratio = 1,
            yflip = true,      
            xrotation = 45,     
            size = (800, 600),   
            left_margin = 20mm,  
            bottom_margin = 20mm,
            right_margin = 10mm
        )
        
        annotate!(annotations)
        display(p)
    end
end

# --- Selección de Modelos ---

# Busca el mejor modelo de cierto tipo en el CSV de resultados
function get_better(tipo_modelo::String, csv_path::String, metric=:B_Accuracy_mean)
    df = CSV.read(csv_path, DataFrame)
    subset = filter(row -> occursin(tipo_modelo, row.Model), df)
    
    if nrow(subset) == 0
        error("No se encontraron modelos del tipo '$tipo_modelo' en $csv_path")
    end
    
    sort!(subset, metric, rev=true)
    best_row = subset[1, :]
    
    println("Mejor $tipo_modelo encontrado:")
    println("Config: $(best_row.Filter) + $(best_row.Reduction) + $(best_row.Model)")
    println("-> $(metric): $(round(best_row[metric], digits=4))\n")
    
    pipe = PersonalizedPipeline(
        scaler    = MyMinMaxScaler(),
        filter    = all_filtros[best_row.Filter],
        reduction = all_reducciones[best_row.Reduction],
        clf       = all_modelos[best_row.Model]
    )
    return pipe
end

# --- Test estadístico ---
function my_friedman_test(matriz_resultados)
    N, k = size(matriz_resultados) # N=folds, k=modelos
    
    # Calcular rankings por fila (por fold)
    ranks = mapslices(tiedrank, matriz_resultados, dims=2)
    
    # Suma de rangos por columna (por modelo)
    R = vec(sum(ranks, dims=1))
    
    # Estadístico de Friedman (Chi-cuadrado aproximación)
    # Fórmula: Q = [12 / (N*k*(k+1))] * sum(R^2) - 3*N*(k+1)
    sum_R_sq = sum(R.^2)
    Q = (12.0 / (N * k * (k + 1))) * sum_R_sq - 3 * N * (k + 1)
    
    # Cálculo del p-valor
    dist = Chisq(k - 1)
    p_val = ccdf(dist, Q)
    
    return Q, p_val, ranks
end

# --- Helpers Proyección 2D ---

# Convierte input a matriz 2D de forma robusta
function to_matrix_2d(X)
    if X isa AbstractMatrix
        return X
    end
    
    try
        Xm = MMI.matrix(X)
        if size(Xm, 2) >= 2
            return Xm[:, 1:2]
        end
    catch
    end
    
    if X isa NamedTuple
        cols = collect(values(X))
        return hcat(cols...)
    end
    
    error("No se pudo convertir X a matriz 2D")
end

# Plotea proyecciones 2D (Train vs Test)
function plot_2d_projection(train2d, ytrain; test2d=nothing, ytest=nothing, title_str="Proyección 2D")
    train_mat = to_matrix_2d(train2d)
    labels = sort(unique(ytrain))
    palette = [:blue, :red, :green, :orange, :purple, :cyan]

    p1 = plot(title=title_str * " - TRAIN", xlabel="Comp 1", ylabel="Comp 2", legend=:outerright)

    for (i, lab) in enumerate(labels)
        mask = ytrain .== lab
        scatter!(p1, train_mat[mask,1], train_mat[mask,2],
                 label=string(lab), markersize=3, alpha=0.6,
                 color=palette[mod1(i, length(palette))])
    end

    if test2d !== nothing && ytest !== nothing
        test_mat = to_matrix_2d(test2d)
        p2 = plot(title=title_str * " - TEST", xlabel="Comp 1", ylabel="Comp 2", legend=:outerright)

        for (i, lab) in enumerate(labels)
            mask = ytest .== lab
            scatter!(p2, test_mat[mask,1], test_mat[mask,2],
                     label=string(lab), markersize=3, alpha=0.6,
                     color=palette[mod1(i, length(palette))])
        end
        return plot(p1, p2, layout=(2,1), size=(900,1200))
    end

    return p1
end

# Ratio de separabilidad Fisher (entre clases / intra clase)
function calculate_separability(X_2d, y)
    global_mean = mean(X_2d, dims=1)
    activities = unique(y)
    
    # Dispersión entre clases (Sb)
    Sb = 0.0
    for activity in activities
        mask = y .== activity
        X_class = X_2d[mask, :]
        class_mean = mean(X_class, dims=1)
        n_class = sum(mask)
        diff = class_mean .- global_mean
        Sb += n_class * sum(diff .^ 2)
    end
    
    # Dispersión intra clases (Sw)
    Sw = 0.0
    for activity in activities
        mask = y .== activity
        X_class = X_2d[mask, :]
        class_mean = mean(X_class, dims=1)
        for i in 1:size(X_class, 1)
            diff = X_class[i:i, :] .- class_mean
            Sw += sum(diff .^ 2)
        end
    end
    
    return Sb / (Sw + 1e-10)
end