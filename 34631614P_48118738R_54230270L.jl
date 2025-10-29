# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles
using JLD2
using Images
using ColorTypes
using Statistics
using MLJBase


function fileNamesFolder(folderName::String, extension::String)    
    isdir(folderName) || error("The directory $folderName doesn't exist.");
    extension = uppercase(extension);
    fileNames = filter(f -> endswith(uppercase(f), ".$extension"), readdir(folderName))
    fileNamesNoExtension = replace.(fileNames, "." * lowercase(extension) => "");
    return sort(fileNamesNoExtension);
end;


function loadDataset(datasetName::String, datasetFolder::String;
    datasetType::DataType=Float32)
    fileName = "$datasetName.tsv"
    path = joinpath(datasetFolder, fileName)
    isfile(path) || return nothing
    data = readdlm(path, '\t')
    headings = data[1, :]
    index_target = findfirst(x -> uppercase(x) == "TARGET", headings)
    
    if index_target === nothing
        println("'TARGET' column not found in dataset.")
        return nothing
    end;
    
    inputs_columns = setdiff(1:size(data, 2), index_target)
    inputs_raw = data[2:end, inputs_columns]
    targets_raw = data[2:end, index_target]
    inputs = convert(Matrix{datasetType}, inputs_raw)
    targets = convert(Vector{Bool}, targets_raw .== 1)  
    return (inputs, targets)
end;


function loadImage(imageName::String, datasetFolder::String;
    datasetType::DataType=Float32, resolution::Int=128)
    fileName = "$imageName.tif"
    path = joinpath(datasetFolder, fileName)
    isfile(path) || return nothing
    image = load(path)
    image = imresize(image, (resolution, resolution))
    image_gray = Gray.(image)
    image_matrix = convert(Matrix{datasetType}, image_gray)
    return image_matrix
end;


function convertImagesNCHW(imageVector::Vector{<:AbstractArray{<:Real,2}})
    imagesNCHW = Array{eltype(imageVector[1]), 4}(undef, length(imageVector), 1, size(imageVector[1],1), size(imageVector[1],2));
    for numImage in Base.OneTo(length(imageVector))
        imagesNCHW[numImage,1,:,:] .= imageVector[numImage];
    end;
    return imagesNCHW;
end;


function loadImagesNCHW(datasetFolder::String;
    datasetType::DataType=Float32, resolution::Int=128)
    imageNames = fileNamesFolder(datasetFolder, "tif");
    images = loadImage.(imageNames, datasetFolder; datasetType=datasetType, resolution=resolution);
    imagesNCHW = convertImagesNCHW(images);
    return imagesNCHW
end;


showImage(image      ::AbstractArray{<:Real,2}                                      ) = display(Gray.(image));
showImage(imagesNCHW ::AbstractArray{<:Real,4}                                      ) = display(Gray.(     hcat([imagesNCHW[ i,1,:,:] for i in 1:size(imagesNCHW ,1)]...)));
showImage(imagesNCHW1::AbstractArray{<:Real,4}, imagesNCHW2::AbstractArray{<:Real,4}) = display(Gray.(vcat(hcat([imagesNCHW1[i,1,:,:] for i in 1:size(imagesNCHW1,1)]...), hcat([imagesNCHW2[i,1,:,:] for i in 1:size(imagesNCHW2,1)]...))));


function loadMNISTDataset(datasetFolder::String; labels::AbstractArray{Int,1}=0:9, datasetType::DataType=Float32)
    fileName = "MNIST.jld2"
    path = joinpath(datasetFolder, fileName)
    isfile(path) || return nothing
    data = JLD2.load(path)

    train_images = data["train_imgs"]
    train_labels = collect(data["train_labels"])
    test_images = data["test_imgs"]
    test_labels = collect(data["test_labels"])

    labels = collect(labels)
    if -1 in labels
        train_labels[.!in.(train_labels, [setdiff(labels,-1)])] .= -1;
        test_labels[.!in.(test_labels, [setdiff(labels, -1)])] .= -1;
    end;

    train_indices = in.(train_labels, [labels])
    test_indices = in.(test_labels, [labels])

    train_images = train_images[train_indices]
    train_labels = train_labels[train_indices]
    test_images = test_images[test_indices]
    test_labels = test_labels[test_indices]

    train_images = convert.(Matrix{datasetType}, train_images)
    test_images = convert.(Matrix{datasetType}, test_images)

    train_images = convertImagesNCHW(train_images)
    test_images = convertImagesNCHW(test_images)
    
    return (train_images, train_labels, test_images, test_labels)
end;


function intervalDiscreteVector(data::AbstractArray{<:Real,1})
    # Ordenar los datos
    uniqueData = sort(unique(data));
    # Obtener diferencias entre elementos consecutivos
    differences = sort(diff(uniqueData));
    # Tomar la diferencia menor
    minDifference = differences[1];
    # Si todas las diferencias son multiplos exactos (valores enteros) de esa diferencia, entonces es un vector de valores discretos
    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    return all(isInteger.(differences./minDifference, 1e-3)) ? minDifference : 0.
end


function cyclicalEncoding(data::AbstractArray{<:Real,1})
    m = intervalDiscreteVector(data)
    min_value = minimum(data)
    max_value = maximum(data)
    angles = 2*pi .*(data .- min_value) ./ (max_value - min_value + m)
    sines = sin.(angles)
    cosines = cos.(angles)
    return (sines, cosines)
end;


function loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float32)
    data_file = joinpath(datasetFolder, "elec2_data.dat")
    label_file = joinpath(datasetFolder, "elec2_label.dat")
    if !(isfile(data_file) && isfile(label_file))
    return nothing
end
    inputs = readdlm(data_file)
    outputs = readdlm(label_file)
    columns_to_keep = setdiff(1:size(inputs, 2), [1, 4])
    inputs_reduced = inputs[:, columns_to_keep]
    day_column = inputs_reduced[:, 1]
    other_columns = inputs_reduced[:, 2:end]
    sin_day, cos_day = cyclicalEncoding(day_column)
    other_columns_converted = convert(Matrix{datasetType}, other_columns)
    final_inputs = hcat(sin_day, cos_day, other_columns_converted)
    final_inputs = datasetType.(final_inputs)
    final_outputs = vec(convert(Matrix{Bool}, outputs))
    return (final_inputs, final_outputs)
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Flux

indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);

function newClassCascadeNetwork(numInputs::Int, numOutputs::Int)
    if numOutputs == 1
        return Chain(Dense(numInputs, 1, σ))
    else
        return Chain(Dense(numInputs, numOutputs, identity), softmax)
    end
end;


function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=σ)
    outputLayer = previousANN[indexOutputLayer(previousANN)]; 
    previousLayers = previousANN[1:indexOutputLayer(previousANN)-1]; 
    numInputsOutputLayer = size(outputLayer.weight, 2); 
    numOutputsOutputLayer = size(outputLayer.weight, 1); 

    hidden = SkipConnection(Dense(numInputsOutputLayer, 1, transferFunction), (mx,x) -> vcat(x, mx));
    if numOutputsOutputLayer == 1
        output = Dense(numInputsOutputLayer + 1, 1, σ)
    else          
        output = Chain(
            Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, identity), softmax
        )
    end

    ann = Chain(
        previousLayers...,
        hidden, 
        output 
    )

    ann[indexOutputLayer(ann)].weight[:, 1:numInputsOutputLayer] .= outputLayer.weight 
    ann[indexOutputLayer(ann)].weight[:, end] .= 0 
    ann[indexOutputLayer(ann)].bias .= outputLayer.bias 

    return ann

end;


function trainClassANN!(ann::Chain, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)

    X, Y = trainingDataset
    X = Float32.(X)
    Y = Float32.(Y)
    
    function loss(model=ann, x=X, y=Y)

        if size(y, 1) == 1
            return Flux.Losses.binarycrossentropy(model(x), y)
        else
            return Flux.Losses.crossentropy(model(x), y)
        end
    end;

    initial_loss = Float32(loss(ann, X, Y))
    trainingLosses = Float32[initial_loss]

    opt_state = Flux.setup(Adam(learningRate), ann)

    if trainOnly2LastLayers && length(ann) > 2
        Flux.freeze!(opt_state.layers[1:(indexOutputLayer(ann)-2)])
    end

    for _ in 1:maxEpochs
        Flux.train!(loss, ann, [(X,Y)], opt_state)
        currentLoss = Float32(loss(ann, X, Y))
        push!(trainingLosses, currentLoss)

        if currentLoss <= minLoss
            break
        end

        if length(trainingLosses) >= lossChangeWindowSize
            lossWindow = trainingLosses[end - lossChangeWindowSize + 1:end]
            minLossValue, maxLossValue = extrema(lossWindow)
            if minLossValue > 0 && ((maxLossValue - minLossValue) / minLossValue <= minLossChange )
                break
            end
        end
    end

    return trainingLosses
    
end;


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    
    X, Y = trainingDataset
    X = permutedims(Float32.(X))
    Y = permutedims(Y)

    ann = newClassCascadeNetwork(size(X,1), size(Y,1))
    n_losses = trainClassANN!(ann, (X, Y), false;
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)
    
    for i in 1:maxNumNeurons
        ann = addClassCascadeNeuron(ann, transferFunction=transferFunction)
        
        if i > 1
            
            trainOnly2Last = true
            n2_losses = trainClassANN!(ann, (X, Y), trainOnly2Last;
                maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)
            n_losses = vcat(n_losses, n2_losses[2:end])
        end
        
        t_losses = trainClassANN!(ann, (X, Y), false;
            maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)
        
        n_losses = vcat(n_losses, t_losses[2:end])
    
    end

    return ann, n_losses

end;


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    
    X, Y_bool = trainingDataset
    Y = reshape(Y_bool, :, 1)

    return trainClassCascadeANN(maxNumNeurons, (X, Y);
        transferFunction=transferFunction,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        minLossChange=minLossChange,
        lossChangeWindowSize=lossChangeWindowSize)

end;
     

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

HopfieldNet = Array{Float32,2}

function trainHopfield(trainingSet::AbstractArray{<:Real,2})
    S = Float32.(trainingSet)
    N = size(S, 1)
    W = (S'*S)/Float32(N)
    for i in axes(W,1)     
        W[i, i] = 0
    end
    return W
end;


function trainHopfield(trainingSet::AbstractArray{<:Bool,2})
    S = (2 .* trainingSet) .-1
    trainHopfield(S)
end;


function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    N = size(trainingSetNCHW,1)
    S = reshape(trainingSetNCHW,N,:)
    return trainHopfield(S)
end;


function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    S = Float32.(S)
    out = ann * S
    return sign.(out)
end;


function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    S1 = 2 .* Float32.(S) .- 1
    out = stepHopfield(ann, S1)
    return out .>= 0
end;


function runHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    prev_S = nothing;
    prev_prev_S = nothing;
    while S!=prev_S && S!=prev_prev_S
        prev_prev_S = prev_S;
        prev_S = S;
        S = stepHopfield(ann, S);
    end;
    return S
end;


function runHopfield(ann::HopfieldNet, dataset::AbstractArray{<:Real,2})
    outputs = copy(dataset);
    for i in 1:size(dataset,1)
        outputs[i,:] .= runHopfield(ann, view(dataset, i, :));
    end;
    return outputs;
end;


function runHopfield(ann::HopfieldNet, datasetNCHW::AbstractArray{<:Real,4})
    outputs = runHopfield(ann, reshape(datasetNCHW, size(datasetNCHW,1), size(datasetNCHW,3)*size(datasetNCHW,4)));
    return reshape(outputs, size(datasetNCHW,1), 1, size(datasetNCHW,3), size(datasetNCHW,4));
end;


function addNoise(datasetNCHW::AbstractArray{<:Bool,4}, ratioNoise::Real)
    noiseSet = copy(datasetNCHW)
    indices = shuffle(1:length(noiseSet))[1:Int(round(length(noiseSet)*ratioNoise))];
    noiseSet[indices] .= .!noiseSet[indices]
    return noiseSet
end;


function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)
    dataset = copy(datasetNCHW)
    if ratioCrop == 0
        return dataset
    end

    _,_,_,W = size(dataset)
    first_column = Int(round(W* (1 - ratioCrop)) + 1)

    dataset[:,:,:,first_column:W] .= false
    return dataset

end;


function randomImages(numImages::Int, resolution::Int)
    images = randn(Float32, numImages, 1, resolution, resolution) .>= 0
    return images
end;


function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    labels = unique(labelArray)
    N = length(labels)
    _,C,H,W = size(imageArray)
    output_matrix = zeros(eltype(imageArray), N, C, H, W)

    for indexLabel in 1:N
        average = dropdims(mean(imageArray[labelArray.==labels[indexLabel], 1, :, :], dims=1), dims=1)
        output_matrix[indexLabel,:,:,:] = average
    end

    return (output_matrix, labels)
end;


function classifyMNISTImages(imageArray::AbstractArray{<:Bool,4}, templateInputs::AbstractArray{<:Bool,4}, templateLabels::AbstractArray{Int,1})
    N = size(imageArray, 1)
    outputs = fill(-1, N)
    for indexLabel in eachindex(templateLabels)
        label = templateLabels[indexLabel]
        template = templateInputs[[indexLabel], :, :, :];
        indicesCoincidence = vec(all(imageArray .== template, dims=[3,4]));
        outputs[indicesCoincidence] .= label
    end

    return outputs

end;


function calculateMNISTAccuracies(datasetFolder::String, labels::AbstractArray{Int,1}, threshold::Real)
    train_images, train_labels, test_images, test_labels = loadMNISTDataset(datasetFolder; labels=labels, datasetType=Float32)
    template_images, template_labels = averageMNISTImages(train_images, train_labels)
    
    train_images_bin    = train_images    .>= threshold
    test_images_bin     = test_images     .>= threshold
    template_images_bin = template_images .>= threshold
    
    hopfield_net = trainHopfield(template_images_bin)

    train_rec = runHopfield(hopfield_net, train_images_bin)
    train_pred = classifyMNISTImages(train_rec, template_images_bin, template_labels)

    test_rec = runHopfield(hopfield_net, test_images_bin)
    test_pred = classifyMNISTImages(test_rec, template_images_bin, template_labels)

    acc_train = mean(train_pred .== train_labels)
    acc_test  = mean(test_pred  .== test_labels)

    return (acc_train, acc_test)

end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# using ScikitLearn: @sk_import, fit!, predict
# @sk_import svm: SVC

using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
import Main.predict
predict(model, inputs::AbstractArray) = (outputs = MLJ.predict(model, MLJ.table(inputs)); return levels(outputs)[int(outputs)]; )


using Base.Iterators
using StatsBase

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    return batch[1]
end;

function batchTargets(batch::Batch)
    return batch[2]
end;

function batchLength(batch::Batch)
    return size(batchInputs(batch),1)
end;

function selectInstances(batch::Batch, indices::Any)
    inputs = batchInputs(batch)
    targets = batchTargets(batch)
    return (inputs[indices,:], targets[indices])
end;

function joinBatches(batch1::Batch, batch2::Batch)
    inputs = vcat(batchInputs(batch1), batchInputs(batch2))
    targets = vcat(batchTargets(batch1), batchTargets(batch2))
    return (inputs, targets)
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    n = batchLength(dataset)
    index = 1:n

    if shuffleRows
        index = Random.shuffle(index)
    end

    parts = Base.Iterators.partition(index, batchSize)
    batches = map(p -> selectInstances(dataset, collect(p)), parts)
    return batches
end;


function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )

    batch = joinBatches(supportVectors, dataset)
    inputs = batchInputs(batch)
    targets = batchTargets(batch)

    model = SVMClassifier(
        kernel =
        kernel=="linear" ? LIBSVM.Kernel.Linear :
        kernel=="rbf" ? LIBSVM.Kernel.RadialBasis :
        kernel=="poly" ? LIBSVM.Kernel.Polynomial :
        kernel=="sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing,
        cost = Float64(C),
        gamma = Float64(gamma),
        degree = Int32( degree),
        coef0 = Float64(coef0));
    
    if model.kernel === nothing
        error("Kernel no válido. Usa: linear, rbf, poly, sigmoid")
    end
        
    y_bool = categorical(targets .== 1)
    mach = machine(model, MLJ.table(inputs), y_bool)
    MLJBase.fit!(mach)


    indicesNewSupportVectors = sort( mach.fitresult[1].SVs.indices ); 

    #Índices de los vectoes de soporte en el dataset de entrenamiento
    n = size(supportVectors[1], 1)
    old_sv_indices = []
    new_sv_indices = []

    old_sv_indices = indicesNewSupportVectors[indicesNewSupportVectors .<= n]
    new_sv_indices = indicesNewSupportVectors[indicesNewSupportVectors .> n] .- n

    old_sv = selectInstances(supportVectors, old_sv_indices)
    new_sv = selectInstances(dataset, new_sv_indices)
    sv_batch = joinBatches(old_sv, new_sv)
            
    return (mach, sv_batch, (old_sv_indices, new_sv_indices))
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    supportVectors = (Array{Float64,2}(undef, 0, size(batches[1][1], 2)), Array{Float64,1}(undef, 0))

    mach = nothing
    for b in batches
        mach, supportVectors, _ = trainSVM(b, kernel, C;
            degree=degree, gamma=gamma, coef0=coef0,
            supportVectors=supportVectors)
    end

    return mach   
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    loadedData = loadStreamLearningDataset(datasetFolder)
    memory = selectInstances(loadedData, 1:windowSize)
    remainingData = selectInstances(loadedData, (windowSize+1):batchLength(loadedData))
    streamBatches = divideBatches(remainingData, batchSize; shuffleRows=false)
    return (memory, streamBatches)
end;

function addBatch!(memory::Batch, newBatch::Batch)
    x_mem, y_mem = memory
    x_new, y_new = newBatch 
    N = size(x_mem, 1)
    n = size(x_new, 1)
    
    x_mem[1:(N-n),:] .= x_mem[(1+n):N,:]
    y_mem[1:(N-n)] .= y_mem[(1+n):N]
    x_mem[(N-n+1):N,:] .= x_new
    y_mem[(N-n+1):N] .= y_new

    return memory
end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    memory, streamBatches = initializeStreamLearningData(datasetFolder, windowSize, batchSize);
    model, = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0);
    accuracies = Float64[];
    for i in streamBatches
        predictions = predict(model, batchInputs(i))
        accuracy = mean(predictions .== batchTargets(i))
        push!(accuracies, accuracy)
        addBatch!(memory, i)
        model, = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0);
    end
    return accuracies
end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
    X = dataset[1]
    diffs = X .- instance'
    dists = sqrt.(sum(diffs .^ 2, dims=2))
    return vec(dists)
end;

function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    batch_dists = euclideanDistances(dataset, instance)
    indices = partialsortperm(batch_dists, 1:k)
    return selectInstances(dataset, indices)
end;

function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    nearest_batch = nearestElements(dataset, instance, k)
    targets = batchTargets(nearest_batch)
    return mode(targets)
end;

function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    predictKnn(dataset, eachrow(instances), k)
end;


function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

function predictKNN_SVM(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int, C::Real)
    batch = nearestElements(dataset, instance, k)
    batchTargets = batchTargets(batch)

    # Si todos las clases más cercanas son iguales, devolver esa clase
    if length(unique(batchTargets)) == 1
        return batchTargets[1]
    end

    #Si no, entrenar un SVM con esas k instancias y predecir con ese SVM
    mach, _, _ = trainSVM(batch, "Linear", C)
    prediction = predict_mode(mach, reshape(instance, 1, :))[1]
    return prediction
end;

function predictKNN_SVM(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int, C::Real)
    # Para un conjunto de instancias 
    predictKNN_SVM(dataset, eachrow(instances), k, C)
    
end;
