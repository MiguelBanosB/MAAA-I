# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles
using JLD2
using Images
using ColorTypes


function fileNamesFolder(folderName::String, extension::String)    
    isdir(folderName) || error("The directory $folderName doesn't exist.");
    extension_upper = uppercase(extension);
    fileNames = filter(f -> endswith(uppercase(f), ".$extension_upper"), readdir(folderName));
    fileNamesNoExtension = replace.(fileNames, ".$extension_upper" => "");
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
    final_outputs = vec(convert(Matrix{Bool}, outputs))
    return (final_inputs, final_outputs)
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Flux

indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);

function newClassCascadeNetwork(numInputs::Int, numOutputs::Int)
    #
    # Codigo a desarrollar
    #
end;

function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=σ)
    outputLayer = previousANN[indexOutputLayer(previousANN)]; # Capa de salida de la red anterior
    previousLayers = previousANN[1:indexOutputLayer(previousANN)-1]; # Capas anteriores a la salida de la red anterior
    numInputsOutputLayer = size(outputLayer.weight, 2); # Número de entradas de la capa de salida de la red anterior. Weight es la matriz de pesos de la capa (Inputs, Outputs)
    numOutputsOutputLayer = size(outputLayer.weight, 1); # Número de salidas de la capa de salida de la red anterior

    if numOutputsOutputLayer == 1
        output = Dense(numInputsOutputLayer + 1, 1, σ)
    else          
        output = Chain(
            Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, identity), softmax
        )
    end

    ann = Chain(
        previousLayers...,
        SkipConnection(Dense(numInputsOutputLayer, 1, transferFunction), (mx,x) -> vcat(x, mx)), # Nueva neurona en cascada
        output 
    )

    ann[indexOutputLayer(ann)].weight[:, 1:numInputsOutputLayer] .= outputLayer.weight # Accede a la capa de salida, toma su matriz de pesos (las conexiones antiguas) y las copia a la nueva capa
    ann[indexOutputLayer(ann)].weight[:, end] .= 0 # Inicializa a cero las conexiones de la nueva neurona
    ann[indexOutputLayer(ann)].bias .= outputLayer.bias # Copia el bias de la salida antigua a la nueva capa

    return ann

end;

function trainClassANN!(ann::Chain, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    #
    # Codigo a desarrollar
    #
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
        
        if length(ann) > 1
            
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

    return trainClassCascadeANN(maxNumNeurons, (X, Ymat);
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
    #
    # Codigo a desarrollar
    #
end;
function trainHopfield(trainingSet::AbstractArray{<:Bool,2})
    #
    # Codigo a desarrollar
    #
end;
function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    #
    # Codigo a desarrollar
    #
end;

function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;
function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    #
    # Codigo a desarrollar
    #
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
    #
    # Codigo a desarrollar
    #
end;

function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)
    #
    # Codigo a desarrollar
    #
end;

function randomImages(numImages::Int, resolution::Int)
    #
    # Codigo a desarrollar
    #
end;

function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function classifyMNISTImages(imageArray::AbstractArray{<:Bool,4}, templateInputs::AbstractArray{<:Bool,4}, templateLabels::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function calculateMNISTAccuracies(datasetFolder::String, labels::AbstractArray{Int,1}, threshold::Real)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# using ScikitLearn: @sk_import, fit!, predict
# @sk_import svm: SVC

using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
import Main.predict
predict(model, inputs::AbstractArray) = MLJ.predict(model, MLJ.table(inputs));



using Base.Iterators
using StatsBase

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchTargets(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchLength(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function selectInstances(batch::Batch, indices::Any)
    #
    # Codigo a desarrollar
    #
end;

function joinBatches(batch1::Batch, batch2::Batch)
    #
    # Codigo a desarrollar
    #
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    #
    # Codigo a desarrollar
    #
end;

function addBatch!(memory::Batch, newBatch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;

function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    #
    # Codigo a desarrollar
    #
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
    #
    # Codigo a desarrollar
    #
end;

function predictKNN_SVM(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;
