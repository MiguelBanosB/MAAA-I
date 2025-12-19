# Carga de paquetes, constantes y definición de modelos

# Herramientas básicas de manipulación de datos y cálculo
using CSV
using DataFrames
using Glob
using Statistics
using StatsBase
using Random
using LinearAlgebra
using JLD2
using Printf

# Herramientas de análisis estadístico
using HypothesisTests
using Combinatorics # Para generar pares de modelos
using Distributions  # Necesario para calcular el p-valor (Chisq)

# Herramientas de visualización
using PrettyTables
using Plots
using Plots.PlotMeasures
using StatsPlots
using TSne
using ManifoldLearning

# Configuración del backend de gráficos
gr()

# Núcleo del framework de Machine Learning
using MLJ
using MLJBase
import MLJModelInterface as MMI

# Backends y librerías de algoritmos
import MultivariateStats
import NearestNeighborModels
import LIBSVM
import MLJLinearModels
import DecisionTree
import EvoTrees
import LightGBM
import XGBoost

# Librerías para Redes Neuronales
using Flux
import MLJFlux

# Definición de constantes del proyecto
const SEED = 104
const DATA_PATH = "Datos Práctica"
const N_FEATURES = 561

# Carga de modelos lineales y basados en distancias
LogisticClassifier      = MLJ.@load LogisticClassifier pkg=MLJLinearModels verbosity=0
ProbabilisticSVC        = MLJ.@load ProbabilisticSVC pkg=LIBSVM verbosity=0
KNNClassifier           = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0

# Carga de modelos basados en árboles y ensembles
DecisionTreeClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0
RandomForestClassifier  = MLJ.@load RandomForestClassifier pkg=DecisionTree verbosity=0
AdaBoostStumpClassifier = MLJ.@load AdaBoostStumpClassifier pkg=DecisionTree verbosity=0
EvoTreeClassifier       = MLJ.@load EvoTreeClassifier pkg=EvoTrees verbosity=0
XGBoostClassifier       = MLJ.@load XGBoostClassifier pkg=XGBoost verbosity=0
LGBMClassifier          = MLJ.@load LGBMClassifier pkg=LightGBM verbosity=0

# Carga de modelos de redes neuronales
NeuralNetworkClassifier = MLJ.@load NeuralNetworkClassifier pkg=MLJFlux verbosity=0

# Carga de algoritmos de reducción de dimensionalidad
PCA = MLJ.@load PCA pkg=MultivariateStats verbosity=0
ICA = MLJ.@load ICA pkg=MultivariateStats verbosity=0
LDA = MLJ.@load LDA pkg=MultivariateStats verbosity=0

println("Entorno y modelos cargados correctamente")