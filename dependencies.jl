# Archivo de dependencias y configuración inicial

# Librerías Estándar
using CSV               # Para leer archivos .csv 
using DataFrames        # Para manipular tablas de datos
using Glob              # Para búsqueda de archivos con patrones
using Statistics        # Funciones estadísticas básicas
import Statistics: mean, cor, middle, quantile
using StatsBase         # Funciones estadísticas adicionales
using Random            # Para generación de números aleatorios y semillas
using LinearAlgebra     # Operaciones matriciales (necesario para la regresión en RFE)
using PrettyTables      # Para imprimir tablas de resultados formateadas en consola
using JLD2              # Para guardar/cargar variables y estados entre notebooks (.jld2)


# MLJ 
using MLJ            
using MLJBase 
import MLJModelInterface as MMI # Interfaz de bajo nivel para crear Wrappers personalizados


# Algoritmos
import MultivariateStats      # PCA, ICA, LDA
import NearestNeighborModels  # KNN
import LIBSVM                 # SVM
using Flux
import MLJFlux
import DecisionTree 
using MLJLinearModels # Necesario para LogisticClassifier
import EvoTrees
import MLJScikitLearnInterface

# Constantes Globales
const SEED = 104                   # Semilla fijada según el enunciado para reproducibilidad
const DATA_PATH = "Datos Práctica" # Ruta relativa a la carpeta de datos


# Carga de Wrappers
LogisticClassifier = MLJ.@load LogisticClassifier pkg=MLJLinearModels verbosity=0
PCA = MLJ.@load PCA pkg=MultivariateStats verbosity=0
ICA = MLJ.@load ICA pkg=MultivariateStats verbosity=0
LDA = MLJ.@load LDA pkg=MultivariateStats verbosity=0

KNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
ProbabilisticSVC = MLJ.@load ProbabilisticSVC pkg=LIBSVM verbosity=0
NeuralNetworkClassifier = MLJ.@load NeuralNetworkClassifier pkg=MLJFlux verbosity=0

# Cargar modelos de árboles
DecisionTreeClassifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0
RandomForestClassifier = MLJ.@load RandomForestClassifier pkg=DecisionTree verbosity=0
EvoTreeClassifier = MLJ.@load EvoTreeClassifier pkg=EvoTrees verbosity=0

# Para AdaBoost con SVM, necesitamos traerlos de ScikitLearn
# Nota: "pkg=MLJScikitLearnInterface" conecta Julia con la librería de Python
AdaBoostClassifier = MLJ.@load AdaBoostClassifier pkg=MLJScikitLearnInterface verbosity=0
SKSGDClassifier = MLJ.@load SGDClassifier pkg=MLJScikitLearnInterface verbosity=0