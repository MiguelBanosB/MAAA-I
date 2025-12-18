# Carga de paquetes, constantes y definición de modelos

# Importación de herramientas básicas de datos y cálculo
using CSV              # Lectura y escritura de archivos CSV
using DataFrames       # Manipulación y análisis de datos tabulares
using Glob             # Búsqueda de archivos y rutas mediante patrones
using Statistics       # Funciones estadísticas básicas (media, varianza, etc.)
using StatsBase        # Estadística avanzada y utilidades (muestreo, pesos, métricas)
using Random           # Generación y control de números aleatorios y semillas
using LinearAlgebra    # Operaciones algebraicas y matriciales
using JLD2             # Guardado y carga eficiente de datos en formato binario Julia
using HypothesisTests  # Contrastes de hipótesis estadísticas (t-test, Wilcoxon, etc.)
using StatsPlots       # Visualización estadística integrada con Plots.jl
using Printf           # Formateo avanzado de strings y salida por pantalla


# Importación de herramientas de visualización
using PrettyTables                     # Mostrar tablas formateadas en consola y notebooks
using Plots
using Plots.PlotMeasures                            # Librería base para generación de gráficas
using TSne: tsne                       # Implementación del algoritmo t-SNE para reducción dimensional
using ManifoldLearning                 # Métodos de aprendizaje de variedades (LLE, Isomap, etc.)
gr()                                  # Selección del backend GR para Plots (rápido y ligero)

# Importación del núcleo del framework de Machine Learning
using MLJ                              # Framework unificado de machine learning en Julia
using MLJBase                          # Funcionalidades base de MLJ (evaluación, medidas, utilidades)
import MLJModelInterface as MMI        # Interfaz para definir modelos compatibles con MLJ

# Importación de backends y librerías de algoritmos
import MultivariateStats               # Algoritmos estadísticos y de reducción dimensional (PCA, etc.)
import NearestNeighborModels           # Modelos basados en vecinos más cercanos (k-NN)
import LIBSVM                          # Máquinas de Vectores de Soporte (SVM)
import MLJLinearModels                 # Modelos lineales (logística, regresión, etc.)
import DecisionTree                    # Árboles de decisión y Random Forest
import EvoTrees                        # Gradient Boosting basado en árboles
import LightGBM                        # Implementación eficiente de Gradient Boosting (LightGBM)
using XGBoost                          # Algoritmo XGBoost para clasificación y regresión
using Flux                             # Framework de deep learning en Julia
import MLJFlux                         # Integración de modelos Flux dentro del framework MLJ
import MLJScikitLearnInterface         # Acceso a modelos de scikit-learn desde MLJ


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

println("Entorno cargado correctamente")