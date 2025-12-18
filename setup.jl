# Este archivo se encarga de cargar librerías, constantes y modelos.


# --- 1. Herramientas de Datos y Estadística ---
using CSV               # Lectura de datasets
using DataFrames        # Manipulación de tablas de datos
using Glob              # Búsqueda de archivos (para cargar los sujetos)
using Statistics        # Funciones base (mean, std, cor)
using StatsBase         # Funciones estadísticas extra
using Random            # Control de aleatoriedad (semillas)
using LinearAlgebra     # Operaciones matriciales (necesario para RFE)
using JLD2              # Persistencia de datos (guardar/cargar variables preprocesadas)
using PrettyTables      # Visualización estética de tablas en consola
using Plots
gr()

# --- 2. Núcleo MLJ  ---
using MLJ               # Framework principal
using MLJBase           # Funcionalidades base de MLJ
import MLJModelInterface as MMI # Para crear Wrappers personalizados (Filtros)


# --- 3. Librerías de Algoritmos Específicos ---
import MultivariateStats      # Proyecciones: PCA, ICA, LDA
import NearestNeighborModels  # KNN
import LIBSVM                 # SVM (Support Vector Machines)
import MLJLinearModels        # Modelos lineales (Logistic Regression para RFE)
import DecisionTree           # Random Forest y Árboles de decisión
import EvoTrees               # Gradient Boosting (EvoTrees)
using Flux                    # Framework de Redes Neuronales
import MLJFlux                # Interfaz de MLJ para Flux
import MLJScikitLearnInterface # Puente con ScikitLearn (Python) para AdaBoost

import CondaPkg

# --- 4. Constantes Globales ---
const SEED = 104              # Semilla fijada por enunciado para reproducibilidad
const DATA_PATH = "Datos Práctica" # Directorio de los datos crudos
const N_FEATURES = 561        # Número fijo de atributos del dataset HAR


# --- 5. Carga de Modelos ---
LogisticClassifier      = MLJ.@load LogisticClassifier pkg=MLJLinearModels verbosity=0
ProbabilisticSVC        = MLJ.@load ProbabilisticSVC pkg=LIBSVM verbosity=0
KNNClassifier           = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
NeuralNetworkClassifier = MLJ.@load NeuralNetworkClassifier pkg=MLJFlux verbosity=0

DecisionTreeClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0
RandomForestClassifier  = MLJ.@load RandomForestClassifier pkg=DecisionTree verbosity=0
EvoTreeClassifier       = MLJ.@load EvoTreeClassifier pkg=EvoTrees verbosity=0
AdaBoostStumpClassifier = MLJ.@load AdaBoostStumpClassifier pkg=DecisionTree verbosity=0

# Modelos vía ScikitLearn (Python)
SKSGDClassifier         = MLJ.@load SGDClassifier pkg=MLJScikitLearnInterface verbosity=0

# Paquetes de Boosting 
XGBoostClassifier = MLJ.@load XGBoostClassifier pkg=XGBoost verbosity=0
LGBMClassifier    = MLJ.@load LGBMClassifier pkg=LightGBM verbosity=0
CatBoostClassifier = MLJ.@load CatBoostClassifier pkg=CatBoost verbosity=0

# Transformaciones de Reducción de Dimensionalidad
PCA = MLJ.@load PCA pkg=MultivariateStats verbosity=0
ICA = MLJ.@load ICA pkg=MultivariateStats verbosity=0
LDA = MLJ.@load LDA pkg=MultivariateStats verbosity=0

println("Entorno cargado correctamente. Constantes: SEED=$SEED, N_FEATURES=$N_FEATURES")