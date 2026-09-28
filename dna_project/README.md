Predicción de Demanda DNA

Proyecto de Machine Learning orientado a la predicción de activación de demanda de las tipologías de atención de DNA, utilizando información histórica de atenciones, características demográficas y variables temporales.

El proyecto busca construir un modelo de clasificación capaz de identificar si una determinada tipología presentará demanda activa (1) o inactiva (0) en un período determinado.

1. Objetivo del proyecto

Desarrollar un modelo predictivo que permita anticipar la activación de demanda de las diferentes tipologías de atención, utilizando información histórica y variables derivadas.

El problema se plantea como una clasificación binaria:

0: demanda inactiva.
1: demanda activa.

La evaluación considera la naturaleza temporal de los datos, evitando mezclar información futura con información pasada durante el entrenamiento y la validación.

2. Metodología

El desarrollo del proyecto sigue las principales etapas de la metodología CRISP-DM:

Comprensión del negocio
        ↓
Comprensión de los datos
        ↓
Preparación de los datos
        ↓
Modelado
        ↓
Evaluación
        ↓
Despliegue / uso del modelo


Actualmente el proyecto se encuentra desarrollado principalmente en las etapas de:

Comprensión de los datos.
Preparación de los datos.
Modelado.
Evaluación comparativa de modelos.
3. Estructura del proyecto

La estructura actual del proyecto está organizada de la siguiente manera:

dna_project/
│
├── data/
│   ├── raw/
│   │   └── dataset_dna_demanda_b1.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── y_train.csv
│       ├── X_test.csv
│       ├── y_test.csv
│       ├── X_val.csv
│       └── y_val.csv
│
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── logistic_regression_model.py
│   ├── random_forest_model.py
│   ├── xgboost_model.py
│   └── neural_network_model.py
│
├── modeling/
│   ├── __init__.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   ├── model_comparator.py
│   └── roc_plotter.py
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_PreparacionDatos.ipynb
│   ├── 03_Modelado.ipynb
│   ├── 04_LR_Tuning.ipynb
│   └── 04_RF_Tuning.ipynb
│
├── results/
│
├── main_modeling.py
│
└── README.md

4. Datos

El dataset contiene registros asociados a:

Tipología de atención.
Gestión.
Mes.
Nivel de complejidad.
Cantidad de hombres atendidos.
Cantidad de mujeres atendidas.
Cantidad de niños y niñas atendidos.
Cantidad de adolescentes atendidos.
Total de atenciones.
Variable objetivo target_demanda.

Cada registro representa una combinación temporal de una tipología de atención.

5. Comprensión de los datos — EDA

La etapa de análisis exploratorio permite comprender la estructura, calidad y comportamiento de los datos.

Entre los análisis realizados se encuentran:

Calidad de datos
Dimensiones del dataset.
Tipos de variables.
Información general.
Valores faltantes.
Registros duplicados.
Cantidad de ceros.
Estadísticas descriptivas.
Variable objetivo

Se analiza la distribución de:

target_demanda


incluyendo:

Frecuencia de clases.
Distribución porcentual.
Distribución gráfica.
Posible desbalance de clases.
Análisis temporal

Se estudia la distribución de la demanda según:

Gestión.
Mes.
Tipología.

También se analiza la estacionalidad mensual de la activación de demanda.

Análisis demográfico

Se analiza el volumen de atenciones según:

Sexo.
Grupo etario.
Nivel de complejidad.
Análisis de valores atípicos

Se utiliza el método del Rango Intercuartílico (IQR) para identificar valores potencialmente atípicos en total_atenciones.

Debido a la naturaleza de los datos, los valores elevados de atenciones no se consideran automáticamente errores. Los períodos con alta cantidad de atenciones pueden representar precisamente situaciones reales de activación de demanda.

Por este motivo, los registros identificados como potenciales outliers son conservados para el modelado.

6. Preparación de los datos

La preparación de los datos contempla diferentes transformaciones orientadas a generar variables adecuadas para el modelo.

Limpieza

Se realizan:

Eliminación de duplicados.
Tratamiento de valores nulos.
Ordenamiento cronológico.
Conversión de variables categóricas.
Variables demográficas

Se calculan proporciones de:

Hombres.
Mujeres.
Niños y niñas.
Adolescentes.

Para evitar data leakage, estas variables son desplazadas temporalmente mediante shift(1).

De esta manera, el modelo utiliza información del período anterior para realizar la predicción del período actual.

Variables temporales

El mes se transforma mediante codificación cíclica:

mes_sin
mes_cos


Esto permite representar la naturaleza circular de los meses del año.

Variables de memoria temporal

Se generan variables históricas de atención:

atenciones_lag_1
atenciones_lag_2
atenciones_lag_3
promedio_movil_3m


Estas variables permiten incorporar información sobre el comportamiento reciente de cada tipología.

Variables categóricas

La variable:

nivel_complejidad


se transforma mediante One-Hot Encoding.

7. División temporal de los datos

Debido a que el problema tiene naturaleza temporal, no se realiza una división aleatoria convencional.

La división utilizada es:

ENTRENAMIENTO
2023 ───────── 2024
       ↓
   X_train
   y_train

TEST
2025
 ↓
X_test
y_test


El conjunto correspondiente a 2025 se mantiene separado para evaluar el comportamiento de los modelos sobre un período posterior al utilizado durante el entrenamiento.

8. Modelado

El modelado se encuentra implementado utilizando Programación Orientada a Objetos (POO).

Cada algoritmo dispone de su propia clase y archivo.

Actualmente se consideran cuatro modelos:

Regresión Logística.
Random Forest.
XGBoost.
Red Neuronal MLP.

Todos los modelos heredan de una clase base:

BaseModel


que establece una interfaz común para:

create_pipeline()
get_param_grid()
get_name()


Esto permite incorporar nuevos algoritmos sin modificar la lógica general del proceso.

9. Modelos utilizados
Regresión Logística

Se utiliza como modelo de referencia para el problema de clasificación.

Se incorpora:

StandardScaler
+
LogisticRegression


Los hiperparámetros evaluados son:

C
solver
Random Forest

Modelo basado en un conjunto de árboles de decisión.

Se utiliza:

class_weight="balanced"


para considerar el desbalance existente entre las clases.

Los hiperparámetros evaluados son:

n_estimators
max_depth
min_samples_split
XGBoost

Se utiliza XGBClassifier como modelo de boosting.

Debido al desbalance de clases se incorpora:

scale_pos_weight


Los hiperparámetros evaluados son:

n_estimators
max_depth
learning_rate
Red Neuronal MLP

Se utiliza MLPClassifier.

El pipeline incorpora:

StandardScaler
+
MLPClassifier


Los hiperparámetros evaluados son:

hidden_layer_sizes
alpha
10. Validación temporal

Para el ajuste de los modelos se utiliza:

TimeSeriesSplit(n_splits=3)


Esto permite realizar validación cruzada respetando el orden temporal de los datos.

El objetivo es evitar que información de períodos futuros sea utilizada para entrenar modelos destinados a predecir períodos anteriores.

11. Ajuste de hiperparámetros

El ajuste de hiperparámetros se realiza mediante:

GridSearchCV


La métrica utilizada como criterio de optimización es:

F1-Score


El uso de Pipeline permite que las transformaciones como StandardScaler sean ajustadas dentro de cada partición de entrenamiento, reduciendo el riesgo de data leakage durante la validación cruzada.

12. Evaluación

Los modelos optimizados se evalúan sobre el conjunto de prueba correspondiente a la gestión 2025.

Las métricas consideradas son:

Accuracy.
Precision.
Recall.
F1-Score.
ROC-AUC.
Tiempo de entrenamiento.

También se generan curvas ROC para realizar una comparación visual entre los modelos.

13. Arquitectura orientada a objetos

La arquitectura separa las responsabilidades principales del proceso.

models/
│
├── BaseModel
│
├── LogisticRegressionModel
├── RandomForestModel
├── XGBoostModel
└── NeuralNetworkModel

             ↓

modeling/
│
├── ModelTrainer
│      └── Entrenamiento + GridSearchCV
│
├── ModelEvaluator
│      └── Métricas
│
├── ModelComparator
│      └── Comparación de modelos
│
└── ROCCurvePlotter
       └── Visualización ROC


El archivo:

main_modeling.py


actúa como punto de entrada y coordina el flujo general.

14. Flujo de modelado

El flujo actual es:

Carga de datos
      ↓
Definición de validación temporal
      ↓
Selección de modelos
      ↓
Creación de pipelines
      ↓
GridSearchCV
      ↓
Entrenamiento
      ↓
Ajuste de hiperparámetros
      ↓
Evaluación sobre 2025
      ↓
Cálculo de métricas
      ↓
Comparación de modelos
      ↓
Generación de curvas ROC
      ↓
Exportación de resultados

15. Resultados generados

El proceso de modelado genera:

results/
│
├── resultados_modelado_7_4.csv
└── curvas_roc_7_4.png


El archivo CSV contiene la comparación de los modelos según sus principales métricas de clasificación y los hiperparámetros seleccionados.

16. Instalación

Se recomienda utilizar un entorno virtual de Python.

Crear el entorno:

python -m venv .venv


Activarlo en Windows:

.venv\Scripts\activate


Instalar las dependencias:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost

17. Ejecución

Una vez instaladas las dependencias y disponibles los datos preparados:

python main_modeling.py


El programa cargará los conjuntos de entrenamiento y prueba, entrenará los modelos, realizará el ajuste de hiperparámetros y generará los resultados correspondientes.

18. Tecnologías utilizadas
Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
19. Estado actual del proyecto
Etapa	Estado
Comprensión de los datos	✅ Completada
EDA	✅ Completada
Limpieza y preparación	✅ Completada
Feature Engineering	✅ Completada
División temporal	✅ Completada
Modelado	✅ Implementado
Validación temporal	✅ Implementada
Hyperparameter Tuning	✅ Implementado
Comparación de modelos	✅ Implementada
Selección formal del modelo final	⏳ Pendiente
Evaluación final	⏳ Pendiente
Despliegue	⏳ Pendiente
20. Próximos pasos

Como siguientes etapas del proyecto se contempla:

Definir formalmente el criterio de selección del modelo final.
Seleccionar el modelo con mejor desempeño según el criterio establecido.
Realizar la evaluación final del modelo seleccionado.
Analizar la importancia de las variables.
Generar predicciones para nuevos períodos.
Documentar las conclusiones del modelo.
Evaluar alternativas para el despliegue del modelo.