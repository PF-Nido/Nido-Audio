# Diseño Metodológico: Reconocimiento de Aves en Colombia con Fusión de Datos y MLOps

Este documento resume las decisiones arquitectónicas y metodológicas para el proyecto de grado, enfocadas en la clasificación de cantos de aves mediante embeddings acústicos y contexto espaciotemporal.

## 1. El Problema a Resolver
Un clasificador acústico puro puede predecir falsos positivos geográficos (ej. predecir una especie endémica de la Amazonía en los páramos andinos). Se requiere un sistema de **Fusión de Características (Feature Fusion)** que integre la información del nicho ecológico.

Los datos a utilizar serán:
- **Audio:** Grabaciones extraídas de Xeno-Canto.
- **Acústica:** Vectores densos (Embeddings) extraídos usando el modelo base de BirdNET.
- **Tabulares Espaciotemporales:** Elevación (msnm), Hora del día, Día del año.

## 2. Arquitectura Seleccionada: Early Fusion con XGBoost + PCA

Se eligió una fusión temprana de datos utilizando un modelo de ensamble de árboles de decisión (XGBoost). 

### Justificación Técnica:
1. **Manejo de variables mixtas:** XGBoost es el estado del arte para procesar datos estructurados (Elevación, Tiempo), pero es ineficiente procesando vectores de 1024 dimensiones (Embeddings puros).
2. **Reducción de Dimensionalidad (PCA):** Para evitar el sobreajuste (*overfitting*), los embeddings de BirdNET se comprimen usando Análisis de Componentes Principales (PCA), conservando (ej. 50-100) dimensiones fundamentales.
3. **Ingeniería de Características (Feature Engineering):** Variables cíclicas como la hora y el día del año se transformarán en coordenadas trigonométricas (Seno/Coseno) para que el algoritmo comprenda la continuidad del tiempo.
4. **Interpretabilidad (Feature Importance):** Permitirá graficar en la tesis el porcentaje de decisión que tomó el modelo basándose en la acústica vs. la geografía, algo de alto valor para análisis biológico.
5. **Solución matemática:** Evita el "Problema del Veto" de la fusión tardía, analizando de manera conjunta cómo suena y dónde está el ave en un mismo árbol.

## 3. Pipeline de Entrenamiento Continuo (MLOps / Human-in-the-Loop)

Para escalar el sistema y permitir validaciones ciudadanas sin corromper el modelo (*Data Poisoning*), se implementará un flujo de Aprendizaje Activo (*Active Learning*):

1. **Inferencia & Pseudo-etiquetado:** El usuario recibe una predicción del modelo y la confirma/corrige. 
2. **Base de Datos Staging:** Los audios nuevos van a una base de datos temporal (cuarentena), no directo al modelo.
3. **Filtro de Calidad Automático:** Se descartan audios cortos o con metadata geográfica vacía.
4. **Entrenamiento Incremental (Continual Learning):** En lugar de re-entrenar con los 20,000 datos originales, se utilizará el parámetro `xgb_model` de XGBoost para inyectar *batches* (tandas) de audios nuevos validados.
5. **Evaluación Continua:** El modelo actualizado se compara contra un *"Dataset Dorado"* (Golden Dataset) reservado para métricas de validación.

## 4. Notas sobre Extracción de Datos (Xeno-Canto API)
Para optimizar la descarga desde la API v3 de Xeno-Canto:
- Siempre consultar la página 1 primero (`numRecordings`, `numPages`) para verificar la disponibilidad de datos.
- Filtrar por calidad utilizando etiquetas como `q:>C` (Calidades A y B).
- Extraer coordenadas y utilizar una API de Elevación externa para mapear Lat/Lon a msnm (metros sobre el nivel del mar).
