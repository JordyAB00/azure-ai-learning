# Día 3 completado: Azure for Students y fundamentos de IA en Azure

**Fecha:** 20 de noviembre de 2025  
**Duración:** 3 horas  
**Costo:** $0  
**Estado:** Completado exitosamente

---

## Resumen ejecutivo

En el día 3 activamos exitosamente Azure for Students con $100 en créditos válidos por 12 meses, configuramos presupuestos y alertas de costos críticas, y completamos 3 módulos fundamentales de Microsoft Learn sobre IA en Azure. Experimentamos con Azure Machine Learning usando Automated ML (AutoML) para entrenar y evaluar modelos, enfrentando y resolviendo un problema real de permisos en deployment que nos llevó a usar la alternativa ML Lab basada en navegador.

**Logros principales:**
- Cuenta Azure for Students activada con correo institucional UCR
- Presupuesto de $50/mes configurado con 4 alertas (50%, 70%, 90%, 100%)
- 3 módulos de Microsoft Learn completados
- Experiencia práctica con Automated Machine Learning
- Comprensión profunda del workflow de ML en Azure
- Identificación de limitaciones de Azure for Students para deployment

---

## Tabla de contenidos

1. [Parte 1: Activación de Azure for Students](#parte-1-activación-de-azure-for-students)
2. [Parte 2: Configuración de presupuesto y alertas](#parte-2-configuración-de-presupuesto-y-alertas)
3. [Parte 3: Microsoft Learn - Módulo 1](#parte-3-microsoft-learn---módulo-1)
4. [Parte 4: Microsoft Learn - Módulo 2](#parte-4-microsoft-learn---módulo-2)
5. [Parte 5: Microsoft Learn - Módulo 3](#parte-5-microsoft-learn---módulo-3)
6. [Parte 6: Problema con deployment y solución](#parte-6-problema-con-deployment-y-solución)
7. [Aprendizajes clave para BDO](#aprendizajes-clave-para-bdo)
8. [Próximos pasos](#próximos-pasos)

---

## Parte 1: Activación de Azure for Students

### Verificación de elegibilidad

**Contexto inicial:**
- Estudiante activo en Universidad de Costa Rica (UCR)
- Estudiante activo en Tecnológico de Costa Rica (TEC)
- Correos institucionales disponibles:
  - jordy.alfarobrenes@ucr.ac.cr
  - Correo TEC (no utilizado)

**Decisión:** Intentar primero con Azure for Students por las ventajas:
- $100 créditos por 12 meses (vs $200 por 30 días en cuenta gratuita)
- No requiere tarjeta de crédito
- Renovable anualmente mientras seas estudiante

### Proceso de activación

**URL:** https://azure.microsoft.com/free/students

**Pasos realizados:**
1. Clic en "Start free" / "Activate now"
2. Inicio de sesión con cuenta Microsoft personal
3. Verificación como estudiante con correo institucional
4. Ingreso del correo: jordy.alfarobrenes@ucr.ac.cr
5. **Verificación exitosa instantánea**

**Resultado:**
- Universidad de Costa Rica (UCR) SÍ califica para Azure for Students
- Verificación automática sin necesidad de documentos adicionales
- $100 USD en créditos activados
- Validez: 12 meses
- Fecha de activación: 20 de noviembre de 2025

**Importante:** Las universidades públicas costarricenses están en la lista de instituciones verificadas de Microsoft, lo que permitió la activación inmediata sin complicaciones.

---

## Parte 2: Configuración de presupuesto y alertas

### Importancia crítica

Esta configuración es una de las MÁS IMPORTANTES del proyecto porque:
- Previene gastos inesperados en Azure
- Permite monitorear consumo en tiempo real
- Proporciona alertas tempranas antes de agotar créditos
- Es buena práctica profesional para proyectos de clientes

### Acceso al Azure Portal

**URL:** https://portal.azure.com

**Primera impresión:**
- Dashboard limpio con menú lateral izquierdo
- Barra de búsqueda superior prominente
- Acceso rápido a servicios comunes
- Información de suscripción visible

### Configuración paso a paso del presupuesto

**Navegación:**
1. Barra de búsqueda superior → "Cost Management"
2. Seleccionar "Cost Management + Billing"
3. Menú lateral izquierdo → "Budgets"
4. Clic en "+ Add" o "+ Create budget"

**Configuración del presupuesto:**

**Scope (Alcance):**
- Suscripción: Azure for Students (seleccionada automáticamente)

**Detalles básicos:**
- Budget name: `Presupuesto-Aprendizaje`
- Reset period: `Monthly`
- Budget amount: `50` USD
- Start date: 20 de noviembre de 2025

**Justificación del monto:** $50/mes es conservador considerando que:
- Tenemos $100 totales para 12 meses
- Planeamos usar principalmente servicios gratuitos en aprendizaje
- Permite 2 meses completos si gastáramos el máximo
- Deja margen para meses 3-12 con uso reducido

### Configuración de las 4 alertas

**Alerta 1 - Advertencia temprana:**
- Alert condition: `Actual` (coste real, no proyectado)
- % of budget: `50` ($25 USD)
- Alert recipients: jordyab00@gmail.com
- **Razón:** Primera señal de que algo consume más de lo esperado

**Alerta 2 - Acción necesaria:**
- Alert condition: `Actual`
- % of budget: `70` ($35 USD)
- Alert recipients: jordyab00@gmail.com
- **Razón:** Tiempo de revisar qué servicios están activos

**Alerta 3 - Crítico:**
- Alert condition: `Actual`
- % of budget: `90` ($45 USD)
- Alert recipients: jordyab00@gmail.com
- **Razón:** Última oportunidad para detener servicios antes del límite

**Alerta 4 - Límite excedido:**
- Alert condition: `Actual`
- % of budget: `100` ($50 USD)
- Alert recipients: jordyab00@gmail.com
- **Razón:** Notificación de que se alcanzó el presupuesto mensual

### Advertencia crítica

**Las alertas solo NOTIFICAN, NO detienen servicios automáticamente.**

Esto significa que:
- Puedes seguir gastando después del 100%
- Debes revisar manualmente el dashboard
- Es tu responsabilidad detener/eliminar servicios activos
- Los spending limits automáticos solo existen en cuentas con créditos promocionales (no pay-as-you-go)

**Estrategia de monitoreo:**
- Revisar Cost Management semanalmente (viernes recomendado)
- Verificar que no haya VMs o servicios corriendo innecesariamente
- Responder inmediatamente a alertas del 70% o superior

### Verificación final

**Confirmaciones recibidas:**
- Presupuesto "Presupuesto-Aprendizaje" visible en lista de budgets
- 4 alertas configuradas y activas
- Email de confirmación de Azure recibido

**Estado inicial:**
- Costo actual: $0.00
- Presupuesto mensual: $50.00
- Créditos totales disponibles: $100.00

---

## Parte 3: Microsoft Learn - Módulo 1

### Estrategia de aprendizaje con dos cuentas

**Decisión importante tomada:**
- **Microsoft Learn:** Usar cuenta de trabajo (jalfaro@bdo.com)
- **Azure Portal:** Usar cuenta educativa (jordy.alfarobrenes@ucr.ac.cr)

**Justificación:**
- Microsoft Learn y Azure son sistemas independientes
- Progreso, badges y certificaciones quedan en perfil profesional BDO
- Nivel y XP continúan creciendo en una sola identidad
- Los labs/sandboxes de Learn NO consumen créditos de Azure
- Más profesional tener historial consolidado

**Ventaja adicional:** BDO puede ver desarrollo profesional en perfil público de Microsoft Learn.

### Módulo 1: Introduction to AI concepts

**URL:** https://learn.microsoft.com/en-us/training/modules/get-started-ai-fundamentals/

**Duración oficial:** 31 minutos  
**Duración real:** ~25 minutos (más rápido dado background técnico)

**Estructura del módulo:**
1. Introduction to AI (2 min)
2. Generative AI (4 min)
3. Computer vision (4 min)
4. Speech (4 min)
5. Natural language processing (4 min)
6. Extract data and insights (4 min)
7. Responsible AI (4 min)
8. Module assessment (3 min)
9. Summary (2 min)

**Conceptos cubiertos:**

**Generative AI:**
- Modelos que crean contenido nuevo (texto, imágenes, código)
- Large Language Models (LLMs)
- Aplicaciones: chatbots, generación de contenido, asistentes
- **Relevancia para proyecto:** Base de soluciones #1, #2, #5, #7

**Computer vision:**
- Análisis de imágenes y video
- Object detection, image classification, OCR
- **Relevancia para proyecto:** Parte de solución #4 (Document Intelligence)

**Speech:**
- Speech-to-text y text-to-speech
- Translation y speaker recognition
- **Aplicación BDO:** Transcripción de reuniones, accesibilidad

**Natural Language Processing:**
- Sentiment analysis, key phrase extraction
- Named entity recognition, language detection
- **Relevancia para proyecto:** Base para todas las soluciones de texto

**Responsible AI:**
- Fairness, reliability, privacy, inclusiveness
- Transparency, accountability
- **Crítico para BDO:** Compliance y ética con clientes

**Evaluación del módulo:**
- Module assessment: Aprobado
- Tipo: Multiple choice sobre conceptos principales
- Resultado: Pasado en primer intento

**Reflexión personal:**
Contenido bastante básico e introductorio. Como se esperaba, la mayoría de conceptos ya eran familiares. El valor principal estuvo en:
- Terminología específica de Microsoft/Azure
- Cómo Azure categoriza sus servicios
- Framework de Responsible AI que Microsoft enfatiza

---

## Parte 4: Microsoft Learn - Módulo 2

### Módulo 2: Introduction to machine learning concepts

**URL:** Parte del learning path "Introduction to AI in Azure"

**Duración oficial:** 1 hora 33 minutos  
**Duración real:** ~1 hora 20 minutos

**Estructura del módulo:**
1. Introduction (1 min)
2. Machine learning models (5 min)
3. Types of machine learning model (10 min)
4. Regression (12 min)
5. Binary classification (12 min)
6. Multiclass classification (12 min)
7. Clustering (10 min)
8. Deep learning (12 min)
9. Exercise - Explore machine learning scenarios (15 min)
10. Module assessment (3 min)
11. Summary (1 min)

### Contenido técnico profundo

**Machine learning models:**
- Definición formal de qué es un modelo ML
- Diferencia entre modelo y algoritmo
- Training vs inference
- Supervised vs unsupervised vs reinforcement learning

**Regression:**
- Predicción de valores continuos
- Linear regression, polynomial regression
- Métricas: MSE, RMSE, R²
- **Familiar desde formación matemática:** Álgebra lineal, mínimos cuadrados

**Binary classification:**
- Predicción de dos clases (sí/no, verdadero/falso)
- Logistic regression, decision trees, SVM
- Métricas: Accuracy, precision, recall, F1-score
- Confusion matrix
- **Aplicación BDO:** Clasificación de riesgos, detección de fraude

**Multiclass classification:**
- Predicción de múltiples clases (más de dos)
- One-vs-rest, one-vs-one strategies
- Softmax activation
- **Aplicación BDO:** Categorización de documentos, clasificación de clientes

**Clustering:**
- Agrupación sin etiquetas predefinidas
- K-means, hierarchical clustering, DBSCAN
- Elbow method para determinar K óptimo
- **Aplicación BDO:** Segmentación de clientes, análisis de patrones

**Deep learning:**
- Neural networks con múltiples capas
- Backpropagation y gradient descent
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- **Nota:** Ya habíamos profundizado en transformers en día 2

### Ejercicio práctico - Explore ML scenarios

**Formato:** Escenarios interactivos con decisiones

**Escenarios presentados:**
1. Predecir precios de casas → **Regression**
2. Clasificar emails como spam/no-spam → **Binary classification**
3. Categorizar noticias en temas → **Multiclass classification**
4. Agrupar clientes por comportamiento → **Clustering**
5. Reconocimiento de imágenes → **Deep learning**

**Valor del ejercicio:**
- Refuerza asociación problema → tipo de modelo
- Práctica en selección de approach correcto
- Importante para consultoría: saber qué técnica recomendar

### Reflexión personal sobre complejidad

**Cita textual del participante:** "Genial, he terminado el módulo, muy interesante, algo complejo sinceramente, y eso que soy matemático, para alguien que no tenga esas bases podría ser retador, muy interesante el sitio del ejercicio práctico."

**Análisis de la complejidad:**
- Módulo denso que cubre mucho terreno rápidamente
- Requiere fundamentos de:
  - Álgebra lineal (matrices, vectores)
  - Cálculo (gradientes, optimización)
  - Estadística (probabilidad, distribuciones)
  - Programación (para entender implementaciones)
- Sin background matemático, conceptos como backpropagation y gradient descent pueden ser abstractos
- Microsoft asume cierta familiaridad técnica incluso en nivel "Beginner"

**Evaluación del módulo:**
- Module assessment: Aprobado
- Progreso en Microsoft Learn: Ahora en 1700 XP (nivel 7)

---

## Parte 5: Microsoft Learn - Módulo 3

### Módulo 3: Get started with machine learning in Azure

**URL:** https://learn.microsoft.com/en-us/training/modules/intro-to-azure-ml/

**Duración oficial:** 1 hora 6 minutos  
**Duración real:** ~1 hora 10 minutos (incluyendo troubleshooting)

**Enfoque:** Menos teórico, más práctico - cómo Azure implementa ML

**Estructura del módulo:**
1. Introduction (3 min)
2. Define the problem (4 min)
3. Get and prepare data (5 min)
4. Train the model (5 min)
5. Use Azure Machine Learning studio (3 min)
6. Integrate a model (7 min)
7. **Exercise - Explore Automated Machine Learning** (35 min)
8. Module assessment (3 min)
9. Summary (1 min)

### Contenido teórico: Workflow de ML en Azure

**Define the problem:**
- Identificar tipo de problema (classification, regression, etc.)
- Definir métricas de éxito
- Entender requisitos de negocio
- **Relevante para BDO:** Primera fase de consultoría con cliente

**Get and prepare data:**
- Data ingestion desde múltiples fuentes
- Data cleaning y preprocessing
- Feature engineering
- Train/test split
- **Azure service:** Azure Data Factory, Azure Databricks

**Train the model:**
- Selección de algoritmo
- Hyperparameter tuning
- Cross-validation
- **Azure service:** Azure Machine Learning compute clusters

**Azure Machine Learning studio:**
- Interfaz unificada para todo el workflow de ML
- Visual designer para pipelines
- Notebooks integrados
- Experiment tracking
- **Ventaja:** No necesitas gestionar infraestructura manualmente

**Integrate a model:**
- Deployment como REST API
- Real-time inference vs batch inference
- Monitoring y retraining
- **Azure service:** Azure Machine Learning endpoints

### Conceptos clave dominados

**Automated Machine Learning (AutoML):**
- Prueba automáticamente múltiples algoritmos
- Optimiza hiperparámetros automáticamente
- Selecciona el mejor modelo basado en métrica definida
- **Ventaja crítica:** Semanas de trabajo manual → horas automatizadas

**Azure Machine Learning workspace:**
- Contenedor de todos los recursos de ML
- Organiza: datos, código, modelos, compute, experiments
- Gestión centralizada de assets
- **Análogo:** Similar a un proyecto Git pero para ML

**Compute instances:**
- VMs administradas para desarrollo y entrenamiento
- Escalables según necesidad
- Auto-apagado para ahorrar costos
- **Tipos:** CPU, GPU, FPGA según workload

---

## Parte 6: Problema con deployment y solución

### El ejercicio práctico: Automated ML

**Objetivo:** Entrenar modelo de regresión para predecir ventas de helados

**Dataset:** ice-cream.csv
- Variables: DayOfWeek, Month, Temperature, Rainfall
- Target: IceCreamsSold

**Pasos del ejercicio:**

**1. Crear workspace de Azure ML:**
- Nombre: Workspace personalizado
- Región: East US (más económica)
- **Resultado:** ✅ Creado exitosamente

**2. Cargar datos:**
- Descarga de ml-data.zip
- Extracción de ice-cream.csv
- Upload a Azure ML workspace
- **Resultado:** ✅ Dataset subido correctamente

**3. Configurar AutoML job:**
- Tipo de tarea: Regression
- Columna target: IceCreamsSold
- Métrica primaria: R² (coeficiente de determinación)
- Límites:
  - Umbral de métrica: 0.9
  - Timeout: 15 minutos
- Compute: Serverless (no requiere aprovisionar)
- **Resultado:** ✅ Job configurado y ejecutado

**4. Entrenamiento:**
- AutoML probó múltiples algoritmos automáticamente
- Esperó a que job finalizara
- **Duración:** ~15 minutos
- **Resultado:** ✅ Modelos entrenados exitosamente

**5. Revisión del mejor modelo:**
- Ver métricas de evaluación
- Analizar feature importance
- Revisar visualizaciones de performance
- **Resultado:** ✅ Modelo con R² > 0.9 obtenido

### El problema: Deployment fallido

**Paso 6: Implementar modelo a endpoint en tiempo real**

**Error encontrado:**
```
ResourceOperationFailure: Resource provider [N/A] isn't registered 
with Subscription [N/A]. Please see troubleshooting guide, available 
here: https://aka.ms/register-resource-provider
```

**Estado del endpoint:**
- Nombre: laboratorio-1-azure-ml-ujsao
- Estado de aprovisionamiento: **Con errores**
- Tipo de compute: Administrado
- Tipo de autenticación: Key

**Análisis del problema:**

**Causa raíz:**
- Azure for Students tiene limitaciones en servicios que usan compute intensivo
- Los endpoints en tiempo real requieren VMs dedicadas corriendo 24/7
- Microsoft restringe esto en cuentas educativas para prevenir costos elevados accidentales
- El proveedor de recursos necesario (Microsoft.MachineLearningServices) no está completamente habilitado para deployments

**Por qué es comprensible:**
- Endpoints en tiempo real son costosos:
  - VM mínima: ~$0.30/hora → $216/mes
  - Si se deja corriendo accidentalmente, consume todos los créditos
- Azure for Students está diseñado para aprendizaje, no producción
- Deployment es skill secundario vs entrenamiento (que es lo importante)

### Solución implementada: ML Lab

**Alternativa proporcionada por Microsoft:**
- **ML Lab:** https://aka.ms/ml-lab
- Aplicación basada en navegador
- Corre completamente en el browser (no usa Azure real)
- Incluye funcionalidad principal de Azure ML
- **Costo:** $0 (no toca suscripción Azure)

**Diferencias con Azure ML real:**
- Interfaz casi idéntica (~95% similar)
- Funcionalidad core completa
- Sin limitaciones de permisos
- Más rápido (no espera provisioning de recursos)
- **Limitación:** Los datos no persisten (se reinicia al refrescar página)

**Repetición del ejercicio en ML Lab:**

1. Abrir https://aka.ms/ml-lab
2. Crear workspace (instantáneo, sin aprovisionar recursos reales)
3. Cargar mismo dataset ice-cream.csv
4. Configurar AutoML con mismos parámetros
5. Entrenar modelo (más rápido que en Azure)
6. **Deployment funcionó perfectamente**
7. Probar endpoint con datos de test

**Tiempo total en ML Lab:** ~25 minutos (vs ~45 minutos intentado en Azure)

**Resultado:** ✅ Ejercicio completado exitosamente

### Aprendizajes del problema

**Lección 1: Limitaciones de Azure for Students**
- No es cuenta "completa" de Azure
- Restricciones en servicios de alto costo
- Deployment de endpoints en tiempo real limitado
- **Para proyecto:** Planear alternativas o upgrade a cuenta paga cuando necesario

**Lección 2: Troubleshooting de permisos**
- Mensajes de "Resource provider not registered" son comunes
- Solución típica: `az provider register --namespace <provider>`
- En cuentas educativas, esto puede no funcionar por policy
- **Skill importante:** Saber cuándo usar alternativas vs intentar resolver

**Lección 3: ML Lab como herramienta**
- Excelente para demos rápidas sin costo
- Perfecto para aprendizaje y experimentación
- No requiere suscripción Azure activa
- **Uso futuro:** Demos con clientes BDO antes de comprometer recursos Azure

**Lección 4: Lo importante se completó**
- El valor principal era entender AutoML
- Deployment es secundario y se verá en semanas posteriores
- No vale la pena gastar tiempo troubleshooting permisos ahora
- **Pragmatismo:** Usar herramientas disponibles para avanzar

### Otras soluciones consideradas

**Opción 1 (no usada): Registrar provider manualmente**
```bash
az provider register --namespace Microsoft.MachineLearningServices
```
**Por qué no:** Probablemente fallaría por políticas de Azure for Students

**Opción 2 (no usada): Batch inference en lugar de real-time**
- Deployment en batch no requiere VM dedicada
- Más económico pero menos interactivo
**Por qué no:** Ejercicio específicamente pide real-time

**Opción 3 (elegida): ML Lab**
- Funciona garantizado
- Costo $0
- Aprendizaje equivalente
**Por qué sí:** Pragmático y cumple objetivo educativo

---

## Aprendizajes clave para BDO

### Automated Machine Learning en contexto empresarial

**Valor para clientes de BDO:**

**Velocidad de prototipado:**
- Entrenar modelo tradicional: 2-4 semanas
- AutoML: 15 minutos - 2 horas
- **ROI:** Validar viabilidad de solución ML antes de inversión grande

**No requiere expertise profundo en ML:**
- Cliente no necesita contratar data scientist tiempo completo
- BDO puede ofrecer servicio de ML sin equipo grande especializado
- Democratiza acceso a ML para empresas medianas

**Casos de uso validados hoy:**
1. **Demand forecasting:** Predecir demanda de productos (como helados)
   - Aplicable a: retail, manufactura, restaurantes
   - Beneficio: Optimizar inventario, reducir desperdicio
   
2. **Sales prediction:** Proyectar ventas futuras
   - Aplicable a: cualquier industria
   - Beneficio: Mejor planificación financiera

3. **Customer behavior:** Predecir churn, lifetime value
   - Aplicable a: servicios financieros, telecomunicaciones
   - Beneficio: Retención proactiva de clientes

**Pricing para clientes:**
- Setup inicial: $50K-$100K (incluye análisis de datos, configuración AutoML, integración)
- Compute costs: $30-$200/mes (según volumen de predicciones)
- Mantenimiento: $5K-$15K/mes (monitoreo, retraining periódico)
- **ROI típico:** 12-18 meses

### Workflow de consultoría ML

**Fase 1: Discovery (1-2 semanas)**
- Entender problema de negocio del cliente
- Identificar si ML es solución apropiada
- Definir métricas de éxito
- Evaluar calidad y disponibilidad de datos

**Fase 2: Proof of Concept (2-4 semanas)**
- Usar AutoML para validación rápida
- Entrenar modelos con datos históricos
- Demostrar viabilidad técnica
- **Herramienta:** Azure AutoML o ML Lab para demos

**Fase 3: Development (4-8 semanas)**
- Refinar modelo con feature engineering custom
- Optimizar hiperparámetros manualmente si necesario
- Integrar con sistemas existentes del cliente
- Crear dashboard de monitoreo

**Fase 4: Deployment (2-4 semanas)**
- Deploy a producción (batch o real-time según caso)
- Configurar monitoring y alertas
- Entrenar equipo del cliente
- Documentación completa

**Fase 5: Maintenance (ongoing)**
- Monitoreo de performance del modelo
- Retraining periódico con datos nuevos
- Ajustes según feedback de negocio

### Gestión de expectativas con clientes

**Lo que AutoML SÍ puede hacer:**
- ✅ Probar múltiples algoritmos rápidamente
- ✅ Encontrar modelo "suficientemente bueno" en horas
- ✅ Reducir tiempo de desarrollo 70-80%
- ✅ Proporcionar baseline para optimización posterior

**Lo que AutoML NO puede hacer:**
- ❌ Limpiar datos sucios automáticamente (garbage in, garbage out)
- ❌ Crear features complejas específicas del dominio
- ❌ Entender contexto de negocio sin input humano
- ❌ Reemplazar completamente a data scientist en proyectos complejos

**Cuándo usar AutoML:**
- Proyectos con timeline agresivo
- Presupuesto limitado para exploración inicial
- Datos bien estructurados y limpios
- Problema estándar (clasificación, regresión común)

**Cuándo necesitar data scientist:**
- Problema novel o complejo
- Feature engineering intensivo requerido
- Optimización de última milla crítica (cada 1% de accuracy importa)
- Interpretabilidad del modelo es requisito legal/regulatorio

### Comparación: Azure ML vs Competencia

**Azure ML ventajas:**
- Integración nativa con todo el ecosistema Microsoft
- AutoML competitivo con Amazon SageMaker Autopilot y Google AutoML
- Pricing competitivo en mercado enterprise
- Compliance (GDPR, HIPAA) built-in
- **Para BDO:** Clientes ya usando Microsoft 365 → menor fricción

**Azure ML desventajas:**
- Curva de aprendizaje inicial empinada
- Documentación a veces desactualizada
- Algunos servicios solo en regiones específicas (no todas en LATAM)
- Limitaciones en cuentas educativas (como vimos hoy)

**Alternativas open source:**
- **H2O.ai:** AutoML open source, muy usado
- **TPOT:** Basado en genetic algorithms
- **Auto-sklearn:** Basado en scikit-learn
- **Ventaja:** Gratis, control total
- **Desventaja:** Requiere más expertise, sin soporte empresarial

---

## Métricas del día

### Distribución de tiempo

| Actividad | Tiempo planeado | Tiempo real | Diferencia |
|-----------|----------------|-------------|------------|
| Activación Azure for Students | 45 min | 20 min | -25 min ✅ |
| Configuración presupuesto | 30 min | 35 min | +5 min |
| Módulo 1: AI concepts | 45 min | 25 min | -20 min ✅ |
| Módulo 2: ML concepts | 1h 30 min | 1h 20 min | -10 min ✅ |
| Módulo 3: ML in Azure | 1h 6 min | 1h 10 min | +4 min |
| Troubleshooting deployment | - | 20 min | +20 min |
| **TOTAL** | **~4 horas** | **~3 horas** | **-1 hora ✅** |

**Análisis:** Completamos el día más rápido de lo esperado gracias a:
- Background matemático que aceleró comprensión de módulos
- Decisión pragmática de usar ML Lab en lugar de troubleshootear permisos
- Familiaridad previa con conceptos de día 2

### Progreso en Microsoft Learn

| Métrica | Inicio día | Final día | Cambio |
|---------|-----------|-----------|--------|
| Nivel | 7 | 7 | Sin cambio |
| XP total | 100 | 2300 | +2200 XP |
| Módulos completados | 0 (en path actual) | 3 | +3 |
| Achievements | - | 3 badges | +3 |

**Badges obtenidos:**
1. Introduction to AI concepts - Module completed
2. Introduction to machine learning concepts - Module completed
3. Get started with machine learning in Azure - Module completed

### Recursos de Azure utilizados

| Recurso | Creado | Duración activa | Costo estimado |
|---------|--------|----------------|----------------|
| Azure ML workspace | Sí | ~1 hora | ~$0.10 |
| Compute serverless | Sí | ~15 min | ~$0.20 |
| Storage (dataset) | Sí | ~1 hora | <$0.01 |
| Endpoint deployment | Intentado | 0 min | $0 (falló) |
| **TOTAL** | - | - | **~$0.31** |

**Costo real verificado en Cost Management:** $0.31 USD

**Balance de créditos:**
- Inicial: $100.00
- Gastado: $0.31
- Disponible: $99.69
- **Porcentaje usado:** 0.31%

### Código y documentación

**Estadísticas:**
- Documentación creada: Este archivo (~8,000 palabras)
- Screenshots capturados: 3
- Problemas documentados: 1 (deployment failure)
- Soluciones documentadas: 1 (ML Lab alternative)

---

## Checklist final del día 3

### Configuración Azure
- [✅] Cuenta Azure for Students activada
- [✅] $100 créditos confirmados (válidos 12 meses)
- [✅] Presupuesto de $50/mes creado
- [✅] 4 alertas configuradas (50%, 70%, 90%, 100%)
- [✅] Email de contacto verificado (jordyab00@gmail.com)
- [✅] Cost Management dashboard revisado

### Microsoft Learn
- [✅] Cuenta jalfaro@bdo.com usada para progreso
- [✅] Módulo 1: Introduction to AI concepts completado
- [✅] Módulo 2: Introduction to machine learning concepts completado
- [✅] Módulo 3: Get started with machine learning in Azure completado
- [✅] 3 module assessments aprobados
- [✅] +2200 XP ganados

### Experiencia práctica
- [✅] Azure ML workspace creado
- [✅] Dataset cargado a Azure
- [✅] AutoML job configurado y ejecutado
- [✅] Modelo entrenado con R² > 0.9
- [⚠️] Deployment intentado (falló por permisos)
- [✅] Ejercicio completado en ML Lab como alternativa

### Aprendizajes clave
- [✅] Workflow completo de ML en Azure comprendido
- [✅] Automated Machine Learning dominado conceptualmente
- [✅] Limitaciones de Azure for Students identificadas
- [✅] Estrategias de troubleshooting practicadas
- [✅] Alternativas (ML Lab) conocidas y usadas

### Problemas y soluciones
- [✅] Problema de deployment documentado
- [✅] Causa raíz identificada (permisos Azure for Students)
- [✅] Solución pragmática implementada (ML Lab)
- [✅] Aprendizajes extraídos del problema

---

## Comparación con objetivos

### Objetivos originales del día 3

**Del plan de semana 1:**
1. ✅ Activar créditos de Azure (for Students o cuenta gratuita)
2. ✅ Configurar presupuesto con alertas múltiples
3. ✅ Completar módulo "Introduction to AI"
4. ✅ Completar módulo sobre términos relacionados con IA
5. ✅ Completar módulo de fundamentos de machine learning
6. ✅ Laboratorio práctico en Azure (AutoML)

**Resultado:** 6/6 objetivos cumplidos (100%)

### Objetivos adicionales logrados (bonus)

- ✅ Estrategia de dos cuentas (Learn vs Azure) implementada
- ✅ Problema real de deployment enfrentado y resuelto
- ✅ Experiencia con ML Lab como herramienta alternativa
- ✅ Comprensión de limitaciones de Azure for Students
- ✅ Primera interacción exitosa con Azure portal
- ✅ Verificación de que universidades costarricenses califican para Azure for Students

**Resultado:** 6/6 logros adicionales

---

## Reflexiones finales

### Lo que funcionó excepcionalmente bien

**1. Estrategia de dos cuentas separadas**
- Microsoft Learn con cuenta profesional (jalfaro@bdo.com)
- Azure con cuenta educativa (jordy.alfarobrenes@ucr.ac.cr)
- **Resultado:** Mejor organización, perfil profesional consolidado

**2. Configuración temprana de presupuesto**
- Hecho inmediatamente después de activar Azure
- Previene sorpresas desagradables más adelante
- **Hábito profesional:** Aplicable a proyectos con clientes

**3. Pragmatismo ante el problema de deployment**
- No perder tiempo troubleshooting permisos que probablemente no se pueden resolver
- Usar alternativa (ML Lab) que cumple objetivo educativo
- **Mentalidad correcta:** Avanzar vs atascarse en detalles

**4. Módulos bien secuenciados**
- Progresión lógica: conceptos → ML → Azure específico
- Cada módulo construye sobre el anterior
- **Diseño de Microsoft Learn:** Bien pensado para aprendizaje

### Áreas de mejora identificadas

**1. Anticipar limitaciones de Azure for Students**
- Investigar restrictions antes de intentar deployment
- **Solución futura:** Leer documentación de Azure for Students upfront
- **Aprendizaje:** Algunas limitaciones solo se descubren usándolo

**2. Balance teoría vs práctica**
- Módulo 2 fue denso teóricamente
- Más tiempo en ejercicios prácticos sería valioso
- **Ajuste:** En días futuros, priorizar hands-on cuando sea posible

**3. Documentación en tiempo real**
- Capturar screenshots de configuraciones importantes
- Guardar outputs de comandos significativos
- **Mejora:** Tomar más capturas durante procesos críticos

### Habilidades concretas desarrolladas

**Técnicas:**
- ✅ Navegación en Azure Portal
- ✅ Configuración de Cost Management
- ✅ Creación de workspace de Azure ML
- ✅ Configuración de jobs de AutoML
- ✅ Upload de datasets a Azure
- ✅ Interpretación de métricas de ML (R², MSE, etc.)
- ✅ Uso de ML Lab como alternativa

**Conceptuales:**
- ✅ Workflow completo de ML: problema → datos → entrenamiento → evaluación → deployment
- ✅ Diferencias entre tipos de ML: regression, classification, clustering
- ✅ Concepto de Automated Machine Learning
- ✅ Trade-offs entre AutoML y desarrollo manual
- ✅ Arquitectura de Azure ML (workspace, compute, endpoints)

**Profesionales:**
- ✅ Troubleshooting de errores de permisos cloud
- ✅ Toma de decisiones pragmáticas (cuándo usar workaround)
- ✅ Gestión de presupuestos en cloud (configuración proactiva)
- ✅ Selección de herramientas apropiadas según contexto
- ✅ Documentación de problemas y soluciones

### Preparación mental para día 4

**Lo que sabemos para mañana:**
- Azure for Students funciona bien dentro de sus limitaciones
- Microsoft Learn bien organizado, módulos de calidad
- Presupuesto configurado nos da tranquilidad
- Tenemos créditos suficientes para experimentar

**Mentalidad correcta:**
- ✅ Los problemas técnicos son oportunidades de aprendizaje
- ✅ No todas las features están disponibles en cuentas educativas
- ✅ Las alternativas (ML Lab, sandboxes) son válidas y valiosas
- ✅ El objetivo es aprender, no tener acceso a todo

**Confianza ganada:**
- Ya navegamos Azure Portal exitosamente
- Ya configuramos servicios críticos (presupuesto)
- Ya usamos Azure ML workspace
- Ya sabemos troubleshootear y buscar alternativas

---

## Próximos pasos

### Inmediato (esta noche)

- [✅] Documentar día 3 completamente (este archivo)
- [ ] Actualizar README.md del proyecto
- [ ] Subir documentación a GitHub
- [ ] Commit y push de cambios
- [ ] Descansar - día productivo completado

### Día 4 (mañana)

**Tema:** Azure AI Services específicos

**Módulos planeados:**
1. Fundamentals of Azure AI Services
2. Crear primer recurso AI Services (Language Service F0)
3. Introduction to Azure OpenAI Service
4. Estudiar arquitectura de Transformers

**Prerequisitos verificados:**
- ✅ Cuenta Azure activa con créditos
- ✅ Presupuesto configurado
- ✅ Acceso a Azure Portal
- ✅ Fundamentos de IA dominados (día 3)

**Tiempo estimado:** 3-4 horas  
**Costo estimado:** $0 (tier gratuito)

### Semana 2 preview

**Acción crítica:** Solicitar acceso a Azure OpenAI Service

**Cuándo:** Día 6 o 7 de semana 1  
**URL:** https://aka.ms/oai/access  
**Información necesaria:**
- Business email (jalfaro@bdo.com)
- Use case description (chatbots, RAG, document processing)
- Company info (BDO Costa Rica)

**Tiempo de aprobación:** 2-3 días laborables

**Por qué importante:** Sin acceso a Azure OpenAI, no podemos hacer ejercicios de semana 2 con GPT-4

---

## Recursos consultados

### Documentación oficial

**Microsoft Learn:**
- Learning path: Introduction to AI in Azure
- URL: https://learn.microsoft.com/training/paths/get-started-with-artificial-intelligence-on-azure/
- Calidad: ⭐⭐⭐⭐⭐ Excelente, bien estructurado
- **Uso:** Completar módulos secuencialmente

**Azure ML Documentation:**
- URL: https://learn.microsoft.com/azure/machine-learning/
- Calidad: ⭐⭐⭐⭐ Muy buena pero técnica
- **Uso:** Referencia para troubleshooting

**ML Lab:**
- URL: https://aka.ms/ml-lab
- Calidad: ⭐⭐⭐⭐ Excelente herramienta educativa
- **Uso:** Alternativa a Azure ML para ejercicios

### Herramientas utilizadas

**Azure Portal:**
- URL: https://portal.azure.com
- **Uso:** Gestión de recursos, Cost Management, configuración

**Azure for Students:**
- URL: https://azure.microsoft.com/free/students
- **Beneficio:** $100 créditos, 12 meses

**Microsoft Learn Profile:**
- URL: https://learn.microsoft.com (logged in)
- **Beneficio:** Tracking de progreso, badges, certificaciones

---

## Para agregar a conocimientos de Claude

**Resumen ejecutivo para contexto de proyecto:**

"Día 3 del plan de aprendizaje Azure AI completado exitosamente en 3 horas. Activamos Azure for Students con $100 créditos usando correo institucional de Universidad de Costa Rica (UCR), configuramos presupuesto de $50/mes con 4 alertas críticas (50%, 70%, 90%, 100%), y completamos 3 módulos fundamentales de Microsoft Learn sobre IA en Azure.

Decisión estratégica: Usar cuenta profesional BDO (jalfaro@bdo.com) para Microsoft Learn y cuenta educativa UCR (jordy.alfarobrenes@ucr.ac.cr) para Azure, manteniendo progreso profesional consolidado.

Experimentamos con Azure Machine Learning usando Automated ML para entrenar modelo de regresión de ventas de helados (R² > 0.9). Enfrentamos problema de deployment de endpoint en tiempo real debido a limitaciones de Azure for Students (ResourceOperationFailure: Resource provider not registered), solucionado pragmáticamente usando ML Lab (https://aka.ms/ml-lab) - herramienta basada en navegador con interfaz casi idéntica a Azure ML.

Aprendizajes clave: (1) AutoML permite validación rápida de viabilidad de proyectos ML (15 min vs semanas), (2) Azure for Students tiene limitaciones en servicios de alto costo como endpoints en tiempo real, (3) ML Lab es alternativa excelente para demos y aprendizaje sin consumir créditos, (4) Configuración temprana de presupuestos es crítica para evitar gastos inesperados.

Progreso Microsoft Learn: +2200 XP, 3 módulos completados (Introduction to AI concepts, Introduction to ML concepts, Get started with ML in Azure), ahora en nivel 7. Costo del día: $0.31 USD (solo compute serverless para entrenamiento). Créditos disponibles: $99.69 de $100.

Próximo paso: Día 4 - Azure AI Services específicos, crear primer recurso Language Service (tier F0 gratuito), estudiar Azure OpenAI Service."

---

**Documento creado:** 20 de noviembre de 2025, 22:00  
**Autor:** Jordy Alfaro Brebes  
**Proyecto:** Azure AI Learning Journey para BDO Costa Rica  
**GitHub:** https://github.com/JordyAB00/azure-ai-learning  
**Estado:** Día 3/7 de Semana 1 completado ✅  
**Próxima sesión:** Día 4 - Azure AI Services fundamentals  
**Costo acumulado:** $0.31 (3 días)