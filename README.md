# Azure AI Learning Journey

Documentación de mi aprendizaje en Azure AI Services y desarrollo de soluciones de IA para BDO Costa Rica.

## Progreso general

**Días completados:** 3/7 (Semana 1)  
**Tiempo invertido:** ~12 horas  
**Costo acumulado:** $0.31  
**Última actualización:** 20 de noviembre de 2025, 22:00

---

## Semana 1: Fundamentos y configuración

### ✅ Día 1 completado (19 nov 2025)
- Python 3.14.0 instalado y verificado
- VS Code configurado con extensiones de Azure y Python
- Git y GitHub configurados con SSH authentication
- Ollama instalado con modelo llama3.2:3b
- Primer script en Python interactuando con LLM local
- Ambiente virtual creado y funcionando
- Repositorio en GitHub inicializado

**Documentación:** [Ver día 1 completo](./documentacion/Dia_1_completado.md)

### ✅ Día 2 completado (20 nov 2025)
- Google Colab configurado con GPU T4 gratuita
- Cuenta de Hugging Face creada y explorada
- Modelos probados: GPT-2, Mistral-7B, BERT multilingual
- Análisis de sentimientos implementado en español
- Conceptos fundamentales dominados: tokenización, embeddings, attention
- 2 notebooks funcionales creados
- Problemas reales resueltos: compatibilidad Phi-3, gestión memoria GPU

**Modelos comparados:**
| Modelo | Calidad español | Estado |
|--------|----------------|--------|
| GPT-2 | ⭐ Malo | ✅ Funcionó (limitado) |
| Mistral-7B | ⭐⭐⭐⭐⭐ Excelente | ✅ Funcionó perfecto |
| Phi-3-mini | - | ❌ Error compatibilidad |
| BERT Sentiment | ⭐⭐⭐⭐⭐ Excelente | ✅ Funcionó perfecto |

**Documentación:** [Ver día 2 completo](./documentacion/Dia_2_completado.md)

### ✅ Día 3 completado (20 nov 2025)
- Azure for Students activado ($100 créditos, 12 meses)
- Cuenta UCR (jordy.alfarobrenes@ucr.ac.cr) verificada exitosamente
- Presupuesto $50/mes configurado con 4 alertas críticas
- Estrategia de dos cuentas: Learn (BDO) + Azure (UCR)
- 3 módulos Microsoft Learn completados (+2200 XP)
- Azure Machine Learning workspace creado
- Automated ML experimentado (modelo R² > 0.9)
- Problema de deployment resuelto con ML Lab
- Limitaciones de Azure for Students identificadas

**Módulos completados:**
1. Introduction to AI concepts (31 min)
2. Introduction to machine learning concepts (1h 33 min)
3. Get started with machine learning in Azure (1h 6 min)

**Costo del día:** $0.31 (compute serverless para AutoML)

**Documentación:** [Ver día 3 completo](./documentacion/Dia_3_completado.md)

### 📋 Próximos días

**Día 4:** Azure AI Services fundamentals (pendiente)
- Fundamentals of Azure AI Services
- Crear primer recurso Language Service (tier F0)
- Introduction to Azure OpenAI Service
- Arquitectura de Transformers en profundidad

**Día 5:** Prompt engineering avanzado (pendiente)
**Día 6:** APIs REST y Azure OpenAI (pendiente)
**Día 7:** Revisión y preparación semana 2 (pendiente)

---

## Objetivo del proyecto

Dominar 7 soluciones de IA en 4-6 meses para implementar en BDO Costa Rica:

1. **IA generativa para contenido/marketing** - Solución #1 más demandada
2. **Chatbots inteligentes** - ROI más rápido y medible
3. **Sistemas RAG** (Retrieval-Augmented Generation) - Core de knowledge management
4. **Procesamiento inteligente de documentos** - Azure Document Intelligence
5. **Asistentes virtuales internos** - Aplicable a todas las industrias
6. **Forecasting y analítica predictiva** - AutoML + Azure ML
7. **Agentes de IA autónomos** - Estado del arte 2025

**Timeline:** 6 meses total  
**Certificación objetivo:** AI-102 (Azure AI Engineer Associate)

---

## Estructura del proyecto
```
azure-ai-learning/
├── README.md                          # Este archivo
├── semana-01/                         # Semana 1 - Fundamentos
│   ├── venv/                          # Ambiente virtual (no en Git)
│   ├── test_ollama.py                 # Script día 1
│   ├── Fundamentos_de_LLMs.ipynb      # Notebook Colab día 2
│   └── Comparacion_de_modelos.ipynb   # Notebook comparación día 2
├── documentacion/                     # Documentación detallada
│   ├── Dia_1_completado.md           # Día 1 documentado
│   ├── Dia_2_completado.md           # Día 2 documentado
│   └── Dia_3_completado.md           # Día 3 documentado
└── recursos/                          # Guías y referencias
    └── guia-referencia.md             # Comandos esenciales
```

---

## Herramientas configuradas

### Ambiente local
- ✅ Python 3.14.0
- ✅ VS Code con extensiones (Python, Jupyter, Azure)
- ✅ Git con SSH authentication
- ✅ Ollama con llama3.2:3b

### Herramientas cloud (gratuitas)
- ✅ Google Colab con GPU T4
- ✅ Hugging Face (acceso a 200,000+ modelos)
- ✅ Azure for Students ($100 créditos, 12 meses)
- ✅ ML Lab (alternativa browser-based a Azure ML)

### Cuentas estratégicas
- **Microsoft Learn:** jalfaro@bdo.com (progreso profesional consolidado)
- **Azure Portal:** jordy.alfarobrenes@ucr.ac.cr (créditos educativos)
- **Razón:** Separación de progreso educativo vs recursos cloud

### Frameworks y librerías
- ✅ transformers 4.45.0+
- ✅ torch 2.0+
- ✅ ollama (Python SDK)

---

## Conceptos dominados hasta ahora

### Día 1
- Ambiente de desarrollo Python profesional
- Control de versiones con Git/GitHub
- Ejecución de LLMs locales con Ollama
- Primeros pasos con modelos de lenguaje

### Día 2
- **Tokenización:** Conversión texto ↔ números
- **Embeddings:** Representación vectorial de significado
- **Attention mechanism:** Corazón de transformers
- **Límites de contexto:** Chunking strategies para RAG
- **Gestión de memoria GPU:** Troubleshooting práctico
- **Comparación de modelos:** Evaluación de calidad

### Día 3
- **Azure Portal:** Navegación y gestión de recursos
- **Cost Management:** Presupuestos, alertas, monitoreo
- **Azure ML workflow:** Problema → datos → entrenamiento → evaluación → deployment
- **Automated ML:** Validación rápida de viabilidad de proyectos ML
- **Tipos de ML:** Regression, classification, clustering detalladamente
- **Troubleshooting cloud:** Resolución de errores de permisos
- **Pragmatismo técnico:** Cuándo usar workarounds vs resolver problemas

---

## Aprendizajes clave para BDO

### Selección de modelos
- ❌ GPT-2 (2019): Solo para educación, NO para clientes
- ✅ Mistral-7B (2024): Excelente para prototipos y demos
- ✅ Phi-3/4 (2024): Estado del arte para casos específicos
- ✅ Azure OpenAI GPT-4: Para producción con clientes

### Gestión de costos Azure
- Configurar presupuestos ANTES de empezar a usar servicios
- Alertas múltiples (50%, 70%, 90%, 100%) para control granular
- Usar tiers gratuitos (F0) siempre que sea posible
- Monitorear Cost Management semanalmente
- Detener/eliminar recursos inmediatamente después de usar
- **Lección aprendida:** Alertas solo notifican, NO detienen automáticamente

### Automated Machine Learning para clientes
- **Velocidad:** Validar viabilidad ML en horas vs semanas
- **Costo:** Prototipos rápidos sin contratar data scientist full-time
- **Casos de uso:** Demand forecasting, sales prediction, churn prediction
- **Pricing clientes:** Setup $50K-$100K, maintenance $5K-$15K/mes
- **ROI típico:** 12-18 meses
- **Limitación:** No reemplaza data scientist en proyectos complejos

### Problemas reales enfrentados
1. **Compatibilidad de versiones** (Phi-3-mini día 2)
   - Solución: Tener siempre plan B, probar alternativas
2. **Out of Memory GPU** (TinyLlama día 2)
   - Solución: Gestión explícita de memoria con `torch.cuda.empty_cache()`
3. **Calidad variable en español** (GPT-2 día 2)
   - Solución: Usar modelos multilingües modernos
4. **Deployment fallido Azure for Students** (día 3)
   - Solución: Usar ML Lab como alternativa pragmática

---

## Estrategia de costos

| Fase | Herramientas | Costo mensual |
|------|--------------|---------------|
| **Aprendizaje (actual)** | Ollama + Colab + Hugging Face + Azure F0 | $0-5 |
| **Prototipos (mes 2-3)** | Azure Free Tier + Colab + ML Lab | $5-30 |
| **Desarrollo (mes 4-6)** | Azure servicios selectivos | $30-100 |
| **Producción (post-6 meses)** | Azure OpenAI + AI Search | Variable según uso |

**Meta:** Mantener costos bajo $50/mes durante aprendizaje completo  
**Estado actual:** $0.31 en 3 días = ~$3/mes promedio (muy por debajo de meta) ✅

---

## Próximos hitos

### Semana 1 (días 4-7)
- [ ] Día 4: Azure AI Services fundamentals
- [ ] Día 5: Prompt engineering avanzado
- [ ] Día 6: APIs REST y preparación Azure OpenAI
- [ ] Día 7: Revisión semana 1 y solicitud acceso Azure OpenAI

### Semana 2 (días 8-14)
- [ ] Azure OpenAI Service access aprobado
- [ ] Primer chatbot con GPT-4
- [ ] Sistema RAG básico implementado
- [ ] Generador de contenido funcional

### Mes 2
- [ ] Certificación AI-102 obtenida
- [ ] 3-4 proyectos portfolio completos
- [ ] Primer demo para cliente interno BDO

### Mes 6
- [ ] 7 soluciones dominadas
- [ ] Portfolio con 10+ proyectos
- [ ] Capacidad de implementar para clientes reales

---

## Recursos utilizados

### Documentación oficial
- [Microsoft Learn](https://learn.microsoft.com/training/azure/) - Paths gratuitos
- [Hugging Face Docs](https://huggingface.co/docs/transformers/) - Transformers
- [Azure AI Docs](https://learn.microsoft.com/azure/ai-services/) - AI Services
- [Azure ML Docs](https://learn.microsoft.com/azure/machine-learning/) - ML Services

### Herramientas de aprendizaje
- [ML Lab](https://aka.ms/ml-lab) - Azure ML en navegador (gratis)
- [Azure Portal](https://portal.azure.com) - Gestión de recursos
- [DeepLearning.AI](https://www.deeplearning.ai/) - Cursos gratuitos
- Udemy Pro (acceso completo vía BDO)

### Comunidades
- [Hugging Face Discord](https://discord.gg/huggingface)
- [Azure AI Discord](https://discord.gg/yrTeVQwpWm)
- r/MachineLearning, r/LocalLLaMA, r/learnmachinelearning

---

## Métricas de progreso

| Métrica | Objetivo 6 meses | Actual | % Completado |
|---------|------------------|--------|--------------|
| Días completados | 180 | 3 | 1.7% |
| Horas invertidas | 360-540 | ~12 | 2.5% |
| Costo acumulado | <$300 | $0.31 | 0.1% ✅ |
| Soluciones dominadas | 7 | 0 | En progreso |
| Proyectos portfolio | 10+ | 0 | Fundamentos |
| Certificaciones | AI-102 | 0 | Preparación |
| Microsoft Learn XP | - | 2300 | Nivel 7 |

**Progreso:** 1.7% del timeline (ligeramente adelante del plan)  
**Ritmo actual:** Sostenible y eficiente

---

## Cómo usar este repositorio

### Para seguir mi progreso
1. Revisa [documentacion/](./documentacion/) para días completados
2. Notebooks en [semana-XX/](./semana-01/) para código ejecutable
3. README.md (este archivo) para overview general

### Para replicar mi aprendizaje
1. Comienza con [Dia_1_completado.md](./documentacion/Dia_1_completado.md)
2. Sigue cada día secuencialmente
3. Usa el código en los notebooks como base
4. Adapta según tus necesidades específicas

### Para contribuir
- Issues: Reportar errores en documentación
- Pull requests: Mejoras y correcciones
- Discussions: Compartir experiencias similares

---

## Contacto y referencias

**Proyecto:** Azure AI Learning Journey  
**Propósito:** Desarrollo de capacidades de IA para BDO Costa Rica  
**Timeline:** Noviembre 2025 - Abril 2026  
**GitHub:** [JordyAB00/azure-ai-learning](https://github.com/JordyAB00/azure-ai-learning)

**Inspirado por:**
- Plan de capacitación BDO 4-6 meses
- Demanda del mercado latinoamericano de IA
- Estrategia nacional de IA Costa Rica (ENIA 2024-2027)

---

**Última actualización:** 20 de noviembre de 2025, 22:00  
**Estado del proyecto:** 🟢 En progreso activo  
**Próxima sesión:** Día 4 - Azure AI Services fundamentals  
**Créditos Azure disponibles:** $99.69 de $100.00