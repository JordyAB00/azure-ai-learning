# Azure AI Learning Journey

Documentación de mi aprendizaje en Azure AI Services y desarrollo de soluciones de IA para BDO Costa Rica.

## Progreso general

**Días completados:** 2/7 (Semana 1)  
**Tiempo invertido:** ~9 horas  
**Costo acumulado:** $0  
**Última actualización:** 20 de noviembre de 2025

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

### 📋 Próximos días

**Día 3:** Azure for Students y fundamentos (pendiente)
- Activación de créditos Azure
- Configuración de presupuestos y alertas
- Microsoft Learn: Introduction to AI in Azure

**Día 4:** Azure AI Services (pendiente)
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
│   └── Dia_2_completado.md           # Día 2 documentado
└── recursos/                          # Guías y referencias
    └── cheatsheet.md                  # Comandos esenciales
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
- 🔄 Azure for Students (pendiente día 3)

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

---

## Aprendizajes clave para BDO

### Selección de modelos
- ❌ GPT-2 (2019): Solo para educación, NO para clientes
- ✅ Mistral-7B (2024): Excelente para prototipos y demos
- ✅ Phi-3/4 (2024): Estado del arte para casos específicos
- ✅ Azure OpenAI GPT-4: Para producción con clientes

### Gestión de costos
- Tokenización eficiente (español ~30% más tokens en modelos antiguos)
- Selección apropiada de modelo según caso de uso
- Implementación de caching cuando sea posible
- Monitoreo continuo con Azure Cost Management

### Problemas reales enfrentados
1. **Compatibilidad de versiones** (Phi-3-mini)
   - Solución: Tener siempre plan B, probar alternativas
2. **Out of Memory GPU** (TinyLlama)
   - Solución: Gestión explícita de memoria con `torch.cuda.empty_cache()`
3. **Calidad variable en español** (GPT-2)
   - Solución: Usar modelos multilingües modernos

---

## Estrategia de costos

| Fase | Herramientas | Costo mensual |
|------|--------------|---------------|
| **Aprendizaje (actual)** | Ollama + Colab + Hugging Face | $0 |
| **Prototipos (mes 2-3)** | Azure Free Tier + Colab | $0-30 |
| **Desarrollo (mes 4-6)** | Azure servicios selectivos | $30-100 |
| **Producción (post-6 meses)** | Azure OpenAI + AI Search | Variable según uso |

**Meta:** Mantener costos bajo $50/mes durante aprendizaje completo

---

## Próximos hitos

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

### Cursos
- DeepLearning.AI (gratuitos durante beta)
- Udemy Pro (acceso completo)
- Microsoft Learn Paths (gratuitos)

### Comunidades
- [Hugging Face Discord](https://discord.gg/huggingface)
- [Azure AI Discord](https://discord.gg/yrTeVQwpWm)
- r/MachineLearning, r/LocalLLaMA

---

## Métricas de progreso

| Métrica | Objetivo 6 meses | Actual |
|---------|------------------|--------|
| Días completados | 180 | 2 |
| Horas invertidas | 360-540 | ~9 |
| Costo acumulado | <$300 | $0 |
| Soluciones dominadas | 7 | 0 (en progreso) |
| Proyectos portfolio | 10+ | 0 |
| Certificaciones | AI-102 | 0 |

**Progreso:** 1.1% del timeline (adelante del plan)

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

**Última actualización:** 20 de noviembre de 2025  
**Estado del proyecto:** 🟢 En progreso activo  
**Próxima sesión:** Día 3 - Azure for Students