# Azure AI Learning Journey

Documentación de mi aprendizaje en Azure AI Services y desarrollo de soluciones de IA para BDO Costa Rica.

## Semana 1: Fundamentos y configuración

### Completado
- Python 3.14.0 instalado
- VS Code configurado con extensiones de Azure y Python
- Git y GitHub configurados correctamente
- SSH authentication funcionando

### En progreso
- Instalación de Ollama
- Primer modelo de lenguaje local

### Próximos pasos
- Google Colab con GPU
- Cuenta de Hugging Face
- Azure for Students

## Objetivo del proyecto

Dominar 7 soluciones de IA en 4-6 meses:
1. IA generativa para contenido/marketing
2. Chatbots inteligentes
3. Sistemas RAG (Retrieval-Augmented Generation)
4. Procesamiento inteligente de documentos
5. Asistentes virtuales internos
6. Forecasting y analítica predictiva
7. Agentes de IA autónomos

## Costo objetivo

$0 durante aprendizaje inicial usando:
- Ollama (modelos locales)
- Google Colab (GPU gratuita)
- Azure Free Tier
- Microsoft Learn (gratis)
```

4. Guarda el archivo (Ctrl+S)

## Paso 4: Crear archivo .gitignore (3 minutos)

En VS Code, crea otro archivo nuevo:

1. Clic en **New File**
2. Guárdalo como `.gitignore` (con el punto al inicio)
3. Copia y pega este contenido:
```
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Ambiente virtual
env/
ENV/

# Credenciales y configuración
.env
.env.local
*.key
*.pem
config.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# Sistema operativo
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Azure
.azure/

# Logs
*.log