# Día 1: Configuración del ambiente de desarrollo

**Fecha:** 19 de noviembre de 2025  
**Duración:** 3-4 horas  
**Costo:** $0

## Resumen ejecutivo

En este primer día establecimos las bases técnicas del proyecto de aprendizaje de Azure AI. Configuramos todas las herramientas necesarias para trabajar con modelos de lenguaje localmente, sin gastar dinero, y establecimos un flujo de trabajo profesional con control de versiones.

## Objetivos cumplidos

- Instalación y configuración de Python 3.14.0
- Configuración de VS Code con extensiones para desarrollo de IA
- Instalación y configuración de Git con autenticación SSH
- Creación del repositorio en GitHub para documentar el aprendizaje
- Instalación de Ollama para ejecutar LLMs localmente
- Descarga del modelo llama3.2:3b (2 GB)
- Primer script en Python interactuando con un modelo de lenguaje
- Creación de ambiente virtual Python para aislar dependencias

---

## Parte 1: Instalación de Python 3.14.0

### Pasos realizados

1. Descarga desde https://www.python.org/downloads/
2. Instalación con opción "Add Python to PATH" marcada
3. Verificación con comando `python --version`

### Resultado

```
Python 3.14.0
```

### ¿Por qué Python 3.14.0?

- Versión más reciente y estable
- Compatible con todas las librerías de IA modernas
- Mejoras de rendimiento sobre versiones anteriores
- Soporte a largo plazo de la comunidad

---

## Parte 2: Configuración de VS Code

### Extensiones instaladas

1. **Python** (microsoft.python)
   - Autocompletado inteligente
   - Debugging integrado
   - Linting y formateo de código
   - Soporte para Jupyter notebooks

2. **Jupyter** (ms-toolsai.jupyter)
   - Ejecutar notebooks directamente en VS Code
   - Visualización de resultados inline
   - Conexión con kernels de Python

3. **Azure Account** (ms-vscode.azure-account)
   - Autenticación con Azure desde VS Code
   - Gestión de recursos de Azure
   - Deploy de aplicaciones

4. **Azure Resources** (ms-azuretools.vscode-azureresourcegroups)
   - Visualizar recursos de Azure
   - Crear y gestionar servicios
   - Monitorear costos

### Cómo instalar extensiones

1. Presionar `Ctrl+Shift+X`
2. Buscar el nombre de la extensión
3. Clic en "Install"
4. Reiniciar VS Code si es necesario

---

## Parte 3: Git y GitHub

### 3.1: Instalación de Git

**Descarga:** https://git-scm.com/downloads  
**Opciones:** Todas las predeterminadas están bien

### 3.2: Configuración inicial de Git

```bash
# Configurar nombre (aparecerá en commits)
git config --global user.name "Tu Nombre Completo"

# Configurar email (debe coincidir con GitHub)
git config --global user.email "tu_email@ejemplo.com"

# Verificar configuración
git config --global --list
```

### 3.3: Generación de clave SSH

La autenticación SSH es más segura y conveniente que usar contraseñas.

```bash
# Generar clave SSH
ssh-keygen -t ed25519 -C "tu_email@ejemplo.com"

# Presionar Enter 3 veces (ubicación por defecto, sin passphrase)
```

**Resultado:** Se crean dos archivos en `C:\Users\TuUsuario\.ssh\`:
- `id_ed25519` (clave privada, NUNCA compartir)
- `id_ed25519.pub` (clave pública, esta se sube a GitHub)

### 3.4: Configuración del agente SSH en Windows

```powershell
# Iniciar servicio SSH (PowerShell como Administrador)
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent

# Agregar clave al agente
ssh-add $env:USERPROFILE\.ssh\id_ed25519
```

**Salida esperada:**
```
Identity added: C:\Users\JordyAlfaroBrebes\.ssh\id_ed25519 (jordyab00@gmail.com)
```

### 3.5: Agregar clave SSH a GitHub

```powershell
# Copiar clave pública al portapapeles
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | clip
```

**En GitHub:**
1. Settings → SSH and GPG keys
2. New SSH key
3. Title: "PC-BDO-Windows"
4. Key: Pegar la clave
5. Add SSH key

### 3.6: Verificar conexión

```bash
ssh -T git@github.com
```

**Salida esperada:**
```
Hi JordyAB00! You've successfully authenticated, but GitHub does not provide shell access.
```

Este mensaje es correcto. Significa que la autenticación funciona.

---

## Parte 4: Creación del repositorio del proyecto

### 4.1: Crear repositorio en GitHub

1. GitHub → New repository
2. Repository name: `azure-ai-learning`
3. Description: "Documentación de mi aprendizaje en Azure AI Services"
4. Public
5. NO marcar "Add README file"
6. Create repository

### 4.2: Crear proyecto local

```bash
# Navegar a Documents
cd C:\Users\JordyAlfaroBrebes\Documents

# Crear carpeta del proyecto
mkdir azure-ai-learning
cd azure-ai-learning

# Inicializar Git
git init
```

### 4.3: Crear archivos iniciales

**README.md:**
```markdown
# Azure AI Learning Journey

Documentación de mi aprendizaje en Azure AI Services y desarrollo de soluciones de IA para BDO Costa Rica.

## Semana 1: Fundamentos y configuración

### Completado
- Python 3.14.0 instalado
- VS Code configurado con extensiones de Azure y Python
- Git y GitHub configurados correctamente
- SSH authentication funcionando
- Ollama instalado con modelo llama3.2:3b
- Primer script en Python interactuando con LLM local
- Ambiente virtual creado para semana 1

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

**.gitignore:**
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
```

### 4.4: Primer commit y push

```bash
# Agregar archivos
git add .

# Ver qué se agregará
git status

# Hacer commit
git commit -m "Primer commit: Setup inicial del proyecto Azure AI Learning"

# Cambiar rama a main
git branch -M main

# Conectar con GitHub
git remote add origin git@github.com:JordyAB00/azure-ai-learning.git

# Subir código
git push -u origin main
```

---

## Parte 5: Instalación y uso de Ollama

### 5.1: ¿Qué es Ollama?

Ollama es una herramienta que permite ejecutar modelos de lenguaje grandes (LLMs) localmente en tu computadora, similar a Docker pero para modelos de IA.

**Ventajas:**
- Costo cero después de la instalación
- Sin necesidad de internet una vez descargado el modelo
- Privacidad total (datos no salen de tu máquina)
- Experimenta sin límites de API calls
- Perfecto para aprendizaje y prototipos

### 5.2: Instalación

1. Descargar desde https://ollama.com/download
2. Ejecutar `OllamaSetup.exe`
3. Esperar instalación automática (~2 minutos)
4. Verificar: `ollama --version`

### 5.3: Descargar modelo llama3.2:3b

```bash
ollama pull llama3.2:3b
```

**Características del modelo:**
- Tamaño: ~2 GB
- Parámetros: 3 mil millones
- Desarrollado por Meta
- Optimizado para instrucciones
- Bueno para aprendizaje y prototipos
- Responde en español e inglés

**Tiempo de descarga:** 10-15 minutos (dependiendo de internet)

### 5.4: Uso interactivo

```bash
# Iniciar chat con el modelo
ollama run llama3.2:3b

# Escribir prompts y recibir respuestas
>>> Hola, explícame qué es un modelo de lenguaje

# Salir
>>> /bye
```

### 5.5: Otros modelos disponibles

```bash
# Ver modelos descargados
ollama list

# Modelos populares para descargar después
ollama pull llama3.2:1b    # Más pequeño, más rápido
ollama pull mistral:7b     # Más capaz, más lento
ollama pull codellama      # Especializado en código
```

---

## Parte 6: Primer script en Python con Ollama

### 6.1: Crear estructura de carpetas

```bash
cd C:\Users\JordyAlfaroBrebes\Documents\azure-ai-learning
mkdir semana-01
cd semana-01
```

### 6.2: Crear ambiente virtual

Los ambientes virtuales aíslan las dependencias de cada proyecto.

```bash
# Crear ambiente virtual
python -m venv venv

# Activar (PowerShell)
.\venv\Scripts\Activate.ps1

# Activar (CMD)
venv\Scripts\activate
```

**Si PowerShell da error de permisos:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Cuando está activado, verás:**
```
(venv) PS C:\...\semana-01>
```

### 6.3: Instalar librería de Ollama

```bash
pip install ollama
```

### 6.4: Script completo: test_ollama.py

```python
import ollama

# Primera prueba: respuesta simple
print("=== Prueba 1: Pregunta simple ===")
response = ollama.chat(
    model='llama3.2:3b',
    messages=[
        {
            'role': 'user',
            'content': '¿Qué es Azure OpenAI Service en 2 oraciones?',
        },
    ]
)

print(response['message']['content'])
print("\n" + "="*50 + "\n")

# Segunda prueba: con contexto de sistema
print("=== Prueba 2: Con rol de sistema ===")
response = ollama.chat(
    model='llama3.2:3b',
    messages=[
        {
            'role': 'system',
            'content': 'Eres un experto en inteligencia artificial que explica conceptos de forma clara y concisa.'
        },
        {
            'role': 'user',
            'content': 'Explícame qué es un chatbot RAG en un párrafo.',
        },
    ]
)

print(response['message']['content'])
print("\n" + "="*50 + "\n")

# Tercera prueba: conversación multi-turno
print("=== Prueba 3: Conversación multi-turno ===")
messages = [
    {
        'role': 'user',
        'content': '¿Cuál es la capital de Costa Rica?',
    }
]

response = ollama.chat(model='llama3.2:3b', messages=messages)
print("Usuario: ¿Cuál es la capital de Costa Rica?")
print(f"Asistente: {response['message']['content']}\n")

# Agregar la respuesta al historial y hacer otra pregunta
messages.append(response['message'])
messages.append({
    'role': 'user',
    'content': '¿Y cuál es su población aproximada?',
})

response = ollama.chat(model='llama3.2:3b', messages=messages)
print("Usuario: ¿Y cuál es su población aproximada?")
print(f"Asistente: {response['message']['content']}")
```

### 6.5: Ejecutar el script

```bash
python test_ollama.py
```

### 6.6: Conceptos importantes del código

**1. Estructura de mensajes:**
```python
messages = [
    {'role': 'system', 'content': 'Instrucciones para el modelo'},
    {'role': 'user', 'content': 'Pregunta del usuario'},
    {'role': 'assistant', 'content': 'Respuesta del modelo'}
]
```

**2. Roles:**
- `system`: Instrucciones generales sobre cómo debe comportarse el modelo
- `user`: Mensajes del usuario
- `assistant`: Respuestas del modelo (se agregan para mantener contexto)

**3. Conversaciones multi-turno:**
Para que el modelo "recuerde" la conversación, debes mantener todo el historial de mensajes en el array.

---

## Parte 7: Documentar el progreso en Git

```bash
# Volver a la carpeta principal
cd C:\Users\JordyAlfaroBrebes\Documents\azure-ai-learning

# Ver cambios
git status

# Agregar todos los archivos
git add .

# Commit descriptivo
git commit -m "Día 1 completado: Ollama instalado y primer script con LLM local"

# Subir a GitHub
git push
```

---

## Guía de referencia: Comandos esenciales

### PowerShell: Navegación y gestión de archivos

```powershell
# Ver directorio actual
pwd

# Listar archivos y carpetas
ls
# o
dir

# Listar incluyendo archivos ocultos
Get-ChildItem -Force

# Cambiar de directorio
cd nombre-carpeta
cd ..                    # Subir un nivel
cd C:\ruta\completa      # Ir a ruta específica
cd ~                     # Ir a carpeta de usuario

# Crear carpeta
mkdir nombre-carpeta
# o
New-Item -ItemType Directory -Name nombre-carpeta

# Crear archivo
New-Item -ItemType File -Name archivo.txt
# o
echo "contenido" > archivo.txt

# Ver contenido de archivo
Get-Content archivo.txt
# o
cat archivo.txt

# Copiar contenido al portapapeles
Get-Content archivo.txt | clip

# Eliminar archivo
Remove-Item archivo.txt

# Eliminar carpeta y su contenido
Remove-Item carpeta -Recurse -Force

# Limpiar pantalla
cls
# o
clear
```

### Git: Comandos básicos

```bash
# Configuración inicial (solo una vez)
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"

# Ver configuración
git config --global --list

# Inicializar repositorio
git init

# Ver estado de archivos
git status

# Agregar archivos al staging
git add archivo.py           # Un archivo específico
git add .                    # Todos los archivos
git add *.py                 # Todos los archivos .py

# Hacer commit
git commit -m "Mensaje descriptivo"

# Ver historial de commits
git log
git log --oneline            # Versión compacta
git log --graph --oneline    # Con gráfico de ramas

# Ver cambios antes de commit
git diff

# Conectar con repositorio remoto
git remote add origin git@github.com:usuario/repo.git

# Ver repositorios remotos configurados
git remote -v

# Subir cambios
git push
git push -u origin main      # Primera vez (establece tracking)

# Descargar cambios
git pull

# Ver ramas
git branch

# Crear y cambiar a nueva rama
git checkout -b nueva-rama

# Cambiar de rama
git checkout nombre-rama

# Deshacer cambios en archivo (antes de commit)
git checkout -- archivo.py

# Ver un commit específico
git show abc1234
```

### Git: Comandos avanzados útiles

```bash
# Deshacer último commit (mantiene cambios)
git reset --soft HEAD~1

# Deshacer último commit (elimina cambios)
git reset --hard HEAD~1

# Ver quién modificó cada línea de un archivo
git blame archivo.py

# Crear etiqueta (tag) para versión
git tag -a v1.0 -m "Versión 1.0"
git push origin v1.0

# Guardar cambios temporalmente sin commit
git stash
git stash pop                # Recuperar cambios guardados

# Clonar repositorio
git clone git@github.com:usuario/repo.git

# Actualizar información de repositorio remoto
git fetch

# Buscar en commits
git log --grep="palabra"

# Ver archivos en un commit específico
git show abc1234:archivo.py
```

### Python: Ambientes virtuales

```bash
# Crear ambiente virtual
python -m venv nombre-env

# Activar ambiente virtual
# Windows PowerShell:
.\nombre-env\Scripts\Activate.ps1
# Windows CMD:
nombre-env\Scripts\activate
# Linux/Mac:
source nombre-env/bin/activate

# Desactivar ambiente virtual
deactivate

# Instalar paquete
pip install nombre-paquete

# Instalar versión específica
pip install nombre-paquete==1.2.3

# Instalar desde requirements.txt
pip install -r requirements.txt

# Ver paquetes instalados
pip list

# Crear requirements.txt
pip freeze > requirements.txt

# Actualizar pip
python -m pip install --upgrade pip

# Desinstalar paquete
pip uninstall nombre-paquete

# Buscar paquete
pip search nombre-paquete
```

### Ollama: Comandos útiles

```bash
# Ver versión
ollama --version

# Descargar modelo
ollama pull nombre-modelo

# Listar modelos descargados
ollama list

# Ejecutar modelo interactivamente
ollama run nombre-modelo

# Eliminar modelo
ollama rm nombre-modelo

# Ver información de un modelo
ollama show nombre-modelo

# Ejecutar modelo con parámetros específicos
ollama run nombre-modelo --temperature 0.5

# Comandos dentro del chat interactivo:
/bye                         # Salir
/clear                       # Limpiar pantalla
/help                        # Ver ayuda
```

### VS Code: Atajos de teclado esenciales

```
# General
Ctrl+Shift+P                 # Paleta de comandos
Ctrl+P                       # Buscar archivo
Ctrl+,                       # Configuración
Ctrl+`                       # Abrir/cerrar terminal
Ctrl+B                       # Mostrar/ocultar sidebar

# Edición
Ctrl+S                       # Guardar
Ctrl+Shift+S                 # Guardar como
Ctrl+X                       # Cortar línea
Ctrl+C                       # Copiar línea
Ctrl+V                       # Pegar
Ctrl+Z                       # Deshacer
Ctrl+Shift+Z                 # Rehacer
Ctrl+/                       # Comentar/descomentar
Ctrl+D                       # Seleccionar siguiente ocurrencia
Alt+↑/↓                      # Mover línea arriba/abajo
Shift+Alt+↑/↓                # Duplicar línea arriba/abajo
Ctrl+Shift+K                 # Eliminar línea

# Navegación
Ctrl+G                       # Ir a línea
Ctrl+F                       # Buscar
Ctrl+H                       # Reemplazar
Ctrl+Shift+F                 # Buscar en archivos
F12                          # Ir a definición
Alt+←/→                      # Navegar atrás/adelante

# Multi-cursor
Alt+Click                    # Agregar cursor
Ctrl+Alt+↑/↓                 # Agregar cursor arriba/abajo

# Terminal
Ctrl+Shift+`                 # Nueva terminal
Ctrl+Shift+5                 # Dividir terminal
```

### PowerShell: Atajos útiles

```
Tab                          # Autocompletar
Ctrl+C                       # Cancelar comando actual
Ctrl+L                       # Limpiar pantalla
↑/↓                          # Navegar historial de comandos
Ctrl+R                       # Buscar en historial
F7                           # Ver historial de comandos
Home/End                     # Ir al inicio/final de línea
Ctrl+←/→                     # Saltar palabra por palabra
```

---

## Trucos y mejores prácticas

### Git

**1. Commits frecuentes y descriptivos**
```bash
# Mal
git commit -m "cambios"

# Bien
git commit -m "Agregado script de prueba de Ollama con 3 casos de uso"
```

**2. Revisar antes de commit**
```bash
git status              # ¿Qué archivos cambié?
git diff                # ¿Qué cambios específicos hice?
git add .               # Agregar todo
git status              # Verificar de nuevo
git commit -m "..."     # Ahora sí commit
```

**3. .gitignore desde el inicio**
Siempre crea `.gitignore` ANTES del primer commit para evitar subir archivos innecesarios.

**4. Branches para experimentar**
```bash
git checkout -b experimento
# Hacer cambios sin miedo
# Si funciona: merge
# Si no funciona: eliminar branch
```

### Python

**1. Siempre usa ambientes virtuales**
```bash
# Nunca instales paquetes globalmente
pip install paquete          # MAL

# Siempre en ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install paquete          # BIEN
```

**2. Documenta dependencias**
```bash
pip freeze > requirements.txt    # Después de instalar paquetes
```

**3. Nomenclatura de variables**
```python
# Mal
x = ollama.chat(...)
a = "hola"

# Bien
response = ollama.chat(...)
user_message = "hola"
```

### VS Code

**1. Usa Multi-cursor para edición masiva**
- `Ctrl+D` para seleccionar próxima ocurrencia
- Edita todas a la vez

**2. Snippets personalizados**
File → Preferences → User Snippets → Python
```json
{
  "Ollama Chat": {
    "prefix": "ollchat",
    "body": [
      "response = ollama.chat(",
      "    model='llama3.2:3b',",
      "    messages=[",
      "        {'role': 'user', 'content': '$1'},",
      "    ]",
      ")",
      "print(response['message']['content'])"
    ]
  }
}
```

**3. Configuración recomendada**
Ctrl+, y buscar estas configuraciones:
- `Format On Save`: ✓ (formatea código al guardar)
- `Auto Save`: afterDelay
- `Trim Trailing Whitespace`: ✓

### Ollama

**1. Prueba diferentes temperaturas**
```python
# Temperature baja (0.1-0.3): Respuestas más deterministas
# Temperature media (0.5-0.7): Balance creatividad/consistencia
# Temperature alta (0.8-1.0): Respuestas más creativas/variadas
```

**2. System messages son poderosos**
```python
# Define personalidad y estilo desde el inicio
{'role': 'system', 'content': 'Eres un profesor paciente que explica conceptos técnicos con analogías simples.'}
```

**3. Context length limitado**
Llama3.2:3b tiene límite de ~2000 tokens. En conversaciones largas, el modelo "olvida" mensajes antiguos.

---

## Estructura final del proyecto después del Día 1

```
azure-ai-learning/
├── .git/                           # Control de versiones (oculto)
├── .gitignore                      # Archivos a ignorar
├── README.md                       # Documentación principal
└── semana-01/                      # Carpeta de la semana 1
    ├── venv/                       # Ambiente virtual (ignorado por git)
    └── test_ollama.py              # Primer script con Ollama
```

---

## Problemas comunes y soluciones

### Error: "python no se reconoce como comando"

**Causa:** Python no está en PATH

**Solución:**
1. Buscar "Environment Variables" en Windows
2. Editar PATH
3. Agregar: `C:\Users\TuUsuario\AppData\Local\Programs\Python\Python314`
4. Reiniciar terminal

### Error: "Permission denied" en Git push

**Causa:** Clave SSH no configurada o GitHub no la reconoce

**Solución:**
```bash
ssh -T git@github.com    # Verificar conexión
```
Si falla, revisar que la clave esté en GitHub Settings → SSH keys

### Error: "Activate.ps1 cannot be loaded" en PowerShell

**Causa:** Política de ejecución de PowerShell

**Solución:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Ollama no responde o respuesta muy lenta

**Causa:** Modelo muy grande para tu hardware o RAM insuficiente

**Solución:**
- Usar modelo más pequeño: `ollama pull llama3.2:1b`
- Cerrar programas innecesarios
- Esperar ~30 segundos en primera ejecución (modelo se carga en memoria)

### VS Code no reconoce ambiente virtual

**Causa:** Interprete de Python no seleccionado

**Solución:**
1. `Ctrl+Shift+P`
2. "Python: Select Interpreter"
3. Elegir el que tiene `venv` en la ruta

---

## Recursos para profundizar

### Git y GitHub
- Documentación oficial: https://git-scm.com/doc
- GitHub Guides: https://guides.github.com/
- Git CheatSheet: https://education.github.com/git-cheat-sheet-education.pdf

### Python
- Documentación oficial: https://docs.python.org/3/
- Real Python: https://realpython.com/
- Python Package Index: https://pypi.org/

### Ollama
- Documentación: https://ollama.com/docs
- Librería de modelos: https://ollama.com/library
- GitHub: https://github.com/ollama/ollama

### VS Code
- Documentación: https://code.visualstudio.com/docs
- Atajos de teclado: https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf
- Python en VS Code: https://code.visualstudio.com/docs/python/python-tutorial

---

## Métricas del Día 1

| Métrica | Valor |
|---------|-------|
| Tiempo invertido | 3-4 horas |
| Costo total | $0 |
| Herramientas instaladas | 4 (Python, VS Code, Git, Ollama) |
| Líneas de código escritas | ~60 |
| Modelo descargado | llama3.2:3b (2 GB) |
| Commits realizados | 2 |
| Archivos creados | 4 |

---

## Checklist de verificación

Antes de pasar al Día 2, verifica que cumples con todo:

- [ ] Python 3.14.0 instalado y funcionando (`python --version`)
- [ ] VS Code con 4 extensiones instaladas
- [ ] Git configurado con nombre y email
- [ ] SSH funcionando con GitHub (`ssh -T git@github.com`)
- [ ] Repositorio `azure-ai-learning` creado y en GitHub
- [ ] Ollama instalado (`ollama --version`)
- [ ] Modelo llama3.2:3b descargado (`ollama list`)
- [ ] Script `test_ollama.py` ejecutado exitosamente
- [ ] Ambiente virtual creado y funcional
- [ ] README.md actualizado con progreso
- [ ] Último push realizado a GitHub

---

## Próximos pasos (Día 2)

En el Día 2 trabajaremos con:
- Google Colab para acceso gratuito a GPUs
- Hugging Face para explorar modelos open source
- Primeros experimentos con transformers
- Análisis de sentimientos en español
- Generación de texto básica

**Preparación recomendada:**
- Tener cuenta de Google lista
- Conexión a internet estable (usaremos GPU en la nube)
- 3-4 horas disponibles

---

## Notas finales

Este día fue crucial para establecer las bases. Todo lo que venga después depende de tener estas herramientas correctamente configuradas. Tomarse el tiempo para hacerlo bien ahora ahorrará dolores de cabeza después.

La inversión de 3-4 horas en configuración inicial permite trabajar de forma profesional durante los próximos 4-6 meses sin interrupciones técnicas.

**Costo acumulado del proyecto:** $0  
**Próxima inversión:** $0 (Día 2 también es completamente gratuito)

---

**Documentación creada:** 19 de noviembre de 2025  
**Autor:** Jordy Alfaro Brebes  
**Proyecto:** Azure AI Learning Journey  
**GitHub:** https://github.com/JordyAB00/azure-ai-learning