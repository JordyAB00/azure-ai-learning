# Guía de referencia rápida - Azure AI Learning

Comandos y atajos que usarás constantemente. Mantén esta guía abierta mientras trabajas.

---

## Git - Flujo de trabajo diario

```bash
# 1. Ver qué cambió
git status

# 2. Agregar cambios
git add .                    # Todos los archivos
git add archivo.py           # Archivo específico

# 3. Commit con mensaje
git commit -m "Descripción clara de cambios"

# 4. Subir a GitHub
git push

# Flujo completo en 4 comandos:
git status
git add .
git commit -m "Tu mensaje aquí"
git push
```

### Git - Comandos de rescate

```bash
# Ver historial
git log --oneline

# Deshacer cambios en archivo (antes de commit)
git checkout -- archivo.py

# Deshacer último commit (mantiene cambios)
git reset --soft HEAD~1

# Ver diferencias antes de commit
git diff
```

---

## PowerShell - Navegación básica

```powershell
# Ver dónde estoy
pwd

# Listar archivos
ls
dir

# Cambiar directorio
cd nombre-carpeta
cd ..                        # Subir un nivel
cd ~                         # Ir a home

# Crear carpeta
mkdir nombre-carpeta

# Crear archivo
New-Item archivo.txt

# Ver contenido
Get-Content archivo.txt
cat archivo.txt

# Copiar al portapapeles
Get-Content archivo.txt | clip

# Limpiar pantalla
cls
clear
```

---

## Python - Ambientes virtuales

```bash
# Crear ambiente virtual
python -m venv venv

# Activar (PowerShell)
.\venv\Scripts\Activate.ps1

# Activar (CMD)
venv\Scripts\activate

# Cuando activado, verás:
(venv) PS C:\...>

# Desactivar
deactivate

# Instalar paquete
pip install nombre-paquete

# Ver instalados
pip list

# Guardar dependencias
pip freeze > requirements.txt

# Instalar desde requirements.txt
pip install -r requirements.txt
```

---

## Ollama - Comandos esenciales

```bash
# Ver versión
ollama --version

# Descargar modelo
ollama pull llama3.2:3b

# Ver modelos instalados
ollama list

# Ejecutar interactivo
ollama run llama3.2:3b

# Eliminar modelo
ollama rm nombre-modelo

# Dentro del chat:
/bye                         # Salir
/clear                       # Limpiar pantalla
```

### Ollama - Código Python básico

```python
import ollama

# Respuesta simple
response = ollama.chat(
    model='llama3.2:3b',
    messages=[
        {'role': 'user', 'content': 'Tu pregunta aquí'}
    ]
)
print(response['message']['content'])

# Con system message
response = ollama.chat(
    model='llama3.2:3b',
    messages=[
        {'role': 'system', 'content': 'Eres un experto en...'},
        {'role': 'user', 'content': 'Tu pregunta'}
    ]
)

# Conversación multi-turno
messages = [
    {'role': 'user', 'content': 'Primera pregunta'}
]
response = ollama.chat(model='llama3.2:3b', messages=messages)
messages.append(response['message'])
messages.append({'role': 'user', 'content': 'Segunda pregunta'})
response = ollama.chat(model='llama3.2:3b', messages=messages)
```

---

## VS Code - Atajos imprescindibles

### General
```
Ctrl+Shift+P                 # Paleta de comandos (el más importante)
Ctrl+P                       # Buscar archivo
Ctrl+,                       # Abrir configuración
Ctrl+`                       # Abrir/cerrar terminal
Ctrl+B                       # Mostrar/ocultar sidebar
Ctrl+K Ctrl+S                # Ver todos los atajos
```

### Edición
```
Ctrl+S                       # Guardar
Ctrl+/                       # Comentar/descomentar línea
Ctrl+D                       # Seleccionar siguiente ocurrencia
Alt+↑/↓                      # Mover línea arriba/abajo
Shift+Alt+↑/↓                # Duplicar línea
Ctrl+Shift+K                 # Eliminar línea
Ctrl+X                       # Cortar línea (sin seleccionar)
Ctrl+Enter                   # Insertar línea abajo
Ctrl+Shift+Enter             # Insertar línea arriba
```

### Navegación
```
Ctrl+G                       # Ir a línea número
Ctrl+F                       # Buscar en archivo
Ctrl+H                       # Reemplazar
Ctrl+Shift+F                 # Buscar en todos los archivos
F12                          # Ir a definición
Alt+←                        # Volver atrás
Alt+→                        # Ir adelante
```

### Multi-cursor (muy útil)
```
Alt+Click                    # Agregar cursor
Ctrl+Alt+↑                   # Cursor línea arriba
Ctrl+Alt+↓                   # Cursor línea abajo
Ctrl+D                       # Seleccionar siguiente igual
Ctrl+Shift+L                 # Seleccionar todas las ocurrencias
```

### Terminal
```
Ctrl+Shift+`                 # Nueva terminal
Ctrl+Shift+5                 # Dividir terminal
Ctrl+PageUp/PageDown         # Cambiar entre terminales
```

---

## PowerShell - Atajos de teclado

```
Tab                          # Autocompletar comando/ruta
Ctrl+C                       # Cancelar comando
Ctrl+L                       # Limpiar pantalla
↑/↓                          # Historial de comandos
Ctrl+R                       # Buscar en historial
F7                           # Lista completa de historial
Home                         # Inicio de línea
End                          # Final de línea
Ctrl+←/→                     # Saltar palabra por palabra
Ctrl+A                       # Seleccionar todo en línea
```

---

## Patrones comunes de trabajo

### Iniciar nueva sesión de código

```bash
# 1. Abrir PowerShell
# 2. Ir a proyecto
cd C:\Users\JordyAlfaroBrebes\Documents\azure-ai-learning\semana-01

# 3. Activar ambiente virtual
.\venv\Scripts\Activate.ps1

# 4. Abrir VS Code
code .

# 5. Abrir terminal en VS Code (Ctrl+`)
# Ya estás listo para trabajar
```

### Terminar sesión de trabajo

```bash
# 1. En terminal de VS Code o PowerShell
# Ver cambios
git status

# 2. Agregar y commit
git add .
git commit -m "Descripción de lo que hice hoy"

# 3. Subir a GitHub
git push

# 4. Desactivar ambiente virtual
deactivate

# 5. Cerrar VS Code
```

### Crear nuevo script de prueba

```bash
# 1. En terminal (con ambiente virtual activado)
# 2. Crear archivo
New-Item test_nombre.py

# 3. Abrir en VS Code (si no está abierto)
code test_nombre.py

# 4. Escribir código
# 5. Ejecutar
python test_nombre.py

# 6. Si funciona, hacer commit
git add test_nombre.py
git commit -m "Agregado script de prueba para [funcionalidad]"
git push
```

---

## Troubleshooting rápido

### "python no se reconoce"
```bash
# Verificar instalación
where python

# Si no aparece, reiniciar terminal o agregar a PATH
```

### "Activate.ps1 cannot be loaded"
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Permission denied" en Git
```bash
# Verificar SSH
ssh -T git@github.com

# Debería decir: "Hi tu-usuario! You've successfully authenticated"
```

### Ollama no responde
```bash
# 1. Verificar que está corriendo (ícono en system tray)
# 2. Verificar modelo descargado
ollama list

# 3. Si falla, reiniciar Ollama
# Cerrar desde system tray y volver a abrir
```

### VS Code no reconoce ambiente virtual
```
1. Ctrl+Shift+P
2. Escribir: "Python: Select Interpreter"
3. Elegir el que tiene "venv" en la ruta
```

### Git dice "nothing to commit"
```bash
# Ver qué archivos cambiarion
git status

# Si no hay cambios pero sabes que modificaste algo:
# 1. Verifica que guardaste el archivo (Ctrl+S en VS Code)
# 2. Verifica que estás en el directorio correcto (pwd)
```

---

## Estructura de archivos recomendada

```
azure-ai-learning/
├── README.md                # Documentación principal
├── .gitignore               # Qué NO subir a GitHub
├── semana-01/               # Carpeta por semana
│   ├── venv/                # Ambiente virtual (NO en Git)
│   ├── test_ollama.py       # Scripts de prueba
│   └── notas.md             # Tus notas personales
├── semana-02/
│   ├── venv/
│   └── ...
└── recursos/                # Guías de referencia
    └── cheatsheet.md        # Este archivo
```

---

## Comandos para copiar/pegar

### Setup nuevo día de trabajo
```bash
cd C:\Users\JordyAlfaroBrebes\Documents\azure-ai-learning
mkdir semana-0X
cd semana-0X
python -m venv venv
.\venv\Scripts\Activate.ps1
code .
```

### Guardar progreso del día
```bash
git status
git add .
git commit -m "Día X: [Descripción de lo aprendido/hecho]"
git push
```

### Instalar paquetes comunes
```bash
pip install ollama jupyter pandas numpy matplotlib requests python-dotenv
```

---

## Variables de entorno (próximamente)

Cuando trabajes con APIs de Azure:

```bash
# Crear archivo .env
New-Item .env

# Contenido típico:
AZURE_OPENAI_ENDPOINT=https://tu-recurso.openai.azure.com/
AZURE_OPENAI_KEY=tu-clave-aqui
AZURE_OPENAI_DEPLOYMENT=nombre-deployment
```

```python
# Cargar en Python
from dotenv import load_dotenv
import os

load_dotenv()
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
key = os.getenv("AZURE_OPENAI_KEY")
```

**IMPORTANTE:** `.env` SIEMPRE debe estar en `.gitignore`

---

## Modelos Ollama recomendados

| Modelo | Tamaño | Uso recomendado |
|--------|--------|-----------------|
| llama3.2:1b | ~1 GB | Pruebas rápidas, hardware limitado |
| llama3.2:3b | ~2 GB | Balance ideal para aprendizaje |
| mistral:7b | ~4 GB | Más capaz, tareas complejas |
| codellama:7b | ~4 GB | Generación de código |
| phi3:3.8b | ~2.3 GB | Eficiente, bueno para español |

```bash
# Descargar cualquiera:
ollama pull nombre-modelo

# Ver todos disponibles:
# https://ollama.com/library
```

---

## Recursos de emergencia

### Si algo se rompe completamente

**Opción nuclear - Empezar de cero:**
```bash
# 1. Hacer backup de tu código
# Copiar carpeta azure-ai-learning a otro lado

# 2. Clonar repositorio de GitHub
cd C:\Users\JordyAlfaroBrebes\Documents
git clone git@github.com:JordyAB00/azure-ai-learning.git
cd azure-ai-learning

# 3. Recrear ambiente virtual
cd semana-01
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install ollama

# Ya tienes todo funcionando de nuevo
```

### Links útiles para copiar/pegar

- Ollama download: https://ollama.com/download
- Python download: https://www.python.org/downloads/
- VS Code download: https://code.visualstudio.com/
- Git download: https://git-scm.com/downloads
- Tu repositorio: https://github.com/JordyAB00/azure-ai-learning

---

## Checklist diario

Antes de cerrar tu sesión de trabajo:

- [ ] Guardar todos los archivos en VS Code (Ctrl+S)
- [ ] `git status` para ver cambios
- [ ] `git add .` para agregar cambios
- [ ] `git commit -m "..."` con mensaje descriptivo
- [ ] `git push` para subir a GitHub
- [ ] `deactivate` para salir del ambiente virtual
- [ ] Verificar en GitHub que los cambios se subieron

---

**Mantén esta guía abierta en una pestaña de VS Code o impresa junto a tu monitor.**

Última actualización: 19 de noviembre de 2025