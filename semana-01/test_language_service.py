import os
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Cargar variables de entorno desde .env
load_dotenv()

# Obtener credenciales
key = os.getenv("LANGUAGE_KEY")
endpoint = os.getenv("LANGUAGE_ENDPOINT")

# Verificar que tenemos las credenciales
if not key or not endpoint:
    print("❌ Error: Falta configurar LANGUAGE_KEY o LANGUAGE_ENDPOINT en .env")
    exit(1)

print("✅ Credenciales cargadas correctamente\n")
print(f"Endpoint: {endpoint}")
print(f"Key: {key[:10]}... (oculta por seguridad)\n")

# Crear cliente de Language Service
credential = AzureKeyCredential(key)
client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

print("="*60)
print("PRUEBA 1: Análisis de sentimiento")
print("="*60 + "\n")

# Textos de ejemplo sobre BDO
textos = [
    "El servicio de BDO superó nuestras expectativas, excelente trabajo",
    "Muy insatisfechos con la demora en el proyecto, pésima comunicación",
    "El equipo es profesional pero el precio es bastante alto"
]

# Analizar sentimientos
resultados = client.analyze_sentiment(documents=textos, language="es")

for i, resultado in enumerate(resultados):
    print(f"Texto {i+1}: {textos[i]}")
    print(f"Sentimiento: {resultado.sentiment.upper()}")
    print(f"Confianza:")
    print(f"  - Positivo: {resultado.confidence_scores.positive:.2%}")
    print(f"  - Neutral:  {resultado.confidence_scores.neutral:.2%}")
    print(f"  - Negativo: {resultado.confidence_scores.negative:.2%}")
    print()

print("="*60)
print("PRUEBA 2: Extracción de frases clave")
print("="*60 + "\n")

# Texto largo sobre BDO
texto_largo = """
BDO Costa Rica es una firma líder en servicios de auditoría, consultoría 
y asesoría tributaria. Con años de experiencia en el mercado costarricense, 
ofrece soluciones innovadoras que ayudan a las empresas a crecer de manera 
sostenible y cumplir con todas las regulaciones locales. El equipo está 
compuesto por profesionales altamente capacitados en diferentes áreas como 
contabilidad, impuestos, tecnología y transformación digital.
"""

# Extraer frases clave
resultado_frases = client.extract_key_phrases(documents=[texto_largo], language="es")[0]

print(f"Texto analizado: {texto_largo[:100]}...\n")
print("Frases clave extraídas:")
for frase in resultado_frases.key_phrases:
    print(f"  • {frase}")

print("\n" + "="*60)
print("PRUEBA 3: Reconocimiento de entidades nombradas")
print("="*60 + "\n")

texto_entidades = "Jordy Alfaro trabaja en BDO Costa Rica ubicada en San José"

resultado_entidades = client.recognize_entities(documents=[texto_entidades], language="es")[0]

print(f"Texto: {texto_entidades}\n")
print("Entidades reconocidas:")
for entidad in resultado_entidades.entities:
    print(f"  • {entidad.text} → Tipo: {entidad.category} (confianza: {entidad.confidence_score:.2%})")

print("\n" + "="*60)
print("✅ Language Service funcionando correctamente!")
print("="*60)