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