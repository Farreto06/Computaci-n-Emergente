# Chatbot Flask simple

Proyecto mínimo con Flask que expone una página web con un chat y un endpoint `/chat`.

Comportamiento programado: si el mensaje contiene la palabra `hola` (insensible a mayúsculas) la respuesta será:

```
Hola soy un chatbot
```

Cómo usar:

1. Crear un virtualenv y activarlo (opcional pero recomendado)

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instalar dependencias

```bash
pip install -r requirements.txt
```

3. Ejecutar la app

```bash
python app.py
```

4. Abrir http://127.0.0.1:5000/ y escribir "hola" en el chat

Prueba rápida desde la terminal (requiere `requests` si usas `test_chat.py`):

```bash
python test_chat.py
```

Endpoint adicional:

- `GET /messages` — devuelve la lista de comandos/mensajes disponibles que el bot reconoce.

Mensajes predefinidos soportados (ejemplos): `hola`, `adios`, `como estas`, `gracias`, `ayuda`
