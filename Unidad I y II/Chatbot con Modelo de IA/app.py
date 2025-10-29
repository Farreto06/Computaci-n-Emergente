from flask import Flask, request, jsonify, render_template
import os
import openai

app = Flask(__name__)


# Mensajes predefinidos: clave -> respuesta
PREDEFINED_MESSAGES = {
    'hola': 'Hola soy un chatbot',
    'adios': 'Adiós, que tengas buen día',
    'como estas': 'Estoy bien, gracias por preguntar',
    'gracias': 'De nada, para eso estoy',
    'ayuda': 'Puedo responder saludos y agradecer. Prueba: hola, adios, gracias'
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/messages', methods=['GET'])
def messages():
    # Devuelve las claves y descripciones breves
    return jsonify({'messages': list(PREDEFINED_MESSAGES.keys())})


def find_predefined_response(message: str):
    """Busca una respuesta en PREDEFINED_MESSAGES por coincidencia de frase o palabra.

    Lógica simple: busca la frase completa primero, luego busca si alguna clave está
    contenida en el mensaje.
    """
    if not isinstance(message, str):
        return None
    txt = message.lower()
    # coincidencia exacta de frase
    for k in PREDEFINED_MESSAGES:
        if txt.strip() == k:
            return PREDEFINED_MESSAGES[k]
    # coincidencia por presencia
    for k in PREDEFINED_MESSAGES:
        if k in txt:
            return PREDEFINED_MESSAGES[k]
    return None


def call_openai_model(message: str) -> str | None:
    """Call an external AI model (OpenAI) and return a short text reply.

    Uses the OPENAI_API_KEY environment variable. Returns None on error or
    when the key is not set so callers can fallback to predefined replies.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        app.logger.debug('OPENAI_API_KEY not set; skipping AI call')
        return None

    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            messages=[{'role': 'user', 'content': message}],
            max_tokens=150,
            temperature=0.6,
            n=1,
            timeout=10,
        )
        # Extract assistant text
        choices = resp.get('choices') or []
        if not choices:
            return None
        content = choices[0].get('message', {}).get('content')
        return content.strip() if isinstance(content, str) else None
    except Exception as e:
        # Don't expose exception detail to user; log for debugging and fallback
        app.logger.exception('OpenAI request failed')
        return None


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get('message', '') if isinstance(data, dict) else ''

    # Try AI model first
    ai_reply = call_openai_model(message)
    if ai_reply:
        return jsonify({'reply': ai_reply})

    # Fallback to predefined responses
    response = find_predefined_response(message) or "No entiendo. Pide 'ayuda' para ver comandos."
    return jsonify({'reply': response})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
