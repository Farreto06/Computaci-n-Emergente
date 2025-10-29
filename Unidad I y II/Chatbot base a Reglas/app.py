from flask import Flask, request, jsonify, render_template

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


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get('message', '') if isinstance(data, dict) else ''
    response = find_predefined_response(message) or "No entiendo. Pide 'ayuda' para ver comandos."
    return jsonify({'reply': response})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
