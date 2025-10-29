import requests


def test_messages_list():
    url = 'http://127.0.0.1:5000/messages'
    resp = requests.get(url)
    print('/messages', resp.status_code, resp.json())


def test_chat_many():
    url = 'http://127.0.0.1:5000/chat'
    for msg in ['hola', 'adios', 'como estas', 'gracias', 'manzana']:
        resp = requests.post(url, json={'message': msg})
        print('POST', msg, '->', resp.status_code, resp.json())


if __name__ == '__main__':
    print('This test requires the Flask app running on http://127.0.0.1:5000')
    test_messages_list()
    test_chat_many()
