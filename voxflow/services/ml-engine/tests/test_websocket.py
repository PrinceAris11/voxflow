import importlib.util
from pathlib import Path

from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'main.py'
SPEC = importlib.util.spec_from_file_location('voxflow_main', MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

app = MODULE.app
client = TestClient(app)


def test_websocket_echo():
    with client.websocket_connect('/ws') as socket:
        socket.send_json({'type': 'ping'})
        assert socket.receive_json() == {'type': 'pong'}


def test_websocket_audio_ack():
    with client.websocket_connect('/ws') as socket:
        socket.send_json({'type': 'audio', 'data': 'base64sampledata'})
        resp = socket.receive_json()
        assert resp.get('type') == 'ack'
        assert resp.get('received') is True


def test_websocket_transcript_interim():
    with client.websocket_connect('/ws') as socket:
        socket.send_json({'type': 'transcript_interim', 'text': 'um basically testing'})
        resp = socket.receive_json()
        assert resp.get('type') == 'transcript_interim'
        assert 'testing' in resp.get('text', '')


def test_websocket_transcript_final():
    with client.websocket_connect('/ws') as socket:
        socket.send_json({
            'type': 'transcript',
            'text': 'So um basically we are launching tomorrow, right?',
            'use_llm': False,
            'enable_emoji': False,
        })
        resp = socket.receive_json()
        assert resp.get('type') == 'transcript'
        assert 'launching tomorrow' in resp.get('text', '')
        assert resp.get('detected_language') == 'en'

