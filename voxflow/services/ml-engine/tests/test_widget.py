from pathlib import Path
from starlette.testclient import TestClient
import sys

SERVICE_DIR = str(Path(__file__).resolve().parents[1])
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

from main import app

client = TestClient(app)


def test_widget_endpoint():
    response = client.get('/widget')
    assert response.status_code == 200
    html = response.text
    assert 'VoxFlow Widget' in html
    assert 'pill' in html


def test_memory_endpoint():
    response = client.get('/memory')
    assert response.status_code == 200
    data = response.json()
    assert 'transcript_count' in data
    assert 'learned_fillers' in data
    assert 'common_phrases' in data
