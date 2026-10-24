import os

os.environ["DATABASE_URL"] = "sqlite:///./test.db"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, get_llm  # noqa: E402

seen = []


def fake_llm(history):
    seen.append(list(history))
    return f"echo: {history[-1]['content']}"


app.dependency_overrides[get_llm] = lambda: fake_llm
client = TestClient(app)


def test_new_session_and_memory():
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert r.json()["reply"] == "echo: hi"

    r2 = client.post("/chat", json={"message": "again", "session_id": sid})
    assert r2.json()["messages_in_context"] == 3  # user, assistant, user
    assert [m["content"] for m in seen[-1]] == ["hi", "echo: hi", "again"]


def test_history_persisted_and_delete():
    sid = client.post("/chat", json={"message": "one"}).json()["session_id"]
    msgs = client.get(f"/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_validation_and_unknown_session():
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "x", "session_id": "nope"}).status_code == 404


def test_window_limit(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_history_messages", 4)
    sid = client.post("/chat", json={"message": "m0"}).json()["session_id"]
    for i in range(1, 5):
        r = client.post("/chat", json={"message": f"m{i}", "session_id": sid})
    assert r.json()["messages_in_context"] == 4
