"""
test_gate.py — D1 gate client: fail-open contract, skip threshold, shapes.
No server is contacted; _post_systemone is stubbed.
"""
import src.gate as gate


def _stub(monkeypatch, payload):
    monkeypatch.setattr(gate, "_post_systemone", lambda p, t: payload)


def _answers(bank_probs, choice=None):
    a = {"probabilities": bank_probs}
    if choice:
        a["choice"] = choice
    return {"answers": {"bank": a}}


def test_fail_open_when_d1_down(monkeypatch):
    _stub(monkeypatch, None)
    d = gate.gate_chunk("Duckets decided to pause training today.")
    assert d.verdict == "store" and d.gated is False


def test_empty_chunk_skips_without_calling(monkeypatch):
    def boom(p, t):
        raise AssertionError("D1 must not be called for empty chunks")
    monkeypatch.setattr(gate, "_post_systemone", boom)
    d = gate.gate_chunk("   ")
    assert d.verdict == "skip" and d.gated is True


def test_confident_none_skips(monkeypatch):
    _stub(monkeypatch, _answers({"chat": 0.05, "working": 0.03,
                                 "code": 0.02, "none": 0.90}))
    d = gate.gate_chunk("lol ok thanks")
    assert d.verdict == "skip" and d.gated is True


def test_mushy_none_stores_anyway(monkeypatch):
    _stub(monkeypatch, _answers({"chat": 0.30, "working": 0.10,
                                 "code": 0.05, "none": 0.55}))
    d = gate.gate_chunk("borderline text")
    assert d.verdict == "store" and d.bank == "chat"


def test_bank_routing(monkeypatch):
    _stub(monkeypatch, _answers({"chat": 0.01, "working": 0.97,
                                 "code": 0.01, "none": 0.01}))
    d = gate.gate_chunk("The training run stopped at step 323.")
    assert d.verdict == "store" and d.bank == "working"
    assert abs(d.confidence - 0.97) < 1e-9


def test_unparseable_answers_fail_open(monkeypatch):
    _stub(monkeypatch, {"answers": {"bank": {"weird": 1}}})
    d = gate.gate_chunk("something worth keeping, probably")
    assert d.verdict == "store" and d.gated is False
