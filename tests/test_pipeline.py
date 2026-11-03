from biomed_rag.pipeline import build
from biomed_rag import evaluate

def test_structured_answer():
    orch, _ = build()
    s, _ = orch.answer("What is the QA status of probe P-102 and why?")
    assert "fail" in s.answer and "probes#P-102" in s.citations

def test_eval_thresholds():
    out = evaluate.run(write=False, verbose=False)
    assert out["retrieval"]["hybrid"]["recall@4"] >= 0.8
    assert out["end_to_end"]["groundedness"] >= 0.8
