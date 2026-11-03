import sys
from .pipeline import build
from . import evaluate

def main():
    orch, _ = build()
    qs = sys.argv[1:] or ["What is the QA status of probe P-102 and why?",
                          "What disinfection level do endocavitary probes need?"]
    for q in qs:
        s, t = orch.answer(q)
        print(f"\nQ: {q}\nA: {s.answer}\nCitations: {s.citations}\nCritic: {s.critique} | steps={s.steps} | latency={t.total_ms()} ms | trace={t.trace_id}")
    evaluate.run()

main()
