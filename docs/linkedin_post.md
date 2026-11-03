# LinkedIn draft (NOT posted)

Ultrasound QA knowledge is spread across SOPs, methods notes, and spreadsheets of probe test results. I built a small multi-agent RAG system that answers questions across all of them, and I measured how well it works.

What I built (Python, runs offline):
• Hybrid retrieval: FAISS dense vectors + TF-IDF lexical, fused with Reciprocal Rank Fusion
• Planner → Retriever / structured-data tool → Synthesizer → Critic, in a ReAct-style loop. The critic scores coverage and groundedness and can send the retriever back with an expanded query
• Every answer is cited, every agent step is traced with latency, and there's a JSON API ready to containerize for Cloud Run

Example: "What's the QA status of probe P-102 and why?" pulls the failing record from the table and the protocol rule that a >1 cm depth-of-penetration drop triggers service review, then cites both.

Results on a 20-question eval set I wrote:
• Recall@4 0.975, MRR 0.89, nDCG@4 0.92
• 90% of answers contained every required fact, 0.99 groundedness, p95 latency ~2 ms

Honest caveats: the corpus is small and synthetic (I wrote it from my phantom/ultrasound coursework and research interests). Both misses were synthesis-ranking errors, not retrieval errors, so next up is an LLM synthesizer + reranker, plus a paraphrased test set.

Code + eval + GCP mapping (Vertex AI Vector Search, Gemini, Cloud Run): https://github.com/michaellozadalongog/google-fde-biomed-rag-agent

#RAG #BiomedicalEngineering #Ultrasound #AIAgents #Python
