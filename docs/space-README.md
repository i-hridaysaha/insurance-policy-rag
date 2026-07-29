---
title: Insurance Policy Q&A (RAG)
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Insurance Policy Q&A (RAG)

Ask a health-insurance policy a plain-English question. Answers are grounded in the policy's own
clauses, with the exact clause cited, and every citation is verified against what the model was
actually shown. When the policy does not answer, the system refuses instead of guessing.

Generation on this hosted demo runs on a free OpenAI-compatible API. The retrieval-quality numbers
in the project README are measured on local models (qwen2.5:14b, llama3.1:8b); this demo serves a
different, free model, so treat its answer fluency as illustrative, not as the measured result.

Full method, design decisions, and evaluation: https://github.com/i-hridaysaha/insurance-policy-rag
