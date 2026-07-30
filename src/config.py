"""Central configuration. Paths are resolved relative to the repo root."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- data ---
RAW_PDF = ROOT / "data" / "raw" / "optima_secure_prospectus.pdf"
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "processed" / "index"
FAISS_PATH = INDEX_DIR / "dense.faiss"
FAISS_IDS_PATH = INDEX_DIR / "dense_ids.json"
BM25_PATH = INDEX_DIR / "bm25.pkl"

EVAL_QA_PATH = ROOT / "eval" / "qa_pairs.yaml"

# --- document structure ---
#
# The document is cut into REGIONS and each is ingested on its own terms. Regions are not a
# nicety. The clause parser walks a page stream and cuts at headings, so a region boundary
# that is not also a hard stop lets the last block swallow everything after it: Section 17
# (the final heading, p.55) absorbed pages 58-68 -- the whole of Annexure A -- and produced 31
# chunks, 16% of the corpus, full of exactly the mangled table text this pipeline exists to
# avoid indexing. Filtering those out afterwards by page number does not work, because the
# block's page_start is 55 and so it never matches the filter.
#
# PROSE_PAGES        policy body: coverage, exclusions, claims, terms. The real content.
# ANNEXURE_A_PAGES   plan matrix. Excluded here and replaced by the hand-structured rows in
#                    src/ingest/annexure_a.py, because the text layer destroys its columns.
# ANNEXURE_B_PAGES   non-medical items list. Indexed: it is what Protect Benefit (4.3) pays for.
# RATE_CHART_PAGES   premium tables, age x sum-insured x plan x city-tier. Excluded.
# NONPAYABLE_PAGES   Lists II/III/IV of non-payable items. Indexed: real claim-time questions.
PROSE_PAGES = range(1, 59)  # 1-58
ANNEXURE_A_PAGES = range(59, 67)  # 59-66
ANNEXURE_B_PAGES = range(67, 69)  # 67-68
RATE_CHART_PAGES = range(69, 289)  # 69-288
NONPAYABLE_PAGES = range(289, 292)  # 289-291

# Premium computation lives inside the prose body but is the same class of content as the rate
# charts: worked numeric examples, not policy terms. Section 16 is "Calculation of premium for
# Family Floater Policy", Section 17 is "Premium Computation Illustration". Excluding them keeps
# the scope boundary honest and consistent -- we tell users premium lookup is out of scope, so we
# should not half-index it -- and it stops ~35 numeric chunks from crowding the BM25 candidate pool.
PREMIUM_SECTIONS = {"16", "17"}

# A line appearing on at least this fraction of pages is running header/footer furniture and is
# stripped. Detected from the data, not hardcoded, so the pipeline transfers to other insurers'
# documents.
#
# 0.15 is not a guess, it sits in a gap measured in this document. The registration footer is
# typeset with TWO different line wraps: one on 232 pages (79.7%) and another on 59 pages (20.3%).
# Real prose lines never recur above ~3%. Between 3% and 20% there is nothing but rate-chart table
# rows, which are excluded from the corpus anyway. So any threshold in that empty band removes all
# furniture and no content.
#
# The original 0.30 fell between the two footer layouts, stripping the common one and leaving the
# other -- and the 59 pages carrying the surviving layout were the prose pages. The footer leaked
# into clause text (Section 3.1.1 ended up with "HDFHLIA22141V032122 | my:Health Hospital Cash
# Benefit..." inside it) where it added junk tokens to BM25 and dragged the chunk's embedding
# toward every other contaminated chunk.
BOILERPLATE_PAGE_FRACTION = 0.15

# --- retrieval ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768

# all-mpnet-base-v2 has a hard 384-token input window. SentenceTransformer silently
# TRUNCATES anything longer -- no warning, no error, the tail simply never reaches the
# encoder. A chunk above this limit therefore looks perfectly correct in chunks.jsonl and
# in any citation, while being partly invisible to dense retrieval. Enforced in
# scripts/build_index.py, which refuses to build an index containing an over-window chunk.
EMBEDDING_MAX_TOKENS = 384

# Chunking. Insurance clauses are the semantic unit, so chunks follow clause boundaries
# rather than a fixed window. MAX_CHUNK_CHARS is set below the token window (legal prose with
# numbers and Rs amounts tokenises at roughly 3.5 chars/token, so 384 tokens is ~1,340 chars;
# 1,200 leaves headroom for the section header prepended to every chunk).
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 120

# Hybrid fusion. Reciprocal Rank Fusion needs no score normalisation between the dense and
# sparse retrievers, which matters because cosine similarity and BM25 scores are not on
# comparable scales. k=60 is the constant from Cormack et al. (2009).
RRF_K = 60
DEFAULT_TOP_K = 5
CANDIDATE_K = 20  # per-retriever depth before fusion

# BM25 signal gate: drop sparse hits scoring below this fraction of BM25's OWN top hit for the
# query, before they reach the fusion.
#
# This exists because RRF fuses ranks and ignores magnitude, so it cannot tell a confident BM25
# match from a desperate one. In an insurance corpus almost every chunk shares "policy", "claim"
# or "insured" with almost every query, so BM25 always returns a full ranked list, and its tail
# is noise carrying real rank positions. RRF then rewards agreement between a confident dense hit
# and a meaningless sparse one. Concretely: junk ranked #12 by both retrievers scores 2/72 = .0278
# and beats the gold clause ranked #1 by dense alone at 1/61 = .0164. That is how equal-weight RRF
# came in BELOW dense alone (83.9% vs 87.1% recall@5).
#
# The gate makes BM25 abstain instead of guessing. It keeps BM25's real wins (it alone finds the
# "24 consecutive hours" minimum-stay clause and the restore clause) without letting its noise vote.
#
# On the value: measured recall@5 is 87.1% at gates 0.3, 0.4 and 0.6, and 90.3% at 0.5. A lone
# spike inside a flat band is one question out of 31 flipping (each question is worth 3.2%), not a
# real optimum, so 0.5 is not chosen -- banking that would be fitting the eval set. 0.4 sits in the
# stable band. With n=31 no value here is validated; that needs a larger set and a held-out split.
BM25_SIGNAL_GATE = 0.4

# --- generation ---
#
# The generator runs LOCALLY by default. No API key, no billing, no per-query cost, and anyone
# who clones this repo can run the whole pipeline end to end. The Anthropic backend exists so the
# local models can be benchmarked against a frontier baseline, but it is not required.
GEN_MAX_TOKENS = 1500

# Local models, compared head to head on the same eval set. The interesting question is not
# whether a local model can write a fluent insurance answer (it can) but whether it can REFUSE --
# reject a false premise, decline a question the policy does not answer, and keep eight plan
# variants apart. That is where small models are weakest and it is this system's headline feature,
# so it is measured rather than assumed.
OLLAMA_MODELS = ["qwen2.5:14b-instruct", "llama3.1:8b"]
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 300.0

# Ollama's default context window is small, and when a prompt overflows it Ollama SILENTLY DROPS
# the overflow -- no error, no warning. The system prompt plus five retrieved clauses runs around
# 2.5k tokens, so the default would quietly truncate the very clauses the answer is supposed to be
# grounded in. The model would then appear to hallucinate content that we had in fact removed from
# its context ourselves, and we would blame the model. 8192 leaves comfortable headroom; the
# backend raises if a prompt ever fills the window anyway.
OLLAMA_CTX = 8192

# --- hosted query embedding (free, for the 512 MB no-card deploy) ---
#
# The retriever normally embeds the query with a local Sentence-BERT model (torch, ~1 GB RAM). A
# free no-card host (Render/Koyeb) only has 512 MB, so the DEPLOYED instance instead calls a hosted
# endpoint for the SAME model -- sentence-transformers/all-mpnet-base-v2 -- keeping the prebuilt
# index valid and the eval numbers honest. The server then needs no torch at all.
#
# Set EMBED_API_KEY (a free HuggingFace token) and query embedding goes to the API; leave it unset
# and the retriever loads the local model exactly as before. EMBED_API_URL defaults to HF's
# feature-extraction endpoint for the model; point it elsewhere to use another provider.
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_API_URL = os.getenv(
    "EMBED_API_URL",
    # 2026 router endpoint. The old api-inference.huggingface.co host was retired; the hf-inference
    # provider now serves feature-extraction under router.huggingface.co.
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-mpnet-base-v2/pipeline/feature-extraction",
)
EMBED_TIMEOUT = 60.0

# --- hosted demo backend (OpenAI-compatible, free) ---
#
# Only for the DEPLOYED demo. A free host cannot run Ollama, so the deployed instance talks to a
# free OpenAI-compatible API instead. Everything is env-driven so no key ever lands in the repo:
# set GEN_API_KEY (a Groq free-tier key by default) and the API path activates automatically; leave
# it unset and the app falls back to the local Ollama backend, exactly as before.
#
# The default model must support strict json_schema structured output. On Groq that means the
# gpt-oss / kimi families; llama models on Groq do json_object but not strict schema. Override
# GEN_MODEL at deploy time to track whatever the free tier currently offers.
GEN_API_KEY = os.getenv("GEN_API_KEY", "")
GEN_BASE_URL = os.getenv("GEN_BASE_URL", "https://api.groq.com/openai/v1")
GEN_MODEL = os.getenv("GEN_MODEL", "openai/gpt-oss-20b")
OPENAI_COMPAT_TIMEOUT = 120.0

# --- optional frontier baseline (requires API credit; not needed to run this project) ---
#
# Opus 4.8. Three API constraints here are load-bearing and easy to get wrong from memory:
#   1. `thinking` must be {"type": "adaptive"}. `budget_tokens` is REMOVED and returns a 400.
#      Omitting `thinking` entirely runs with NO thinking -- it is not adaptive-by-default.
#   2. `temperature`, `top_p` and `top_k` are REMOVED and return a 400. Steer with the prompt.
#   3. The minimum cacheable prompt prefix is 4096 tokens. Below that, prompt caching silently
#      does nothing -- no error, just zeros in usage. Our system prompt is ~800 tokens, far below
#      it, so caching is instrumented and reported rather than claimed.
ANTHROPIC_MODEL = "claude-opus-4-8"
GEN_EFFORT = "high"  # low | medium | high | xhigh | max

PLANS = [
    "Optima Suraksha",
    "Optima Secure",
    "Optima Super Secure",
    "Optima Secure Global",
    "Optima Secure Global Plus",
    "Optima Select",
    "Optima Lite",
    "Optima Secure +",
]
