# Deploying the free live demo (Render, no card)

This deploys the demo on Render's free tier: no credit card, a clean `https://` URL, nothing to
manage. It works because the deployed server does not run Sentence-BERT locally (that needs ~1 GB
RAM); instead query embedding is offloaded to a free hosted `all-mpnet-base-v2` endpoint, so the
app fits Render's 512 MB free instance. The embedding is the *same* model the index was built with,
verified to reproduce the index's results exactly, so retrieval quality is unchanged.

You need two free API keys, both **no card**:

- **HuggingFace token** (for embedding) — https://huggingface.co/settings/tokens → New token → type
  **Read** → copy.
- **Groq key** (for answer generation) — https://console.groq.com/keys → copy.

## Steps

1. Push this repo to GitHub (already done if you followed along) with the index committed. The
   `render.yaml` at the repo root and the committed `data/processed/` index are what Render needs.

2. Go to https://render.com and sign up (free, no card) — "Sign in with GitHub" is easiest.

3. **New** (top right) → **Blueprint**.

4. Connect your GitHub and pick the `insurance-policy-rag` repo. Render reads `render.yaml` and
   shows one web service.

5. It will prompt for the two secret env vars (they are marked `sync: false` so they are never
   stored in the repo):
   - `EMBED_API_KEY` = your HuggingFace token
   - `GEN_API_KEY` = your Groq key

6. Click **Apply** / **Create**. First build takes a few minutes (installs the slim requirements,
   no torch).

7. When it goes **Live**, Render shows your URL, e.g.
   `https://insurance-policy-rag.onrender.com`. Open it and ask a question.

## Put it on your portfolio

Link "Live demo" to that Render URL. Also paste it into the README's `Live demo:` line (top of
`README.md`), replacing `<your-deploy-url>`.

## What to expect

- **Cold start.** Render's free tier spins the service down after ~15 minutes idle, so the first
  visitor after a quiet spell waits ~50 s while it boots. That is the price of free-with-no-card.
  If a recruiter is likely to click at a known time, hit the URL yourself a minute beforehand.
- **Rate limits.** The free HuggingFace and Groq tiers are rate-limited. Fine for a portfolio demo,
  not for real traffic.
- **Model note.** The demo generates with a free hosted model, not the local models the eval
  measured. The UI and README say so; keep it that way — it is the honest framing.

## Model ids, if generation ever errors

- `GEN_MODEL` (env var, optional) must support strict json_schema on Groq — `openai/gpt-oss-20b`
  (default), `openai/gpt-oss-120b`, or `moonshotai/kimi-k2-instruct`. Check
  https://console.groq.com/docs/models for current ids.
- `EMBED_API_URL` (env var, optional) defaults to the HuggingFace router feature-extraction
  endpoint for `all-mpnet-base-v2`. Only change it to move embedding to another provider.
