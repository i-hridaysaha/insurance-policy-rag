# Deploying the free live demo (HuggingFace Spaces + Groq)

The demo is fully free: a HuggingFace **Docker Space** (2 vCPU / 16 GB, no card) runs the retrieval
stack and FastAPI, and generation goes to **Groq's** free OpenAI-compatible API. No GPU, no billing.

Retrieval runs from the pre-built index bundled into the Space. The source PDF is **not** shipped
(it stays gitignored); only the derived index and the extracted clause text travel with the demo,
which is what a public demo of this document necessarily exposes.

## 1. Get a free Groq API key

1. Sign up at https://console.groq.com (free, no card).
2. Create a key at https://console.groq.com/keys. Copy it.

## 2. Create the Space

1. https://huggingface.co/new-space
2. Owner = you, name = `insurance-policy-rag`, **SDK = Docker** (blank template), visibility = Public.
3. Create.

## 3. Set the generation secret

In the Space's **Settings -> Variables and secrets**:

- Secret `GEN_API_KEY` = your Groq key (secret, so it never lands in the repo).
- (optional) Variable `GEN_MODEL` = the model to use. It must support **strict json_schema**
  structured output. On Groq that means the gpt-oss / kimi families:
  - `openai/gpt-oss-20b` (default, fast)
  - `openai/gpt-oss-120b` (stronger)
  - `moonshotai/kimi-k2-instruct`

  Llama models on Groq do `json_object` but **not** strict schema, so they will not satisfy the
  citation contract. Check https://console.groq.com/docs/models for the current ids.
- (optional) Variable `GEN_BASE_URL` if you point at a different OpenAI-compatible provider.

Leaving `GEN_API_KEY` unset makes the app fall back to local Ollama, which a free host cannot run,
so the secret is required for the hosted demo.

## 4. Push the code + the bundled index

The index (`data/processed/`) is gitignored in the GitHub repo, so it must be **force-added** for
the Space. Do it on a throwaway deploy branch so the GitHub main branch is untouched:

```bash
git remote add space https://huggingface.co/spaces/<your-hf-username>/insurance-policy-rag

git checkout -b space-deploy
cp docs/space-README.md README.md      # the Space reads its Docker config from README.md frontmatter
git add -f data/processed              # ship the pre-built index (normally ignored)
git add README.md
git commit -m "Deploy: Space config + bundled index"
git push space space-deploy:main       # HF Spaces builds from its main branch

git checkout main                      # back to the clean branch; main on GitHub is unchanged
```

The Space builds the Docker image (pre-downloads the ~420 MB Sentence-BERT model into the image, so
first request is not slow), then starts on port 7860. First build takes a few minutes.

## 5. Use it

Open `https://huggingface.co/spaces/<your-hf-username>/insurance-policy-rag`. The single-page UI is
served at `/`; the API is at `/ask` and `/health`.

## Updating later

Repeat step 4 on a fresh `space-deploy` branch (delete the old one first with
`git branch -D space-deploy`). Only the Space's main branch matters to HuggingFace.

## Notes

- Free Spaces sleep after ~48 h idle and wake on the next request (a slow first hit, then normal).
- The Groq free tier is rate-limited; fine for a portfolio demo, not for load.
- Nothing here changes the local experience: clone the repo, run Ollama, and `GEN_API_KEY` stays
  unset, so the app runs the local model exactly as before.
