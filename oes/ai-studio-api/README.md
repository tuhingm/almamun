# OES AI Studio API

This Worker connects the OES AI Studio frontend to Gemini and the GitHub repository `tuhingm/almamun`.

## What it does

- Authenticates only the repository owner account `@tuhingm` through GitHub OAuth Device Flow.
- Reads selected OES source files from the repository.
- Sends the user's instruction and selected source context to the Gemini API.
- Returns proposed complete file replacements for review in the browser.
- Publishes approved changes as a GitHub pull request. It never writes directly to `main`.

## Required setup

1. Create a **GitHub OAuth App** under GitHub Settings → Developer settings → OAuth Apps.
   - Enable **Device Flow** for the app.
   - Set the application name to `OES AI Studio`.
   - The authorization callback URL is not used by Device Flow.
   - Copy the Client ID.
2. In Cloudflare, create a KV namespace named `OES_SESSIONS`.
3. Replace the KV namespace ID in `wrangler.toml` with the ID from Cloudflare.
4. Install Wrangler and authenticate:
   ```bash
   npm install -g wrangler
   wrangler login
   ```
5. Set secrets (do not put keys in the website source):
   ```bash
   wrangler secret put GEMINI_API_KEY
   wrangler secret put GITHUB_CLIENT_ID
   ```
   Paste the Google AI Studio API key and GitHub OAuth App Client ID when prompted.
6. Deploy from this directory:
   ```bash
   wrangler deploy
   ```
7. Copy the deployed Worker URL (for example, `https://oes-ai-studio-api.<your-subdomain>.workers.dev`) and set it in the OES AI Studio page's **API endpoint** setting.

## Security and publishing model

- Gemini and GitHub credentials stay in Cloudflare secrets/KV, never in browser JavaScript.
- GitHub authorization requests only the `public_repo` scope for this public repository, and the Worker rejects any GitHub account other than `@tuhingm`.
- Source access is restricted to `oes/` and `assets/` files.
- Gemini output is path-validated and size-limited.
- Approved changes are committed on a new branch and opened as a pull request for review.
- Sessions expire after seven days. Use Disconnect to delete the active session.

## API endpoints

- `GET /api/health`
- `POST /api/auth/device`
- `POST /api/auth/poll`
- `POST /api/auth/logout`
- `GET /api/repo/tree`
- `POST /api/ai/edit`
- `POST /api/repo/pull-request`

The Worker calls the Gemini Interactions API (the recommended interface for new agentic workflows), uses `store: false` for stateless requests, and defaults to `gemini-3.8-flash`. Override the model with the Worker variable `GEMINI_MODEL`.
