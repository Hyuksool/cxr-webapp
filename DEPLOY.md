# CXR Webapp Deployment Guide

## Architecture

```
Vercel (Next.js Frontend)  ←→  Railway (FastAPI Backend)
  - CXR image upload              - TorchXRayVision DenseNet (18 pathologies)
  - Results display               - CLIP Zero-Shot (CheXzero-style)
  - Radiology report              - GradCAM heatmap
                                  - Claude report generation
```

## 1. Railway Backend Deployment

### A. Login and Initialize

```bash
cd /Users/Hkwon/Documents/mybot_ver1/cxr-webapp
railway login
railway init  # Creates new project, name it: cxr-webapp
```

### B. Set Environment Variables

```bash
railway variables set ANTHROPIC_API_KEY=<your_key>
railway variables set ENABLE_ZERO_SHOT=true
```

### C. Deploy

```bash
railway up
```

Railway auto-detects `railway.toml` → uses `cxr-backend/Dockerfile`.

### D. Get Backend URL

After deploy, Railway provides URL like:
`https://cxr-webapp-production-xxxx.up.railway.app`

Health check: `GET /health` → returns model status

---

## 2. Vercel Frontend Deployment

### A. Install Vercel CLI (if needed)

```bash
npm install -g vercel
```

### B. Create Vercel Secret

```bash
vercel secrets add cxr_backend_url https://cxr-webapp-production-xxxx.up.railway.app
```
(Replace with actual Railway URL)

### C. Also add Anthropic key for Claude report generation

```bash
vercel env add ANTHROPIC_API_KEY production
# Enter your Anthropic API key when prompted
```

### D. Deploy

```bash
cd /Users/Hkwon/Documents/mybot_ver1/cxr-webapp
vercel --prod
```

---

## 3. Local Development

### Backend (Terminal 1)
```bash
cd cxr-webapp/cxr-backend
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8200
```
- Health: http://localhost:8200/health
- API docs: http://localhost:8200/docs

### Frontend (Terminal 2)
```bash
cd cxr-webapp
echo "CXR_BACKEND_URL=http://localhost:8200" > .env.local
echo "ANTHROPIC_API_KEY=your_key_here" >> .env.local
npm run dev  # → http://localhost:3001
```

---

## 4. Environment Variables Summary

| Variable | Where | Value |
|----------|-------|-------|
| `ANTHROPIC_API_KEY` | Railway + Vercel | Anthropic API key |
| `ENABLE_ZERO_SHOT` | Railway | `true` (default) or `false` |
| `CXR_BACKEND_URL` | Vercel | Railway deployment URL |
| `FRONTEND_URL` | Railway | Vercel deployment URL (for CORS) |

---

## 5. Model Download Notes

- **TorchXRayVision DenseNet**: ~100MB, auto-downloaded from PyTorch Hub on first request
- **CLIP ViT-B/32**: ~600MB, auto-downloaded from HuggingFace Hub on startup
- Railway has persistent storage for model cache (set `TRANSFORMERS_CACHE=/app/.cache`)

### Add to railway.toml for persistent model cache:
```toml
[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300  # 5 min for model download
```

---

## 6. GitHub Setup (Optional, for auto-deploy)

```bash
cd cxr-webapp
git remote add origin <your_github_repo_url>
git push -u origin main
```

Then in Railway dashboard: connect GitHub repo for auto-deploy on push.
