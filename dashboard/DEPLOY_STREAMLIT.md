# Streamlit Deployment Branch

Use this when deploying the teacher dashboard to Streamlit Community Cloud.

## Why

The full thesis repository has a heavy Python environment. For the deployed dashboard, Streamlit only needs a small subset of packages. Deploying the full stack slows builds and can cause flaky installs.

Files:

- `requirements.full.txt`: the full thesis environment
- `requirements.streamlit.txt`: the minimal dashboard environment

## Fastest workflow

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\dashboard\prepare_streamlit_branch.ps1
git add requirements.txt requirements.streamlit.txt requirements.full.txt dashboard/prepare_streamlit_branch.ps1 dashboard/DEPLOY_STREAMLIT.md
git commit -m "Prepare Streamlit deployment branch"
git push origin streamlit-deploy
```

That script:

1. creates a branch called `streamlit-deploy`
2. replaces the root `requirements.txt` with the minimal Streamlit set

## Streamlit Community Cloud settings

Use:

- repo: `LGP-score`
- branch: `streamlit-deploy`
- main file: `dashboard/streamlit_app.py`

Keep your existing app secrets in place.

## If you want to do it manually

```powershell
git switch -c streamlit-deploy
Copy-Item .\requirements.streamlit.txt .\requirements.txt -Force
git add requirements.txt requirements.streamlit.txt requirements.full.txt
git commit -m "Use minimal dependencies for Streamlit deployment"
git push origin streamlit-deploy
```
