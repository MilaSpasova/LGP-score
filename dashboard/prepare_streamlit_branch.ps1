param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$BranchName = "streamlit-deploy"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path $RepoRoot).Path
Set-Location $repoRoot

$streamlitReq = Join-Path $repoRoot "requirements.streamlit.txt"
$rootReq = Join-Path $repoRoot "requirements.txt"

if (-not (Test-Path $streamlitReq)) {
    throw "Missing $streamlitReq"
}

git switch -c $BranchName
Copy-Item $streamlitReq $rootReq -Force

Write-Host ""
Write-Host "Created branch '$BranchName' and replaced root requirements.txt with the Streamlit-only dependency set."
Write-Host "Next steps:"
Write-Host "  git add requirements.txt requirements.streamlit.txt requirements.full.txt dashboard/prepare_streamlit_branch.ps1 dashboard/DEPLOY_STREAMLIT.md"
Write-Host "  git commit -m `"Prepare Streamlit deployment branch`""
Write-Host "  git push origin $BranchName"
