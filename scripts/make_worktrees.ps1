<#
.SYNOPSIS
    Create one git worktree per Person 2 lane (B, C, D, E) off person2/trueframe.

.DESCRIPTION
    Run from anywhere inside the main checkout:
        powershell -ExecutionPolicy Bypass -File scripts\make_worktrees.ps1
    Creates ..\probity-p2-<x> on branch person2/lane-<x> for each lane, then installs the locked
    environment there (uv sync --frozen, plus the recon extra for OpenCV). Existing worktrees are
    left alone. Lane A (tracking/quality) is complete and has no worktree.
#>
param(
    [string[]] $Lanes = @('b', 'c', 'd', 'e'),
    [string] $Base = 'person2/trueframe'
)

# Native tools write progress to stderr; failures are detected by exit code instead.
$ErrorActionPreference = 'Continue'

function Invoke-Checked {
    param([string] $Exe, [string[]] $Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$repo = (& git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repo) { throw 'Run this script inside the Probity checkout.' }
$repo = [System.IO.Path]::GetFullPath($repo)
$parent = Split-Path -Parent $repo

if ($repo -match '(?i)onedrive' -or ($env:OneDrive -and $repo.StartsWith($env:OneDrive, [System.StringComparison]::OrdinalIgnoreCase))) {
    Write-Warning "Repository is inside OneDrive ($repo). Sync can lock or rewrite files in worktrees; move the checkout outside OneDrive."
}

Invoke-Checked git @('-C', $repo, 'config', 'core.longpaths', 'true')

& git -C $repo show-ref --verify --quiet "refs/heads/$Base"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating $Base from HEAD"
    Invoke-Checked git @('-C', $repo, 'branch', $Base, 'HEAD')
}

foreach ($x in $Lanes) {
    $x = $x.ToLowerInvariant()
    $path = Join-Path $parent "probity-p2-$x"
    $branch = "person2/lane-$x"
    if (Test-Path -LiteralPath $path) {
        Write-Host "skip lane ${x}: $path already exists"
        continue
    }
    & git -C $repo show-ref --verify --quiet "refs/heads/$branch"
    if ($LASTEXITCODE -eq 0) {
        Invoke-Checked git @('-C', $repo, 'worktree', 'add', $path, $branch)
    } else {
        Invoke-Checked git @('-C', $repo, 'worktree', 'add', $path, '-b', $branch, $Base)
    }
    Push-Location -LiteralPath $path
    try {
        Invoke-Checked uv @('sync', '--frozen', '--extra', 'recon')
    } finally {
        Pop-Location
    }
    Write-Host "lane ${x}: $path on $branch"
}

Write-Host ''
& git -C $repo worktree list
Write-Host ''
Write-Host 'In each worktree, before merging:  uv run python scripts/check_lane.py --lane <x> --base person2/trueframe'
