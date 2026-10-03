# sysone installer for Windows x64. Fetches everything needed to run, verifies it, and puts `sysone` on PATH:
#   - the release package (sysone.exe + ONNX Runtime 1.28.0) from GitHub Releases, checked against SHA256SUMS
#   - the model (tokenizer, agent config, ONNX graphs; ~1.7 GB) from Hugging Face, checked file by file
# No Python, no admin rights. Re-running resumes: files whose sha256 already matches are not downloaded again.
#
#   irm https://raw.githubusercontent.com/inovacc/sysone/main/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/inovacc/sysone/main/install.ps1))) -Version v0.1.0 -Dir D:\sysone
param(
    [string] $Version = 'latest',
    [string] $Dir = (Join-Path $env:LOCALAPPDATA 'Programs\sysone'),
    [switch] $NoPath
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is ~10x slower with the progress bar
$Repo = 'inovacc/sysone'
$ModelRepo = 'Dyam/sysone-laya-typed-decisions-onnx'

# The model files and their sha256 (convaiinnovations/laya-typed-decisions @ 1a793eb5, exported to ONNX).
$ModelFiles = [ordered]@{
    'model/rl_agent_config.json'            = 'ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8'
    'model/encoder/config.json'             = '5268d24ad3b77c8151de5dcb0762ba4391619aad9ab0bda33e36fb083cfeae6d'
    'model/tokenizer/tokenizer.json'        = '6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30'
    'model/tokenizer/tokenizer_config.json' = '08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1'
    'bundle/encoder.onnx'                   = '639458c5ffe559df96109dc3eead9675a0fd7215fac7ce554e40d7cf366cdaae'
    'bundle/encoder.onnx.data'              = 'b4e0037c4ab0e9cd6a347e98f95d01e293541d65b5f93cd2bfb2dd3bfa75258c'
    'bundle/head.onnx'                      = 'fac3ff4ef448ed17ede4abe997a2f40e38904f9a5df542b653e462ca302c51e5'
    'bundle/head.onnx.data'                 = '222d93680f0510f6ed149906aad2c48c05a3ca2ac21853da8204d7e054344e9a'
    'bundle/export.json'                    = '36c210196ca6c2934b1f52ca2251158a23daad7e145bcba0120c591939eff69c'
}

function Say([string] $m) { Write-Host "sysone: $m" }
function Sha256([string] $p) { (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower() }
function Fetch([string] $url, [string] $dest, [string] $sha) {
    if ((Test-Path $dest) -and (Sha256 $dest) -eq $sha) { Say "ok      $($dest.Substring($Dir.Length + 1))"; return }
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    $tmp = "$dest.part"
    for ($i = 1; $i -le 3; $i++) {
        try { Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing; break }
        catch { if ($i -eq 3) { throw "download failed: $url ($_)" }; Start-Sleep -Seconds (2 * $i) }
    }
    $got = Sha256 $tmp
    if ($got -ne $sha) { Remove-Item $tmp -Force; throw "checksum mismatch for $url`n  expected $sha`n  got      $got" }
    Move-Item $tmp $dest -Force
    Say "fetched $($dest.Substring($Dir.Length + 1))"
}

if (-not [Environment]::Is64BitOperatingSystem) { throw 'sysone needs 64-bit Windows' }
New-Item -ItemType Directory -Force $Dir | Out-Null

# 1. release package
$api = if ($Version -eq 'latest') { "https://api.github.com/repos/$Repo/releases/latest" } else { "https://api.github.com/repos/$Repo/releases/tags/$Version" }
$rel = Invoke-RestMethod -Uri $api -Headers @{ 'User-Agent' = 'sysone-installer' }
$tag = $rel.tag_name
$zipName = "sysone-$tag-windows-x64.zip"
$zipAsset = $rel.assets | Where-Object name -eq $zipName
$sumAsset = $rel.assets | Where-Object name -eq 'SHA256SUMS'
if (-not $zipAsset -or -not $sumAsset) { throw "release $tag has no $zipName / SHA256SUMS" }
Say "release $tag -> $Dir"
$sums = (Invoke-WebRequest -Uri $sumAsset.browser_download_url -UseBasicParsing).Content
$sumText = if ($sums -is [byte[]]) { [Text.Encoding]::UTF8.GetString($sums) } else { $sums }
$zipSha = ($sumText -split "`n" | Where-Object { $_ -match "\s\*?$([regex]::Escape($zipName))\s*$" } | ForEach-Object { ($_ -split '\s+')[0] }) | Select-Object -First 1
if (-not $zipSha) { throw "SHA256SUMS has no line for $zipName" }
$zipPath = Join-Path $env:TEMP $zipName
if (-not ((Test-Path $zipPath) -and (Sha256 $zipPath) -eq $zipSha.ToLower())) {
    Invoke-WebRequest -Uri $zipAsset.browser_download_url -OutFile $zipPath -UseBasicParsing
    if ((Sha256 $zipPath) -ne $zipSha.ToLower()) { throw "checksum mismatch for $zipName" }
}
Get-Process -Name sysone -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$Dir*" } | ForEach-Object { throw "sysone is running from $Dir (pid $($_.Id)); stop it first" }
Expand-Archive -Path $zipPath -DestinationPath $Dir -Force
Say "installed sysone.exe + ONNX Runtime"

# 2. model
foreach ($rel in $ModelFiles.Keys) {
    Fetch "https://huggingface.co/$ModelRepo/resolve/main/$rel" (Join-Path $Dir ($rel -replace '/', '\')) $ModelFiles[$rel]
}

# 3. PATH (user scope)
if (-not $NoPath) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not (($userPath -split ';') -contains $Dir)) {
        [Environment]::SetEnvironmentVariable('Path', (($userPath.TrimEnd(';') + ";$Dir").TrimStart(';')), 'User')
        Say "added $Dir to your user PATH (open a new terminal)"
    }
}

& (Join-Path $Dir 'sysone.exe') version | Out-Null
if ($LASTEXITCODE) { throw "sysone.exe version -> exit $LASTEXITCODE" }
Say "done. Start the server:  sysone serve --threads 4   (http://127.0.0.1:8000/v1/systemone)"
