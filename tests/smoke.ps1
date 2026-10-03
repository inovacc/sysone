# Smoke test of an installed or packaged sysone: start `sysone serve` from -Dir (which must hold the binary, the ONNX
# Runtime library, model/ and bundle/), POST every tests/smoke/<id>/request.json and require the response bytes to equal
# tests/smoke/<id>/response.json — the answers the Windows x64 build gave, byte-identical to the Python reference.
#
#   pwsh tests/smoke.ps1 -Dir <package dir>
param([Parameter(Mandatory)] [string] $Dir, [string] $Addr = '127.0.0.1:8799', [int] $Threads = 2)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $Dir $(if ($IsWindows) { 'sysone.exe' } else { 'sysone' })
$log = Join-Path ([IO.Path]::GetTempPath()) 'sysone-smoke.log'
$p = Start-Process -FilePath $exe -ArgumentList @('serve', '--addr', $Addr, '--threads', "$Threads") -PassThru -RedirectStandardError $log
try {
    $deadline = (Get-Date).AddSeconds(300)
    while ($true) {
        if ($p.HasExited) { Get-Content $log; throw "sysone exited with $($p.ExitCode)" }
        try { if ((Invoke-RestMethod "http://$Addr/health" -TimeoutSec 5).status -eq 'ok') { break } } catch { }
        if ((Get-Date) -gt $deadline) { Get-Content $log; throw 'no /health within 300 s' }
        Start-Sleep -Milliseconds 500
    }
    $fail = 0
    foreach ($case in Get-ChildItem (Join-Path $PSScriptRoot 'smoke') -Directory | Sort-Object Name) {
        $req = [IO.File]::ReadAllBytes((Join-Path $case.FullName 'request.json'))
        $want = [IO.File]::ReadAllBytes((Join-Path $case.FullName 'response.json'))
        $client = [Net.Http.HttpClient]::new(); $client.Timeout = [TimeSpan]::FromMinutes(10)
        $content = [Net.Http.ByteArrayContent]::new($req)
        $content.Headers.ContentType = 'application/json'
        $resp = $client.PostAsync("http://$Addr/v1/systemone", $content).GetAwaiter().GetResult()
        $got = $resp.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
        $same = [int]$resp.StatusCode -eq 200 -and [Linq.Enumerable]::SequenceEqual($got, $want)
        if (-not $same) {
            $fail++
            "FAIL $($case.Name) http=$([int]$resp.StatusCode)"
            "  want: $([Text.Encoding]::UTF8.GetString($want))"
            "  got:  $([Text.Encoding]::UTF8.GetString($got))"
        } else { "ok   $($case.Name)" }
    }
    if ($fail) { throw "$fail smoke case(s) differ" }
    'smoke: every response byte-identical to the reference'
} finally {
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
}
