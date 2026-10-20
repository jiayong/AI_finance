# 把 FactorBench 克隆到固定提交，并设置 FACTORBENCH_REPO
#
# 用法：
#   .\scripts\bootstrap_factorbench.ps1
#   .\scripts\bootstrap_factorbench.ps1 -Dest D:\workspace\FactorBench-Release
#
# 为什么锁 SHA 而不是跟 main：
#   本项目的适配层按这个版本的内部 API 写的（factor_bench.portfolio.backtest、
#   BacktestConfig、performance_metrics 的签名与语义）。上游一改，适配层就会静默
#   算错或直接崩。锁 SHA 是唯一可靠的保证。

param(
    [string]$Dest = "$env:USERPROFILE\FactorBench-Release",
    [switch]$NoEnv
)

$ErrorActionPreference = "Stop"

$Repo = "https://github.com/ZhuoHan1998/FactorBench-Release.git"
$Sha  = "8dbc5b42e005a4038dd52e4790b30b467a5c1575"

Write-Host "目标目录: $Dest"
Write-Host "固定提交: $Sha"

if (Test-Path (Join-Path $Dest ".git")) {
    Write-Host "[1/3] 目录已是 git 仓库，取得该提交..."
    Push-Location $Dest
    git fetch --all --quiet
    $ok = $null -ne (git rev-parse --verify --quiet $Sha)
    if (-not $ok) {
        Write-Host "      本地没有该 SHA，尝试直接 fetch..."
        git fetch origin $Sha --quiet
    }
    git checkout $Sha --quiet
    Pop-Location
} else {
    Write-Host "[1/3] 克隆..."
    git clone $Repo $Dest
    Push-Location $Dest
    git checkout $Sha --quiet
    Pop-Location
}

Push-Location $Dest
$actual = git rev-parse HEAD
Pop-Location

if ($actual -ne $Sha) {
    throw "提交不匹配：期望 $Sha，实际 $actual"
}
Write-Host "      OK  HEAD = $actual"

Write-Host "[2/3] 校验包结构..."
$pkg = Join-Path $Dest "factor_bench"
if (-not (Test-Path $pkg)) {
    throw "找不到 $pkg —— 克隆目录不对，或上游改了结构"
}
Write-Host "      OK  factor_bench/ 存在"

if ($NoEnv) {
    Write-Host "[3/3] 跳过环境变量设置（-NoEnv）"
} else {
    Write-Host "[3/3] 设置用户级环境变量 FACTORBENCH_REPO..."
    [Environment]::SetEnvironmentVariable("FACTORBENCH_REPO", $Dest, "User")
    $env:FACTORBENCH_REPO = $Dest
    Write-Host "      OK  已设为 $Dest（新开终端生效，当前窗口已生效）"
}

Write-Host ""
Write-Host "完成。下一步："
Write-Host "  python verify_factorbench.py"
