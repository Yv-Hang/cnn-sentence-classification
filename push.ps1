# push.ps1 —— 用 Windows 凭据管理器里的 GitHub 凭据推送（无需交互登录）
# 用法：
#   .\push.ps1                    # 提交信息默认 "update"
#   .\push.ps1 "改了什么"
#
# 背景：DSH 沙箱里 git 调不动 credential helper（cygwin sh 建不了命名管道），
# 所以这里先用 git-credential-wincred.exe 直接读出凭据，再以 http.extraheader
# 的方式交给 git，全程不需要交互式登录。

param([string]$Message = "update")

$ErrorActionPreference = "Stop"

$gitExe = (Get-Command git).Source
$gitRoot = Split-Path (Split-Path $gitExe -Parent) -Parent
$helper = Join-Path $gitRoot "mingw64\libexec\git-core\git-credential-wincred.exe"
if (-not (Test-Path $helper)) { throw "找不到 $helper" }

$cred = ("protocol=https`nhost=github.com`n`n" | & $helper get) | Out-String
$user = ([regex]::Match($cred, 'username=(.+)')).Groups[1].Value.Trim()
$tok = ([regex]::Match($cred, 'password=(.+)')).Groups[1].Value.Trim()
if (-not $user -or -not $tok) { throw "凭据管理器里没有 github.com 的凭据，请先 git push 登录一次" }

$b64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${user}:${tok}"))

git add -A
$staged = git diff --cached --name-only
if ($staged) {
    git commit -m $Message | Out-Null
    "已提交：$Message"
} else {
    "没有新改动，直接推送已有提交"
}

git -c http.extraheader="AUTHORIZATION: basic $b64" push origin main
"推送完成：https://github.com/$user/cnn-sentence-classification"
