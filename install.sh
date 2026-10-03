#!/usr/bin/env sh
# sysone installer for Linux x86_64. Fetches everything needed to run, verifies it, and links `sysone` into ~/.local/bin:
#   - the release package (sysone + ONNX Runtime 1.28.0) from GitHub Releases, checked against SHA256SUMS
#   - the model (tokenizer, agent config, ONNX graphs; ~1.7 GB) from Hugging Face, checked file by file
# No Python, no root. Re-running resumes: files whose sha256 already matches are not downloaded again.
#
#   curl -fsSL https://raw.githubusercontent.com/inovacc/sysone/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/inovacc/sysone/main/install.sh | SYSONE_VERSION=v0.1.0 SYSONE_DIR=/opt/sysone sh
set -eu

REPO="inovacc/sysone"
MODEL_REPO="Dyam/sysone-laya-typed-decisions-onnx"
VERSION="${SYSONE_VERSION:-latest}"
DIR="${SYSONE_DIR:-$HOME/.local/share/sysone}"
BIN_DIR="${SYSONE_BIN_DIR:-$HOME/.local/bin}"

say() { printf 'sysone: %s\n' "$*"; }
die() { printf 'sysone: error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "needs '$1'"; }
need curl; need tar; need sha256sum; need uname

[ "$(uname -s)" = "Linux" ] || die "this installer is for Linux; use install.ps1 on Windows"
[ "$(uname -m)" = "x86_64" ] || die "only x86_64 is published"

sha() { sha256sum "$1" | cut -d' ' -f1; }
fetch() { # url dest sha
    if [ -f "$2" ] && [ "$(sha "$2")" = "$3" ]; then say "ok      ${2#"$DIR"/}"; return; fi
    mkdir -p "$(dirname "$2")"
    curl -fL --retry 3 --retry-delay 2 -o "$2.part" "$1" || die "download failed: $1"
    got="$(sha "$2.part")"
    [ "$got" = "$3" ] || { rm -f "$2.part"; die "checksum mismatch for $1 (expected $3, got $got)"; }
    mv "$2.part" "$2"
    say "fetched ${2#"$DIR"/}"
}

mkdir -p "$DIR" "$BIN_DIR"

# 1. release package
if [ "$VERSION" = "latest" ]; then api="https://api.github.com/repos/$REPO/releases/latest"
else api="https://api.github.com/repos/$REPO/releases/tags/$VERSION"; fi
tag="$(curl -fsSL -H 'User-Agent: sysone-installer' "$api" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n1)"
[ -n "$tag" ] || die "could not resolve release $VERSION"
tgz="sysone-$tag-linux-x64.tar.gz"
base="https://github.com/$REPO/releases/download/$tag"
say "release $tag -> $DIR"
sums="$(curl -fsSL "$base/SHA256SUMS")" || die "release $tag has no SHA256SUMS"
tgz_sha="$(printf '%s\n' "$sums" | awk -v f="$tgz" '$2 == f || $2 == "*"f {print $1}' | head -n1)"
[ -n "$tgz_sha" ] || die "SHA256SUMS has no line for $tgz"
tmp="${TMPDIR:-/tmp}/$tgz"
fetch "$base/$tgz" "$tmp" "$tgz_sha"
tar -xzf "$tmp" -C "$DIR"
chmod +x "$DIR/sysone"
say "installed sysone + ONNX Runtime"

# 2. model (convaiinnovations/laya-typed-decisions @ 1a793eb5, exported to ONNX)
while read -r rel want; do
    fetch "https://huggingface.co/$MODEL_REPO/resolve/main/$rel" "$DIR/$rel" "$want"
done <<'EOF'
model/rl_agent_config.json ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8
model/encoder/config.json 5268d24ad3b77c8151de5dcb0762ba4391619aad9ab0bda33e36fb083cfeae6d
model/tokenizer/tokenizer.json 6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30
model/tokenizer/tokenizer_config.json 08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1
bundle/encoder.onnx 639458c5ffe559df96109dc3eead9675a0fd7215fac7ce554e40d7cf366cdaae
bundle/encoder.onnx.data b4e0037c4ab0e9cd6a347e98f95d01e293541d65b5f93cd2bfb2dd3bfa75258c
bundle/head.onnx fac3ff4ef448ed17ede4abe997a2f40e38904f9a5df542b653e462ca302c51e5
bundle/head.onnx.data 222d93680f0510f6ed149906aad2c48c05a3ca2ac21853da8204d7e054344e9a
bundle/export.json 36c210196ca6c2934b1f52ca2251158a23daad7e145bcba0120c591939eff69c
EOF

# 3. command on PATH
ln -sf "$DIR/sysone" "$BIN_DIR/sysone"
"$DIR/sysone" version >/dev/null || die "sysone version failed"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "add $BIN_DIR to your PATH";; esac
say "done. Start the server:  sysone serve --threads 4   (http://127.0.0.1:8000/v1/systemone)"
