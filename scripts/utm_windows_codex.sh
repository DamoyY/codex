#!/usr/bin/env bash
set -euo pipefail

# This script installs UTM on macOS, clones a Windows template VM, boots it,
# and installs Codex CLI over SSH inside the guest.
#
# Limitations:
# - UTM does not provide a stable CLI for creating a Windows VM from scratch.
# - You must create a Windows template VM once in the UTM GUI, ensure it boots,
#   and enable SSH (OpenSSH Server) for the user you plan to use.
#
# Usage:
#   scripts/utm_windows_codex.sh \
#     --template "$HOME/Documents/UTM/WindowsBase.utm" \
#     --name "CodexWin" \
#     --user "codex"

TEMPLATE="${HOME}/Documents/UTM/WindowsBase.utm"
VM_NAME="CodexWin"
SSH_USER="codex"
UTM_APP="/Applications/UTM.app"
UTM_DIR="${HOME}/Documents/UTM"
FORCE_INSTALL=0

log() {
  printf "[utm-codex] %s\n" "$*"
}

die() {
  printf "[utm-codex] ERROR: %s\n" "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --template)
        TEMPLATE="$2"
        shift 2
        ;;
      --name)
        VM_NAME="$2"
        shift 2
        ;;
      --user)
        SSH_USER="$2"
        shift 2
        ;;
      --force-install)
        FORCE_INSTALL=1
        shift
        ;;
      -h|--help)
        sed -n '1,60p' "$0"
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

install_utm() {
  need_cmd curl
  need_cmd jq
  need_cmd hdiutil
  need_cmd ditto

  if [[ -d "$UTM_APP" && "$FORCE_INSTALL" -eq 0 ]]; then
    log "UTM already installed at $UTM_APP (use --force-install to reinstall)."
    return
  fi

  local tmp dmg_url dmg_path vol_app vol_path
  tmp="$(mktemp -d)"
  dmg_path="${tmp}/UTM.dmg"

  log "Resolving latest UTM release..."
  dmg_url="$(curl -fsSL https://api.github.com/repos/utmapp/UTM/releases/latest \
    | jq -r '.assets[] | select(.name == "UTM.dmg") | .browser_download_url')"
  [[ -n "$dmg_url" && "$dmg_url" != "null" ]] || die "Could not find UTM.dmg URL."

  log "Downloading UTM from ${dmg_url}..."
  curl -fL "$dmg_url" -o "$dmg_path"

  log "Mounting DMG..."
  vol_path="$(hdiutil attach "$dmg_path" -nobrowse -quiet | awk '{print $3}' | tail -n1)"
  vol_app="${vol_path}/UTM.app"
  [[ -d "$vol_app" ]] || die "UTM.app not found in mounted DMG."

  log "Installing UTM to $UTM_APP..."
  rm -rf "$UTM_APP"
  ditto "$vol_app" "$UTM_APP"

  log "Detaching DMG..."
  hdiutil detach "$vol_path" -quiet || true
}

utmctl_path() {
  local p="${UTM_APP}/Contents/MacOS/utmctl"
  [[ -x "$p" ]] || die "utmctl not found at $p"
  printf "%s\n" "$p"
}

clone_and_start_vm() {
  local utmctl="$1"
  [[ -d "$TEMPLATE" ]] || die "Template VM not found: $TEMPLATE"

  mkdir -p "$UTM_DIR"
  local target="${UTM_DIR}/${VM_NAME}.utm"
  if [[ -e "$target" ]]; then
    die "Target VM already exists: $target"
  fi

  log "Cloning template VM to ${target}..."
  "$utmctl" clone "$TEMPLATE" "$target"

  log "Starting VM ${target}..."
  "$utmctl" start "$target"

  printf "%s\n" "$target"
}

wait_for_ip() {
  local utmctl="$1"
  local vm="$2"
  local ip=""
  log "Waiting for guest IP address via utmctl..."
  for _ in {1..120}; do
    ip="$("$utmctl" ip-address "$vm" 2>/dev/null | awk 'NF{print $1; exit}')"
    if [[ -n "$ip" ]]; then
      printf "%s\n" "$ip"
      return
    fi
    sleep 5
  done
  die "Timed out waiting for guest IP. Ensure guest tools/agent are installed."
}

install_codex_over_ssh() {
  local ip="$1"
  need_cmd ssh

  log "Installing Codex CLI inside the guest at ${ip} (user: ${SSH_USER})..."
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${SSH_USER}@${ip}" 'powershell -NoProfile -Command "
      $ErrorActionPreference = \"Stop\";
      if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw \"winget is required in the template VM.\";
      }
      winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements;
      npm install -g @openai/codex;
      try {
        Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null;
      } catch {
        Write-Host \"OpenSSH capability may already be present: $($_.Exception.Message)\";
      }
      Start-Service sshd;
      Set-Service -Name sshd -StartupType Automatic;
      if (-not (Get-NetFirewallRule -Name \"OpenSSH-Server-In-TCP\" -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -Name \"OpenSSH-Server-In-TCP\" -DisplayName \"OpenSSH Server (TCP-In)\" -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null;
      }
      codex --version;
    "'
}

main() {
  parse_args "$@"
  install_utm
  local utmctl vm_path ip
  utmctl="$(utmctl_path)"
  vm_path="$(clone_and_start_vm "$utmctl")"
  ip="$(wait_for_ip "$utmctl" "$vm_path")"
  log "Guest IP: ${ip}"
  install_codex_over_ssh "$ip"
  log "Done. You can SSH with: ssh ${SSH_USER}@${ip}"
}

main "$@"

