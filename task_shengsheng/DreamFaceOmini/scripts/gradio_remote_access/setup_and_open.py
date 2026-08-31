#!/usr/bin/env python3
"""One-shot Gradio remote access: install SSH key/config, start tunnel, open browser.

Works on macOS / Linux / Windows (needs OpenSSH client + Python 3.8+).

Usage:
  python setup_and_open.py
  python setup_and_open.py --lan          # bind 0.0.0.0 for LAN (same subnet only)
  python setup_and_open.py --key-file /path/to/id_rsa

Before first run, place the private key next to this script as:
  scripts/gradio_remote_access/fgly_id_rsa
or pass --key-file.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ── Remote SSH / Gradio settings (edit here if server changes) ──
SSH_HOST_ALIAS = "fgly"
SSH_HOSTNAME = "10.217.219.2"
SSH_PORT = 2923
SSH_USER = "root"
LOCAL_PORT = 7864
REMOTE_HOST = "127.0.0.1"
REMOTE_PORT = 7863
KEY_FILENAME = "fgly_id_rsa"
# ───────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_KEY = SCRIPT_DIR / KEY_FILENAME


def ssh_dir() -> Path:
    return Path.home() / ".ssh"


def dest_key_path() -> Path:
    return ssh_dir() / KEY_FILENAME


def ssh_config_path() -> Path:
    return ssh_dir() / "config"


def build_host_block(identity_file: Path) -> str:
    identity = str(identity_file.as_posix()) if os.name != "nt" else str(identity_file)
    return (
        f"Host {SSH_HOST_ALIAS}\n"
        f"    HostName {SSH_HOSTNAME}\n"
        f"    Port {SSH_PORT}\n"
        f"    User {SSH_USER}\n"
        f"    IdentityFile {identity}\n"
        f"    StrictHostKeyChecking accept-new\n"
    )


def merge_ssh_config(block: str) -> None:
    cfg_path = ssh_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists():
        cfg_path.write_text(block + "\n", encoding="utf-8")
        return

    text = cfg_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^Host\s+{re.escape(SSH_HOST_ALIAS)}\s*$.*?(?=^Host\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(block + "\n", text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n" + block + "\n"
    cfg_path.write_text(text, encoding="utf-8")


def set_key_permissions(key_path: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ["icacls", str(key_path), "/inheritance:r"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["icacls", str(key_path), "/grant:r", f"{os.environ.get('USERNAME', '')}:R"],
            check=False,
            capture_output=True,
        )
    else:
        os.chmod(key_path, 0o600)
        try:
            os.chmod(ssh_dir(), 0o700)
        except OSError:
            pass


def install_private_key(source: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(
            f"私钥不存在: {source}\n"
            f"请把私钥放到 {BUNDLE_KEY}\n"
            f"或使用: python {Path(__file__).name} --key-file /path/to/key"
        )
    ssh_dir().mkdir(parents=True, exist_ok=True)
    dest = dest_key_path()
    shutil.copy2(source, dest)
    set_key_permissions(dest)
    print(f"[ok] 私钥已安装: {dest}")
    return dest


def port_in_use(port: int, bind_host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
            return False
        except OSError:
            return True


def wait_for_port(port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_in_use(port):
            return True
        time.sleep(0.2)
    return False


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def find_ssh_executable() -> str:
    ssh = shutil.which("ssh")
    if ssh:
        return ssh
    if os.name == "nt":
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("未找到 ssh 命令，请安装 OpenSSH 客户端。")


def start_tunnel(bind_host: str) -> subprocess.Popen:
    ssh = find_ssh_executable()
    forward = f"{bind_host}:{LOCAL_PORT}:{REMOTE_HOST}:{REMOTE_PORT}"
    cmd = [
        ssh,
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        forward,
        SSH_HOST_ALIAS,
    ]
    if bind_host == "0.0.0.0":
        cmd.insert(1, "-g")
    print(f"[run] {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def open_url(url: str) -> None:
    print(f"[open] {url}")
    if not webbrowser.open(url):
        print(f"无法自动打开浏览器，请手动访问: {url}")


def parse_args():
    parser = argparse.ArgumentParser(description="Setup SSH + open DreamFace PiD Gradio")
    parser.add_argument(
        "--key-file",
        default="",
        help=f"私钥源路径，默认使用 {BUNDLE_KEY}",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="绑定 0.0.0.0 供局域网访问（需同网段可达）",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="只建隧道，不打开浏览器",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="只安装私钥和 SSH config，不启动隧道",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key_source = Path(args.key_file).expanduser() if args.key_file else BUNDLE_KEY

    try:
        dest_key = install_private_key(key_source)
        merge_ssh_config(build_host_block(dest_key))
        print(f"[ok] SSH config 已写入: {ssh_config_path()} (Host {SSH_HOST_ALIAS})")
    except Exception as exc:
        print(f"[error] 安装失败: {exc}", file=sys.stderr)
        return 1

    if args.setup_only:
        print("[done] 仅完成 SSH 配置，未启动隧道。")
        return 0

    bind_host = "0.0.0.0" if args.lan else "127.0.0.1"
    url_host = get_lan_ip() if args.lan else "127.0.0.1"
    url = f"http://{url_host}:{LOCAL_PORT}"

    if port_in_use(LOCAL_PORT, bind_host="127.0.0.1") or port_in_use(LOCAL_PORT, bind_host="0.0.0.0"):
        print(f"[info] 本地端口 {LOCAL_PORT} 已在监听，复用现有隧道。")
        if not args.no_browser:
            open_url(url)
        print("按 Ctrl+C 退出（若隧道由本脚本启动，关闭窗口会断开）。")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0

    proc = start_tunnel(bind_host)
    time.sleep(0.5)
    if proc.poll() is not None:
        err = proc.stderr.read() if proc.stderr else ""
        print(f"[error] SSH 隧道启动失败:\n{err}", file=sys.stderr)
        return 1

    if not wait_for_port(LOCAL_PORT):
        proc.terminate()
        print("[error] 等待本地端口超时，隧道可能未建立。", file=sys.stderr)
        return 1

    print(f"[ok] 隧道已建立: 本地 {LOCAL_PORT} -> 远端 {REMOTE_HOST}:{REMOTE_PORT}")
    if args.lan:
        print(f"[info] 局域网访问: http://{get_lan_ip()}:{LOCAL_PORT} （需同网段互通）")
    if not args.no_browser:
        open_url(url)

    print("\n隧道运行中，关闭此窗口或 Ctrl+C 会断开连接。")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n[done] 隧道已关闭。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
