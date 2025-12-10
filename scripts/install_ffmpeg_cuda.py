#!/usr/bin/env python3
"""Install or verify FFmpeg with NVIDIA CUDA/NVENC/NVDEC support.

This script downloads the prebuilt FFmpeg binaries from the BtbN/FFmpeg-Builds
project on GitHub (https://github.com/BtbN/FFmpeg-Builds) when a suitable
GPU-enabled binary is not already available on the system. The install is
designed to work in three scenarios:

1. System-wide (e.g. Docker image build) where FFmpeg should be placed under
   /usr/local.
2. Linux or macOS conda environments, where FFmpeg is installed under the
   active CONDA_PREFIX.
3. Windows conda environments, where FFmpeg is installed within the environment
   prefix so it is available whenever the environment is activated.

The script checks whether the current ffmpeg binary reports any *cuvid* codecs
and the CUDA hwaccel. If present, installation is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GITHUB_RELEASE_API = "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest"
GITHUB_RELEASE_TAG_API_TEMPLATE = "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/{}"
USER_AGENT = "nsfw-ai-model-server-ffmpeg-installer"

# Pinning knobs -------------------------------------------------------------
#
# Set PREFERRED_RELEASE_TAG to force the script to fetch a specific release tag
# from the BtbN autobuilds (for example, "autobuild-2025-08-31-13-00"). Set to
# None to follow the project's latest release.
PREFERRED_RELEASE_TAG: str | None = None

# Set PREFERRED_FFMPEG_VERSION to filter assets by version token within their
# filename (for example, "6.1" to pick FFmpeg 6.1.x builds). Set to None to
# accept whatever version the release provides.
PREFERRED_FFMPEG_VERSION: str | None = "8.0"

ASSET_RULES = {
    ("Windows", "x86_64"): [
        {"include": ("ffmpeg-n", "win64", "gpl", "shared", ".zip"), "exclude": ()},
        {"include": ("ffmpeg-n", "win64", "gpl", ".zip"), "exclude": ("shared",)},
        {"include": ("ffmpeg-master", "win64", "gpl", "shared", ".zip"), "exclude": ()},
        {"include": ("ffmpeg-master", "win64", "gpl", ".zip"), "exclude": ("shared",)},
    ],
    ("Windows", "arm64"): [
        {"include": ("ffmpeg-n", "winarm64", "gpl", "shared", ".zip"), "exclude": ()},
        {"include": ("ffmpeg-n", "winarm64", "gpl", ".zip"), "exclude": ("shared",)},
        {"include": ("ffmpeg-master", "winarm64", "gpl", "shared", ".zip"), "exclude": ()},
        {"include": ("ffmpeg-master", "winarm64", "gpl", ".zip"), "exclude": ("shared",)},
    ],
    ("Linux", "x86_64"): [
        {"include": ("ffmpeg-n", "linux64", "gpl", ".tar.xz"), "exclude": ("shared",)},
        {"include": ("ffmpeg-n", "linux64", "gpl", ".tar.xz"), "exclude": ()},
        {"include": ("ffmpeg-master", "linux64", "gpl", ".tar.xz"), "exclude": ("shared",)},
        {"include": ("ffmpeg-master", "linux64", "gpl", ".tar.xz"), "exclude": ()},
    ],
    ("Linux", "aarch64"): [
        {"include": ("ffmpeg-n", "linuxarm64", "gpl", ".tar.xz"), "exclude": ("shared",)},
        {"include": ("ffmpeg-n", "linuxarm64", "gpl", ".tar.xz"), "exclude": ()},
        {"include": ("ffmpeg-master", "linuxarm64", "gpl", ".tar.xz"), "exclude": ("shared",)},
        {"include": ("ffmpeg-master", "linuxarm64", "gpl", ".tar.xz"), "exclude": ()},
    ],
}


def _normalize_machine(machine: str) -> str:
    normalized = machine.lower()
    if normalized in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if normalized in {"arm64", "aarch64"}:
        return "arm64" if platform.system() == "Windows" else "aarch64"
    return normalized


def _fetch_release(tag: str | None) -> dict:
    if tag:
        url = GITHUB_RELEASE_TAG_API_TEMPLATE.format(tag)
    else:
        url = GITHUB_RELEASE_API

    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(
            f"GitHub release lookup failed with status {exc.code}: {exc.reason}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to reach GitHub releases endpoint: {exc.reason}") from exc


def _select_asset(
    release_data: dict,
    system: str,
    machine: str,
    version_filter: str | None = None,
) -> dict:
    normalized_machine = _normalize_machine(machine)
    rules = ASSET_RULES.get((system, normalized_machine))
    if not rules:
        raise RuntimeError(f"Unsupported platform combination: {system} {machine}")

    assets = release_data.get("assets", [])
    for rule in rules:
        include_tokens = rule["include"]
        exclude_tokens = rule["exclude"]
        for asset in assets:
            name = asset.get("name", "")
            if version_filter and version_filter not in name:
                continue
            if all(token in name for token in include_tokens) and not any(
                token in name for token in exclude_tokens
            ):
                return asset

    raise RuntimeError(
        f"No matching FFmpeg build found for {system} {machine}; available assets: "
        + ", ".join(asset.get("name", "") for asset in assets)
    )

CHECK_DECODER_TOKEN = "cuvid"
CHECK_HWACCEL_TOKEN = "cuda"


def _log(msg: str) -> None:
    print(f"[install_ffmpeg_cuda] {msg}")


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def ffmpeg_has_cuda_support(ffmpeg_path: Path | None = None) -> bool:
    """Return True if the provided ffmpeg binary has CUDA/NVDEC support."""
    if ffmpeg_path is not None:
        if not ffmpeg_path.exists():
            return False
        executable = str(ffmpeg_path)
    else:
        executable = shutil.which("ffmpeg")
        if not executable:
            return False

    try:
        result = run_command([executable, "-hide_banner", "-decoders"])
    except FileNotFoundError:
        return False

    if result.returncode != 0:
        _log("Existing ffmpeg returned non-zero exit code; treating as unsupported.")
        return False

    decoders_output = f"{result.stdout}\n{result.stderr}".lower()
    if CHECK_DECODER_TOKEN not in decoders_output:
        # Double check using hwaccel listing; some builds may hide decoders but still list cuda
        hwaccel = run_command([executable, "-hide_banner", "-hwaccels"])
        hw_output = f"{hwaccel.stdout}\n{hwaccel.stderr}".lower()
        if CHECK_HWACCEL_TOKEN not in hw_output:
            return False

    return True


def detect_install_prefix(explicit_prefix: str | None = None) -> Path:
    if explicit_prefix:
        return Path(explicit_prefix).expanduser().resolve()

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix).resolve()

    system = platform.system()
    if system == "Windows":
        return Path(sys.exec_prefix).resolve()

    # Default POSIX prefix
    return Path("/usr/local").resolve()


def find_ffmpeg_in_prefix(prefix: Path, system: str) -> Path | None:
    candidates: list[Path] = []
    if system == "Windows":
        candidates.append(prefix / "Library" / "bin" / "ffmpeg.exe")
        candidates.append(prefix / "bin" / "ffmpeg.exe")
    else:
        candidates.append(prefix / "bin" / "ffmpeg")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def download_asset(asset: dict, dest_dir: Path) -> Path:
    url = asset.get("browser_download_url")
    name = asset.get("name")
    if not url or not name:
        raise RuntimeError("Invalid asset metadata returned from GitHub release")

    dest_file = dest_dir / name
    _log(f"Downloading {url} -> {dest_file}")

    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request) as response, open(dest_file, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except HTTPError as exc:
        raise RuntimeError(
            f"Failed to download FFmpeg asset (HTTP {exc.code}): {exc.reason}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to download FFmpeg asset: {exc.reason}") from exc

    return dest_file


def extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    _log(f"Extracting {archive_path} -> {dest_dir}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    elif archive_path.suffixes[-2:] == [".tar", ".xz"]:
        with tarfile.open(archive_path, mode="r:xz") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")

    # Archives contain a single top-level directory.
    entries = [p for p in dest_dir.iterdir() if p.is_dir()]
    if len(entries) != 1:
        raise RuntimeError(f"Unexpected archive structure within {archive_path}")
    return entries[0]


def copy_tree(src: Path, dst: Path) -> None:
    _log(f"Copying {src} -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def install_from_archive(extracted_root: Path, prefix: Path, system: str) -> None:
    subdirs = ["bin", "lib", "lib64", "include", "share"]

    if system == "Windows":
        # Place binaries and libs under the conda environment's Library directory.
        library_dir = prefix / "Library"
        for subdir in subdirs:
            source_dir = extracted_root / subdir
            if not source_dir.exists():
                continue
            if subdir == "bin":
                copy_tree(source_dir, library_dir / "bin")
            elif subdir == "lib":
                copy_tree(source_dir, library_dir / "lib")
            elif subdir == "include":
                copy_tree(source_dir, library_dir / "include")
            else:
                copy_tree(source_dir, library_dir / subdir)
    else:
        # POSIX: install directly under prefix to ensure PATH finds the binaries.
        for subdir in subdirs:
            source_dir = extracted_root / subdir
            if not source_dir.exists():
                continue
            copy_tree(source_dir, prefix / subdir)

        # Ensure ffmpeg/ffprobe are executable (tar preserves this, but be safe)
        bin_dir = prefix / "bin"
        if bin_dir.exists():
            for exe_name in ("ffmpeg", "ffprobe", "ffplay"):
                exe_path = bin_dir / exe_name
                if exe_path.exists():
                    exe_path.chmod(exe_path.stat().st_mode | 0o111)



def main() -> None:
    parser = argparse.ArgumentParser(description="Install FFmpeg with CUDA support")
    parser.add_argument(
        "--prefix",
        help="Installation prefix (defaults to CONDA_PREFIX or /usr/local)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall even if a CUDA-enabled ffmpeg is detected",
    )
    args = parser.parse_args()

    system = platform.system()
    machine = platform.machine()
    prefix = detect_install_prefix(args.prefix)
    _log(f"Installing under prefix: {prefix}")

    ffmpeg_in_prefix = find_ffmpeg_in_prefix(prefix, system)

    if not args.force:
        if ffmpeg_in_prefix and ffmpeg_has_cuda_support(ffmpeg_in_prefix):
            _log(
                f"CUDA-enabled ffmpeg already present at {ffmpeg_in_prefix}; skipping install."
            )
            return
        if not ffmpeg_in_prefix and args.prefix is None and ffmpeg_has_cuda_support():
            _log("CUDA-enabled ffmpeg already present on PATH; skipping install.")
            return

    env_tag_raw = os.environ.get("FFMPEG_RELEASE_TAG", "").strip()
    env_tag = env_tag_raw or None
    release_tag = env_tag if env_tag is not None else PREFERRED_RELEASE_TAG
    if release_tag and release_tag.lower() == "latest":
        release_tag = None

    asset_version = os.environ.get("FFMPEG_VERSION_FILTER", "").strip() or PREFERRED_FFMPEG_VERSION

    if release_tag:
        _log(f"Fetching FFmpeg release tagged '{release_tag}'.")
    else:
        _log("Fetching latest FFmpeg release.")

    release_data = _fetch_release(release_tag)
    if asset_version:
        _log(f"Filtering release assets for version token '{asset_version}'.")
    try:
        asset = _select_asset(release_data, system, machine, version_filter=asset_version)
    except RuntimeError as exc:
        if asset_version:
            raise RuntimeError(
                f"No FFmpeg assets in release '{release_data.get('tag_name')}' matched version token '{asset_version}'."
                " Adjust PREFERRED_FFMPEG_VERSION or set FFMPEG_VERSION_FILTER to a value present in the release."
            ) from exc
        raise
    _log(f"Selected FFmpeg build: {asset.get('name')}")

    if not prefix.exists():
        prefix.mkdir(parents=True, exist_ok=True)

    if not os.access(prefix, os.W_OK):
        raise PermissionError(
            f"Insufficient permissions to write to {prefix}. Run with elevated privileges or specify --prefix."
        )

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        archive_path = download_asset(asset, tmpdir)
        extracted_root = extract_archive(archive_path, tmpdir)
        install_from_archive(extracted_root, prefix, system)

    ffmpeg_in_prefix = find_ffmpeg_in_prefix(prefix, system)
    if ffmpeg_has_cuda_support(ffmpeg_in_prefix):
        _log("FFmpeg with CUDA support installed successfully.")
    else:
        raise RuntimeError("FFmpeg installation completed but CUDA support was not detected.")


if __name__ == "__main__":
    main()
