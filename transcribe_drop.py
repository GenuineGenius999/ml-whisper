#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import sys
import shutil
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import requests
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> None:
        return None

logger = logging.getLogger("transcribe_drop")

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_CHUNK_SECONDS = 10 * 60

AUDIO_EXTS = {
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
    ".flac"
}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using default {default}.", file=sys.stderr)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    print(f"Invalid {name}={value!r}; using default {default}.", file=sys.stderr)
    return default


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing {name}. Set it in .env.", file=sys.stderr)
        sys.exit(1)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a folder and transcribe audio files via OpenAI."
    )
    parser.add_argument(
        "--drop-dir",
        default=os.getenv("DROP_DIR", "drop"),
        help="Folder to scan for audio files.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1"),
        help="OpenAI transcription model.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "output"),
        help="Folder to write transcript text files.",
    )
    parser.add_argument(
        "--processed-dir",
        default=os.getenv("PROCESSED_DIR", "processed"),
        help="Folder to move processed audio files into.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI API base URL.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=_env_float("POLL_INTERVAL", 2.0),
        help="Seconds between scans.",
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=_env_float("STABLE_SECONDS", 2.0),
        help="Seconds since last change before a file is processed.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process current files and exit.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--no-flac-to-mp3",
        action="store_true",
        default=not _env_bool("CONVERT_FLAC_TO_MP3", True),
        help="Disable auto-conversion of .flac files to .mp3 before upload.",
    )
    parser.add_argument(
        "--mp3-bitrate",
        default=os.getenv("MP3_BITRATE", "128k"),
        help="Bitrate to use when converting FLAC to MP3 (e.g. 128k, 192k).",
    )
    parser.add_argument(
        "--temp-dir",
        default=os.getenv("TEMP_DIR", "tmp"),
        help="Directory for temporary conversion/chunk files.",
    )
    parser.add_argument(
        "--max-upload-bytes",
        type=int,
        default=int(os.getenv("MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES))),
        help="Max upload size in bytes before chunking audio.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=int(os.getenv("CHUNK_SECONDS", str(DEFAULT_CHUNK_SECONDS))),
        help="Chunk duration in seconds when chunking is needed.",
    )
    return parser.parse_args()


def _iter_audio_files(drop_dir: Path):
    for path in drop_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        yield path


def _is_ready(path: Path, seen: dict, stable_seconds: float) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        seen.pop(path, None)
        return False

    previous = seen.get(path)
    seen[path] = (stat.st_size, stat.st_mtime)
    if previous is None:
        return False
    if previous != (stat.st_size, stat.st_mtime):
        return False
    return time.time() - stat.st_mtime >= stable_seconds


def _unique_path(target: Path) -> Path:
    if not target.exists():
        return target

    counter = 1
    while True:
        candidate = target.with_name(f"{target.stem}-{counter}{target.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _move_to_processed(path: Path, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_path(processed_dir / path.name)
    shutil.move(str(path), str(destination))
    logger.info("Moved to %s", destination)


def _transcribe_file(
    path: Path, api_key: str, model: str, base_url: str
) -> str:
    url = f"{base_url.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model, "response_format": "text"}

    with path.open("rb") as handle:
        files = {"file": (path.name, handle)}
        start = time.monotonic()
        response = requests.post(
            url,
            headers=headers,
            data=data,
            files=files,
            timeout=600,
        )
        elapsed = time.monotonic() - start

    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} {response.text}")

    logger.info("Received transcript for %s in %.1fs", path.name, elapsed)
    return response.text


def _convert_flac_to_mp3(source: Path, bitrate: str, temp_dir: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    temp_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"{source.stem}-",
        suffix=".mp3",
        delete=False,
        dir=str(temp_dir),
    )
    try:
        target = Path(handle.name)
    finally:
        handle.close()

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-map",
        "0:a:0",
        "-acodec",
        "libmp3lame",
        "-b:a",
        str(bitrate),
        str(target),
    ]

    start = time.monotonic()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(exc.stderr.strip() or "ffmpeg conversion failed") from exc

    elapsed = time.monotonic() - start
    try:
        src_size = source.stat().st_size
        dst_size = target.stat().st_size
        logger.info(
            "Converted %s (%.1f MB) -> %s (%.1f MB) in %.1fs",
            source.name,
            src_size / (1024 * 1024),
            target.name,
            dst_size / (1024 * 1024),
            elapsed,
        )
    except OSError:
        logger.info("Converted %s -> %s in %.1fs", source.name, target.name, elapsed)

    return target


def _try_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("Failed to delete %s: %s", path, exc)


def _probe_duration_seconds(source: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(source),
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    value = result.stdout.strip()
    duration = float(value)
    if duration <= 0:
        raise RuntimeError("non-positive duration from ffprobe")
    return duration


def _estimate_mp3_bytes(duration_seconds: float, bitrate: str) -> int:
    normalized = str(bitrate).strip().lower()
    multiplier = 1
    if normalized.endswith("k"):
        multiplier = 1_000
        normalized = normalized[:-1]
    elif normalized.endswith("m"):
        multiplier = 1_000_000
        normalized = normalized[:-1]

    bits_per_second = int(float(normalized) * multiplier)
    return int(duration_seconds * bits_per_second / 8 * 1.05)


def _chunk_audio_to_mp3_parts(
    source: Path, *, bitrate: str, chunk_seconds: int, output_dir: Path
) -> list:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"{source.stem}-part%03d.mp3"

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-b:a",
        str(bitrate),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(pattern),
    ]

    start = time.monotonic()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "ffmpeg chunking failed") from exc

    parts = sorted(output_dir.glob(f"{source.stem}-part*.mp3"))
    if not parts:
        raise RuntimeError("ffmpeg produced no chunks")

    elapsed = time.monotonic() - start
    logger.info("Created %d chunk(s) in %.1fs", len(parts), elapsed)
    return parts


def main() -> int:
    load_dotenv()
    args = _parse_args()
    api_key = _require_env("OPENAI_API_KEY")

    level_name = str(args.log_level).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        print(
            f"Invalid --log-level {args.log_level!r}; using INFO.",
            file=sys.stderr,
        )
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    drop_dir = Path(args.drop_dir)
    drop_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Watching %s", drop_dir)
    logger.info("Output dir %s", output_dir)
    logger.info("Processed dir %s", processed_dir)
    logger.info("Model %s", args.model)
    logger.info("Temp dir %s", Path(args.temp_dir))
    logger.debug("Base URL %s", args.base_url)
    logger.debug("Poll interval %.2fs", args.interval)
    logger.debug("Stable seconds %.2fs", args.stable_seconds)

    seen = {}
    while True:
        logger.debug("Scanning for audio files...")
        for path in _iter_audio_files(drop_dir):
            output_path = output_dir / f"{path.stem}.txt"
            if output_path.exists():
                seen.pop(path, None)
                logger.info(
                    "Transcript already exists (%s); skipping transcription for %s",
                    output_path.name,
                    path.name,
                )
                _move_to_processed(path, processed_dir)
                continue

            ready = True if args.once else _is_ready(path, seen, args.stable_seconds)
            if not ready:
                logger.debug("Waiting for %s to become stable...", path.name)
                continue

            effective_max = max(0, int(args.max_upload_bytes) - (1024 * 1024))
            needs_chunking = False
            if effective_max and path.stat().st_size > effective_max:
                needs_chunking = True

            if path.suffix.lower() == ".flac" and not args.no_flac_to_mp3 and not needs_chunking:
                try:
                    duration = _probe_duration_seconds(path)
                    estimated_bytes = _estimate_mp3_bytes(duration, args.mp3_bitrate)
                    if estimated_bytes > effective_max:
                        needs_chunking = True
                except Exception as exc:
                    logger.debug(
                        "Duration probe failed for %s (%s); using safe chunking.",
                        path.name,
                        exc,
                    )
                    needs_chunking = True

            with ExitStack() as stack:
                upload_parts = [path]
                if needs_chunking:
                    temp_root = Path(args.temp_dir)
                    temp_root.mkdir(parents=True, exist_ok=True)
                    parts_dir = Path(
                        stack.enter_context(
                            tempfile.TemporaryDirectory(
                                prefix=f"chunks-{path.stem}-",
                                dir=str(temp_root),
                            )
                        )
                    )
                    try:
                        upload_parts = _chunk_audio_to_mp3_parts(
                            path,
                            bitrate=args.mp3_bitrate,
                            chunk_seconds=args.chunk_seconds,
                            output_dir=parts_dir,
                        )
                    except Exception as exc:
                        logger.exception("Chunking failed for %s: %s", path.name, exc)
                        continue
                elif path.suffix.lower() == ".flac" and not args.no_flac_to_mp3:
                    try:
                        temp_mp3 = _convert_flac_to_mp3(path, args.mp3_bitrate, Path(args.temp_dir))
                        stack.callback(_try_unlink, temp_mp3)
                        upload_parts = [temp_mp3]
                    except Exception as exc:
                        logger.exception(
                            "FLAC->MP3 conversion failed for %s (%s); uploading original FLAC",
                            path.name,
                            exc,
                        )

                try:
                    texts = []
                    total = len(upload_parts)
                    for index, part in enumerate(upload_parts, start=1):
                        if total > 1:
                            logger.info("Transcribing chunk %d/%d: %s", index, total, part.name)
                        else:
                            logger.info("Transcribing %s...", part.name)
                        texts.append(
                            _transcribe_file(part, api_key, args.model, args.base_url).rstrip()
                        )
                    combined = "\n\n".join(t for t in texts if t).strip()
                    if combined:
                        combined += "\n"
                except Exception as exc:
                    logger.exception("Error transcribing %s: %s", path.name, exc)
                    continue

                output_path.write_text(combined, encoding="utf-8")
                logger.info("Wrote %s", output_path)
                seen.pop(path, None)
                _move_to_processed(path, processed_dir)

        if args.once:
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
