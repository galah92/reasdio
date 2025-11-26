"""Download AudioCaps audio files for ReasonAQA dataset."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_youtube_id(filepath: str) -> str | None:
    """Extract YouTube ID from AudioCaps filepath.

    Example: AudioCapsLarger/train/Y9GDZfBq_SlE.wav -> 9GDZfBq_SlE
    The 'Y' prefix is added by AudioCaps, the actual ID follows.
    """
    if not filepath or "AudioCaps" not in filepath:
        return None
    filename = Path(filepath).stem  # e.g., Y9GDZfBq_SlE
    if filename.startswith("Y"):
        return filename[1:]  # Remove the Y prefix
    return filename


def download_youtube_audio(youtube_id: str, output_path: Path, sample_rate: int = 32000) -> bool:
    """Download audio from YouTube video."""
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path),
        "--postprocessor-args", f"-ar {sample_rate}",
        "--quiet",
        "--no-warnings",
        url
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def get_unique_audiocaps_files(json_paths: list[Path]) -> set[str]:
    """Get unique AudioCaps filepaths from JSON files."""
    files = set()
    for json_path in json_paths:
        with open(json_path) as f:
            data = json.load(f)
        for item in data:
            if item["filepath1"] and "AudioCaps" in item["filepath1"]:
                files.add(item["filepath1"])
            if item.get("filepath2") and "AudioCaps" in item["filepath2"]:
                files.add(item["filepath2"])
    return files


def main():
    parser = argparse.ArgumentParser(description="Download AudioCaps audio files")
    parser.add_argument("--data-dir", type=Path, default=Path("data/reasonaqa"),
                        help="Directory containing train/val/test.json")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="Output directory for audio files")
    parser.add_argument("--max-files", type=int, default=100,
                        help="Maximum number of files to download (for testing)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel download workers")
    args = parser.parse_args()

    # Get unique AudioCaps files
    json_files = list(args.data_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    files = get_unique_audiocaps_files(json_files)
    print(f"Found {len(files)} unique AudioCaps files")

    # Limit for testing
    files = list(files)[:args.max_files]
    print(f"Downloading {len(files)} files...")

    # Download files
    success = 0
    failed = 0

    def download_file(filepath: str) -> tuple[str, bool]:
        youtube_id = extract_youtube_id(filepath)
        if not youtube_id:
            return filepath, False

        output_path = args.output_dir / filepath
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            return filepath, True

        result = download_youtube_audio(youtube_id, output_path)
        return filepath, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_file, f): f for f in files}
        for future in as_completed(futures):
            filepath, result = future.result()
            if result:
                success += 1
            else:
                failed += 1
            print(f"\rProgress: {success + failed}/{len(files)} (success: {success}, failed: {failed})", end="")

    print(f"\nDone! Downloaded {success}/{len(files)} files")


if __name__ == "__main__":
    main()
