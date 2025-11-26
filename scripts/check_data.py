"""Check data availability and compute statistics."""

import json
from collections import Counter
from pathlib import Path


def check_audio_availability(json_path: str, audio_dir: str = "data") -> dict:
    """Check how many audio files are available for a dataset split."""
    data = json.load(open(json_path))
    audio_dir = Path(audio_dir)

    stats = {
        "total": len(data),
        "available": 0,
        "missing": 0,
        "missing_files": [],
        "sources": Counter(),
    }

    seen_files = set()

    for item in data:
        filepath1 = item.get("filepath1", "")
        filepath2 = item.get("filepath2", "")

        for fp in [filepath1, filepath2]:
            if not fp or fp in seen_files:
                continue
            seen_files.add(fp)

            # Determine source
            if "AudioCaps" in fp:
                stats["sources"]["AudioCaps"] += 1
            elif "Clotho" in fp:
                stats["sources"]["Clotho"] += 1

            # Check if file exists
            full_path = audio_dir / fp
            if full_path.exists():
                stats["available"] += 1
            else:
                stats["missing"] += 1
                if len(stats["missing_files"]) < 10:
                    stats["missing_files"].append(fp)

    return stats


def main():
    print("=" * 60)
    print("DATA AVAILABILITY CHECK")
    print("=" * 60)

    for split in ["clotho_test_subset", "train", "val", "test"]:
        json_path = f"data/reasonaqa/{split}.json"
        if not Path(json_path).exists():
            print(f"\n{split}: JSON file not found")
            continue

        print(f"\n{split}:")
        stats = check_audio_availability(json_path)

        print(f"  Total samples: {stats['total']:,}")
        print(f"  Unique audio files: {stats['available'] + stats['missing']:,}")
        print(f"    Available: {stats['available']:,}")
        print(f"    Missing: {stats['missing']:,}")
        print(f"  Sources: {dict(stats['sources'])}")

        if stats["missing_files"]:
            print(f"  Missing examples (first 10):")
            for f in stats["missing_files"][:5]:
                print(f"    - {f}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    # Check Clotho
    clotho_path = Path("data/ClothoV21/development")
    if clotho_path.exists():
        clotho_count = len(list(clotho_path.glob("*.wav")))
        print(f"\nClotho: {clotho_count} files available in {clotho_path}")
    else:
        print("\nClotho: NOT AVAILABLE - Need to download from Zenodo")

    # Check AudioCaps
    audiocaps_path = Path("data/AudioCapsLarger")
    if audiocaps_path.exists():
        audiocaps_count = sum(1 for _ in audiocaps_path.rglob("*.wav"))
        print(f"AudioCaps: {audiocaps_count} files available")
    else:
        print("AudioCaps: NOT AVAILABLE - Need to download via yt-dlp")


if __name__ == "__main__":
    main()
