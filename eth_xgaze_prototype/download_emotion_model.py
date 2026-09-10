from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


URL = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-12-int8/resolve/main/emotion-ferplus-12-int8.onnx"
OUT = Path("models") / "emotion-ferplus-12-int8.onnx"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"Emotion model already exists: {OUT}")
        return
    print(f"Downloading {URL}")
    urlretrieve(URL, OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
