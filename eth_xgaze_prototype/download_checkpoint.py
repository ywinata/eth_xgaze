from __future__ import annotations

from pathlib import Path

import gdown


ETH_XGAZE_DIR = Path(r"C:\Users\ywinata_kadence\Documents\CV Code\eth-xgaze")
CHECKPOINT = ETH_XGAZE_DIR / "ckpt" / "epoch_24_ckpt.pth.tar"
FILE_ID = "1Ma6zJrECNTjo_mToZ5GKk7EF-0FS4nEC"


def main() -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists():
        print(f"Already exists: {CHECKPOINT}")
        return
    gdown.download(id=FILE_ID, output=str(CHECKPOINT), quiet=False)
    print(f"Saved: {CHECKPOINT}")


if __name__ == "__main__":
    main()
