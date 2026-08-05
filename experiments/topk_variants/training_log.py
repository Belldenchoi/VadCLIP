"""Small console and file logger used by the training entry points."""

from datetime import datetime
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_path=None, append=False):
        self.path = Path(log_path) if log_path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not append:
                self.path.write_text("", encoding="utf-8")

    def log(self, message):
        timestamped = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
        print(timestamped, flush=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as log_file:
                log_file.write(timestamped + "\n")

