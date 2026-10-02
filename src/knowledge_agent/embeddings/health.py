import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from knowledge_agent.config import DATA_DIR, GEMINI_QUOTA_RESET_HOURS, PROVIDER_HEALTH_FILE

DEFAULT_HEALTH_FILE = DATA_DIR / "provider_health.json"
DEFAULT_QUOTA_RESET_HOURS = 24.0
RESET_HINT_PATTERN = re.compile(
    r"(?:reset|available)[a-z]*\s*(?:in|at)?\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|days?|d)\b",
    re.IGNORECASE,
)


def _coerce_hours(raw: str | None) -> float:
    if raw is None:
        return DEFAULT_QUOTA_RESET_HOURS

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_QUOTA_RESET_HOURS

    return value if value > 0 else DEFAULT_QUOTA_RESET_HOURS


def parse_reset_hours(message: str, default: float = DEFAULT_QUOTA_RESET_HOURS) -> float:
    """Best-effort daily-reset hint.

    Gemini's free-tier message does not reliably state when the daily window
    rolls over, so anything unparseable falls back to ``default``. A too-short
    cooldown costs one wasted probe; a too-long one delays recovery.
    """
    match = RESET_HINT_PATTERN.search(message or "")

    if not match:
        return default

    amount = float(match.group(1))
    unit = match.group(2).lower()

    if unit.startswith(("hour", "hr", "h")):
        return amount if amount > 0 else default

    if unit.startswith(("day", "d")):
        hours = amount * 24.0
        return hours if hours > 0 else default

    return default


@dataclass(frozen=True)
class HealthRecord:
    key: str
    reason: str
    unavailable_until: datetime
    recorded_at: datetime

    def is_active(self, now: datetime) -> bool:
        return now < self.unavailable_until

    def seconds_remaining(self, now: datetime) -> int:
        return max(0, int((self.unavailable_until - now).total_seconds()))


class ProviderHealth:
    """Cooldown records for exhausted providers.

    Every read and write is best-effort: a corrupt or unwritable file degrades
    to "no health information" and must never abort an indexing run.
    """

    def __init__(
        self,
        path: Path | None = None,
        default_reset_hours: float | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else _default_path()
        self.default_reset_hours = (
            _coerce_hours(GEMINI_QUOTA_RESET_HOURS)
            if default_reset_hours is None
            else default_reset_hours
        )

    def _read(self) -> dict:
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError):
            return {}

        return data if isinstance(data, dict) else {}

    def _write(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError:
            return

    def records(self) -> dict[str, HealthRecord]:
        parsed: dict[str, HealthRecord] = {}

        for key, value in self._read().items():
            if not isinstance(value, dict):
                continue

            try:
                parsed[key] = HealthRecord(
                    key=key,
                    reason=str(value.get("reason", "")),
                    unavailable_until=datetime.fromisoformat(
                        value["unavailable_until"]
                    ),
                    recorded_at=datetime.fromisoformat(value["recorded_at"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

        return parsed

    def active(self, key: str, now: datetime | None = None) -> HealthRecord | None:
        record = self.records().get(key)

        if record is None:
            return None

        moment = now or datetime.now(timezone.utc)

        return record if record.is_active(moment) else None

    def record_exhausted(
        self,
        key: str,
        reason: str,
        now: datetime | None = None,
        hours: float | None = None,
    ) -> HealthRecord:
        moment = now or datetime.now(timezone.utc)
        window = self.default_reset_hours if hours is None else hours
        until = moment + timedelta(hours=max(window, 0.0))

        record = HealthRecord(
            key=key,
            reason=reason,
            unavailable_until=until,
            recorded_at=moment,
        )

        data = self._read()
        data[key] = {
            "reason": reason,
            "unavailable_until": until.isoformat(),
            "recorded_at": moment.isoformat(),
        }
        self._write(data)

        return record

    def record_from_message(
        self,
        key: str,
        message: str,
        now: datetime | None = None,
    ) -> HealthRecord:
        return self.record_exhausted(
            key,
            message,
            now=now,
            hours=parse_reset_hours(message, self.default_reset_hours),
        )

    def clear(self, key: str) -> None:
        data = self._read()

        if key not in data:
            return

        del data[key]
        self._write(data)


def _default_path() -> Path:
    if not PROVIDER_HEALTH_FILE:
        return DEFAULT_HEALTH_FILE

    candidate = Path(PROVIDER_HEALTH_FILE)

    return candidate if candidate.is_absolute() else PROJECT_RELATIVE(candidate)


def PROJECT_RELATIVE(candidate: Path) -> Path:
    from knowledge_agent.config import PROJECT_ROOT

    return PROJECT_ROOT / candidate