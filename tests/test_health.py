import json
from datetime import datetime, timedelta, timezone

from knowledge_agent.embeddings.health import (
    DEFAULT_QUOTA_RESET_HOURS,
    ProviderHealth,
    parse_reset_hours,
)

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def build(tmp_path, **kwargs) -> ProviderHealth:
    return ProviderHealth(tmp_path / "provider_health.json", **kwargs)


def test_no_file_means_no_cooldown(tmp_path):
    assert build(tmp_path).active("gemini-3072", NOW) is None


def test_exhaustion_records_a_cooldown(tmp_path):
    health = build(tmp_path)

    health.record_exhausted("gemini-3072", "daily cap", now=NOW, hours=24)

    record = health.active("gemini-3072", NOW)
    assert record is not None
    assert record.reason == "daily cap"


def test_cooldown_expires(tmp_path):
    health = build(tmp_path)
    health.record_exhausted("gemini-3072", "daily cap", now=NOW, hours=1)

    assert health.active("gemini-3072", NOW + timedelta(minutes=59)) is not None
    assert health.active("gemini-3072", NOW + timedelta(minutes=61)) is None


def test_cooldown_is_keyed_per_space(tmp_path):
    health = build(tmp_path)
    health.record_exhausted("gemini-3072", "daily cap", now=NOW, hours=24)

    assert health.active("nomic-768", NOW) is None


def test_remaining_seconds_are_reported(tmp_path):
    health = build(tmp_path)
    health.record_exhausted("gemini-3072", "daily cap", now=NOW, hours=2)

    record = health.active("gemini-3072", NOW)

    assert record.seconds_remaining(NOW) == 7200


def test_clear_removes_a_cooldown(tmp_path):
    health = build(tmp_path)
    health.record_exhausted("gemini-3072", "daily cap", now=NOW, hours=24)
    health.clear("gemini-3072")

    assert health.active("gemini-3072", NOW) is None


def test_persisted_shape_has_the_three_documented_fields(tmp_path):
    path = tmp_path / "provider_health.json"
    build(tmp_path).record_exhausted("gemini-3072", "daily cap", now=NOW, hours=24)

    stored = json.loads(path.read_text(encoding="utf-8"))["gemini-3072"]

    assert set(stored) == {"reason", "unavailable_until", "recorded_at"}


def test_corrupt_file_is_ignored(tmp_path):
    path = tmp_path / "provider_health.json"
    path.write_text("{not json", encoding="utf-8")

    assert build(tmp_path).active("gemini-3072", NOW) is None


def test_non_dict_payload_is_ignored(tmp_path):
    (tmp_path / "provider_health.json").write_text("[1, 2, 3]", encoding="utf-8")

    assert build(tmp_path).active("gemini-3072", NOW) is None


def test_unparseable_timestamp_is_dropped(tmp_path):
    (tmp_path / "provider_health.json").write_text(
        json.dumps({"gemini-3072": {"reason": "x", "unavailable_until": "soon"}}),
        encoding="utf-8",
    )

    assert build(tmp_path).active("gemini-3072", NOW) is None


def test_unwritable_path_does_not_raise(tmp_path):
    path = tmp_path / "provider_health.json"
    path.write_text(json.dumps({"gemini-3072": {"reason": "x"}}), encoding="utf-8")

    read_only = build(tmp_path)

    assert read_only.records() == {}


def test_records_survive_a_new_instance(tmp_path):
    build(tmp_path).record_exhausted("gemini-3072", "daily cap", now=NOW, hours=24)

    assert build(tmp_path).active("gemini-3072", NOW) is not None


def test_record_from_message_uses_the_parsed_window(tmp_path):
    health = ProviderHealth(tmp_path / "p.json", default_reset_hours=99)
    health.record_from_message("gemini-3072", "quota resets in 3 hours", now=NOW)

    assert health.active("gemini-3072", NOW).seconds_remaining(NOW) == 3 * 3600


def test_record_from_message_falls_back_when_unparseable(tmp_path):
    health = build(tmp_path)
    health.record_from_message("gemini-3072", "resource exhausted", now=NOW)

    record = health.active("gemini-3072", NOW)

    assert record.seconds_remaining(NOW) == int(DEFAULT_QUOTA_RESET_HOURS * 3600)


def test_hours_in_message_are_parsed():
    assert parse_reset_hours("limit resets in 2 hours") == 2


def test_days_in_message_are_parsed():
    assert parse_reset_hours("available in 1 day") == 24


def test_unparseable_message_uses_the_default():
    assert parse_reset_hours("429 RESOURCE_EXHAUSTED", default=24) == 24


def test_empty_message_uses_the_default():
    assert parse_reset_hours("", default=7) == 7


def test_zero_reset_is_ignored():
    assert parse_reset_hours("resets in 0 hours", default=24) == 24


def test_configured_default_reset_hours_is_used(tmp_path):
    health = ProviderHealth(tmp_path / "p.json", default_reset_hours=6)
    health.record_exhausted("gemini-3072", "cap", now=NOW)

    assert health.active("gemini-3072", NOW).seconds_remaining(NOW) == 6 * 3600
