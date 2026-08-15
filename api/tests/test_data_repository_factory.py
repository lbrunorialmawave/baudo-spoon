"""Tests for DataRepository.from_settings() factory (WS-04 of plan.md).

Ensures that:
- The factory propagates ``reliability_weight_mode`` from the canonical settings.
- A missing ``reliability_weight_mode`` (or other required attribute) is
  rejected loudly instead of falling back to ``continuous``.
- The construction is deterministic for ``mode=bucket`` and ``mode=continuous``.
- Production routers (auction, optimizer) call the factory; this test
  is a static guard rather than an end-to-end test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from api.src.data_repository import DataRepository


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_settings(
    *,
    artifacts_dir: Path | None = None,
    reliability_weight_mode: str = "continuous",
    cache_ttl_seconds: int = 3600,
    r2_endpoint_url: str | None = None,
    r2_access_key_id: str | None = None,
    r2_secret_access_key: str | None = None,
    r2_bucket_name: str = "baudo-spoon-ml-artifacts",
) -> SimpleNamespace:
    """Build a duck-typed ``settings`` object compatible with the factory."""
    return SimpleNamespace(
        artifacts_dir=artifacts_dir or Path("/tmp/test-artifacts"),
        reliability_weight_mode=reliability_weight_mode,
        cache_ttl_seconds=cache_ttl_seconds,
        r2_endpoint_url=r2_endpoint_url,
        r2_access_key_id=r2_access_key_id,
        r2_secret_access_key=r2_secret_access_key,
        r2_bucket_name=r2_bucket_name,
    )


# ── Construction ──────────────────────────────────────────────────────────


class TestFromSettings:
    def test_propagates_continuous_mode(self, tmp_path) -> None:
        settings = _make_settings(artifacts_dir=tmp_path)
        repo = DataRepository.from_settings(settings)
        assert repo.reliability_weight_mode == "continuous"

    def test_propagates_bucket_mode(self, tmp_path) -> None:
        settings = _make_settings(reliability_weight_mode="bucket", artifacts_dir=tmp_path)
        repo = DataRepository.from_settings(settings)
        assert repo.reliability_weight_mode == "bucket"

    def test_propagates_artifacts_dir(self, tmp_path) -> None:
        custom = tmp_path / "artifacts"
        settings = _make_settings(artifacts_dir=custom)
        repo = DataRepository.from_settings(settings)
        assert repo._dir == custom  # type: ignore[attr-defined]

    def test_propagates_r2_bucket_name(self, tmp_path) -> None:
        settings = _make_settings(
            r2_endpoint_url="https://example.r2.cloudflarestorage.com",
            r2_access_key_id="AKIA-test",
            r2_secret_access_key="secret-test",
            r2_bucket_name="custom-bucket",
            artifacts_dir=tmp_path,
        )
        repo = DataRepository.from_settings(settings)
        assert (
            repo._artifact_store._r2_config.bucket_name  # type: ignore[attr-defined]
            == "custom-bucket"
        )

    def test_propagates_cache_ttl(self, tmp_path) -> None:
        settings = _make_settings(cache_ttl_seconds=120, artifacts_dir=tmp_path)
        repo = DataRepository.from_settings(settings)
        assert repo._ttl == 120  # type: ignore[attr-defined]


# ── Strictness / fail-closed ──────────────────────────────────────────────


class TestFromSettingsStrictness:
    def test_missing_reliability_weight_mode_rejected(self) -> None:
        settings = _make_settings()
        del settings.reliability_weight_mode
        with pytest.raises(AttributeError, match="reliability_weight_mode"):
            DataRepository.from_settings(settings)

    def test_missing_artifacts_dir_rejected(self) -> None:
        settings = _make_settings()
        del settings.artifacts_dir
        with pytest.raises(AttributeError, match="artifacts_dir"):
            DataRepository.from_settings(settings)

    def test_missing_r2_bucket_rejected(self) -> None:
        settings = _make_settings()
        del settings.r2_bucket_name
        with pytest.raises(AttributeError, match="r2_bucket_name"):
            DataRepository.from_settings(settings)

    def test_no_silent_fallback_to_continuous(self) -> None:
        """The whole point of WS-04: a missing mode must NOT default to continuous."""
        # A settings-like object that exposes everything EXCEPT mode.
        @dataclass
        class IncompleteSettings:
            artifacts_dir: Path = Path("/tmp")
            cache_ttl_seconds: int = 60
            r2_endpoint_url: str | None = None
            r2_access_key_id: str | None = None
            r2_secret_access_key: str | None = None
            r2_bucket_name: str = "b"

        with pytest.raises(AttributeError):
            DataRepository.from_settings(IncompleteSettings())  # type: ignore[arg-type]


# ── Parametric acceptance ─────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["bucket", "continuous"])
def test_factory_mode_round_trip(mode: str, tmp_path) -> None:
    """Acceptance: mode=bucket → repository=bucket, mode=continuous → repository=continuous."""
    settings = _make_settings(reliability_weight_mode=mode, artifacts_dir=tmp_path)
    repo = DataRepository.from_settings(settings)
    assert repo.reliability_weight_mode == mode


# ── Backward compatibility ────────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_direct_constructor_still_works(self, tmp_path) -> None:
        """Tests and fixtures may still use the direct constructor."""
        repo = DataRepository(
            artifacts_dir=tmp_path,
            reliability_weight_mode="bucket",
        )
        assert repo.reliability_weight_mode == "bucket"

    def test_direct_constructor_default_is_continuous(self, tmp_path) -> None:
        repo = DataRepository(artifacts_dir=tmp_path)
        assert repo.reliability_weight_mode == "continuous"

    def test_direct_constructor_rejects_invalid_mode(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="bucket"):
            DataRepository(
                artifacts_dir=tmp_path,
                reliability_weight_mode="invalid",
            )


# ── Static guard: production routers use the factory ──────────────────────


class TestRoutersUseFactory:
    """The whole point of WS-04: every production router must propagate mode.

    This is a static guard — it inspects the source of the router modules and
    fails if any router instantiates ``DataRepository(...)`` without going
    through ``DataRepository.from_settings(...)``.
    """

    @pytest.mark.parametrize(
        "relpath",
        [
            "api/src/routers/auction.py",
            "api/src/routers/optimizer.py",
            "api/src/main.py",
        ],
    )
    def test_no_raw_constructor_in_production(self, relpath: str, repo_root) -> None:
        from pathlib import Path

        path = Path(repo_root) / relpath
        text = path.read_text(encoding="utf-8")
        # ``DataRepository(`` (opening paren) without ``.from_settings`` is forbidden.
        # We allow ``DataRepository.from_settings(...)`` and we allow the
        # call inside the docstring (``DataRepository(...)`` inside a comment).
        bad_lines: list[str] = []
        for i, line in enumerate(text.splitlines(), start=1):
            if "DataRepository(" not in line:
                continue
            if "DataRepository.from_settings" in line:
                continue
            if line.lstrip().startswith("#") or line.lstrip().startswith('"""'):
                continue
            bad_lines.append(f"  L{i}: {line.rstrip()}")
        assert not bad_lines, (
            f"{relpath} still has raw DataRepository(...) calls — "
            f"replace with DataRepository.from_settings(settings):\n"
            + "\n".join(bad_lines)
        )


@pytest.fixture
def repo_root() -> str:
    """Path of the baudo-spoon repository root."""
    # api/tests/test_data_repository_factory.py → /baudo-spoon
    import os

    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
