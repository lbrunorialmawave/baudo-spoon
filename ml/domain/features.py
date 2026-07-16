from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import final

import polars as pl


class MissingDataPolicy(Enum):
    FAIL = "fail"
    IMPUTE_ROLE_MEDIAN = "impute_role_median"
    IMPUTE_ZERO = "impute_zero"
    PROXY_FEATURE = "proxy_feature"


class Feature(ABC):
    """Abstract base class for all player feature transformations.

    Subclasses declare `name`, `required_columns`, and `missing_data_policy`
    as class-level attributes. The entry point for callers is `safe_compute`,
    which handles column availability before delegating to `compute`.
    """

    name: str
    required_columns: frozenset[str]
    missing_data_policy: MissingDataPolicy
    proxy_feature_name: str | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "missing_data_policy", None) == MissingDataPolicy.PROXY_FEATURE:
            if not getattr(cls, "proxy_feature_name", None):
                raise TypeError(
                    f"{cls.__name__}: missing_data_policy=PROXY_FEATURE requires "
                    "proxy_feature_name to be set at class level"
                )

    @abstractmethod
    def compute(self, data: pl.DataFrame) -> pl.Series:
        """Compute the feature. All required_columns are guaranteed present."""
        ...

    @final
    def safe_compute(self, data: pl.DataFrame) -> pl.Series:
        """Entry point. Checks columns, applies policy if needed, then computes."""
        missing = self.required_columns - frozenset(data.columns)
        if not missing:
            return self.compute(data)
        if self.missing_data_policy == MissingDataPolicy.FAIL:
            raise ValueError(
                f"Feature '{self.name}' requires columns {sorted(missing)} "
                f"which are absent from the input DataFrame."
            )
        return self.apply_missing_policy(data)

    def apply_missing_policy(self, data: pl.DataFrame) -> pl.Series:
        """Apply the declared policy when required_columns are partially absent."""
        policy = self.missing_data_policy

        if policy == MissingDataPolicy.IMPUTE_ZERO:
            augmented = data
            for col in self.required_columns:
                if col not in data.columns:
                    augmented = augmented.with_columns(pl.lit(0.0).alias(col))
            return self.compute(augmented)

        if policy == MissingDataPolicy.IMPUTE_ROLE_MEDIAN:
            # When an entire column is absent (not nulls within a present column),
            # there is no data to compute a median from, so we fall back to 0.0.
            # Real imputation (group-by-role median) applies only when the column
            # exists but has null values — that case is handled inside compute().
            augmented = data
            for col in self.required_columns:
                if col not in data.columns:
                    augmented = augmented.with_columns(pl.lit(0.0).alias(col))
            return self.compute(augmented)

        if policy == MissingDataPolicy.PROXY_FEATURE:
            proxy = self.proxy_feature_name
            assert proxy is not None  # guaranteed by __init_subclass__
            augmented = data
            for col in self.required_columns:
                if col not in data.columns:
                    if proxy not in data.columns:
                        raise ValueError(
                            f"Feature '{self.name}': required column '{col}' is absent "
                            f"and proxy column '{proxy}' is also absent."
                        )
                    augmented = augmented.with_columns(data[proxy].alias(col))
            return self.compute(augmented)

        # FAIL is handled in safe_compute; reaching here is a programming error.
        raise RuntimeError(f"Unhandled MissingDataPolicy: {policy}")
