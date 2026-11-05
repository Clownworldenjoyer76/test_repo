#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/04_select/validate_markets_config.py

import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "markets.yaml"
ERROR_DIR = BASE_DIR / "errors" / "04_select"
LOG_FILE = ERROR_DIR / "validate_markets_config.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_MARKETS = {
    "moneyline": {"home", "away"},
    "puck_line": {"home", "away"},
    "total": {"over", "under"},
}

VALID_PICK_PREFERENCES = {
    "all",
    "best_ev",
    "best_prob",
}

COMMON_REQUIRED_SIDE_KEYS = {
    "enabled",
    "prob_bands",
    "my_model_prob_bands",
    "odds_bands",
    "ev_bands",
    "kelly_bands",
}

OPTIONAL_SIDE_KEYS = {
    "edge_bands",
}

LINE_REQUIRED_MARKETS = {
    "puck_line",
    "total",
}

SECONDARY_MODEL_REQUIRED_KEYS = {
    "enabled",
    "selection_mode",
    "unavailable_behavior",
    "min_train_rows",
    "disagreement_quantile",
    "derived_signal_by_market",
}

VALID_SECONDARY_SELECTION_MODES = {
    "high_disagreement_requires_secondary_support",
}

VALID_SECONDARY_UNAVAILABLE_BEHAVIORS = {
    "use_primary",
}

VALID_DERIVED_SIGNAL_BY_MARKET = {
    "moneyline": {"weighted", "meta"},
    "puck_line": {"weighted", "meta"},
    "total": {"weighted", "meta"},
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def reset_log() -> None:
    LOG_FILE.write_text(
        f"=== validate_markets_config RUN {now()} ===\n",
        encoding="utf-8",
    )


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{now()} | {message}\n")


def add_error(
    errors: list[str],
    message: str,
) -> None:
    errors.append(message)
    log(f"ERROR | {message}")


def is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_config(
    errors: list[str],
):
    if not CONFIG_PATH.exists():
        add_error(
            errors,
            f"MISSING CONFIG | {CONFIG_PATH}",
        )
        return None

    try:
        with CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = yaml.safe_load(f)

    except yaml.YAMLError as exc:
        add_error(
            errors,
            f"INVALID YAML | {CONFIG_PATH} | {exc}",
        )
        return None

    except Exception as exc:
        add_error(
            errors,
            f"READ ERROR | {CONFIG_PATH} | {exc}",
        )
        return None

    if not isinstance(data, dict):
        add_error(
            errors,
            "ROOT MUST BE A MAPPING",
        )
        return None

    markets = data.get("markets")

    if not isinstance(markets, dict):
        add_error(
            errors,
            "MISSING OR INVALID 'markets' MAPPING",
        )
        return None

    nhl = markets.get("nhl")

    if not isinstance(nhl, dict):
        add_error(
            errors,
            "MISSING OR INVALID 'markets.nhl' MAPPING",
        )
        return None

    return nhl


def validate_boolean(
    value,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(value, bool):
        add_error(
            errors,
            f"{label} MUST BE true OR false",
        )


def validate_band_list(
    bands,
    label: str,
    band_type: str,
    errors: list[str],
) -> None:
    if not isinstance(bands, list) or not bands:
        add_error(
            errors,
            f"{label} MUST BE A NON-EMPTY LIST "
            "OF TWO-ELEMENT BANDS",
        )
        return

    for index, band in enumerate(
        bands,
        start=1,
    ):
        band_label = (
            f"{label}[{index}]"
        )

        if (
            not isinstance(
                band,
                (list, tuple),
            )
            or len(band) != 2
        ):
            add_error(
                errors,
                f"{band_label} MUST CONTAIN "
                "EXACTLY TWO VALUES",
            )
            continue

        lower, upper = band

        if (
            not is_finite_number(lower)
            or not is_finite_number(upper)
        ):
            add_error(
                errors,
                f"{band_label} MUST CONTAIN "
                "FINITE NUMERIC VALUES",
            )
            continue

        lower = float(lower)
        upper = float(upper)

        if lower > upper:
            add_error(
                errors,
                f"{band_label} LOWER BOUND "
                "EXCEEDS UPPER BOUND | "
                f"lower={lower} upper={upper}",
            )

        if band_type == "prob":
            if (
                lower < 0
                or upper > 1
            ):
                add_error(
                    errors,
                    f"{band_label} PROBABILITY RANGE "
                    "MUST STAY WITHIN [0,1] | "
                    f"lower={lower} upper={upper}",
                )

        elif band_type == "kelly":
            if lower < 0:
                add_error(
                    errors,
                    f"{band_label} KELLY RANGE "
                    "CANNOT BE NEGATIVE | "
                    f"lower={lower}",
                )

        elif band_type == "ev":
            if lower < -1:
                add_error(
                    errors,
                    f"{band_label} EV LOWER BOUND "
                    "CANNOT BE BELOW -1 | "
                    f"lower={lower}",
                )


def validate_side(
    market_name: str,
    side_name: str,
    side_config,
    errors: list[str],
) -> None:
    label = (
        f"markets.nhl."
        f"{market_name}."
        f"{side_name}"
    )

    if not isinstance(
        side_config,
        dict,
    ):
        add_error(
            errors,
            f"{label} MUST BE A MAPPING",
        )
        return

    required_keys = set(
        COMMON_REQUIRED_SIDE_KEYS
    )

    if (
        market_name
        in LINE_REQUIRED_MARKETS
    ):
        required_keys.add(
            "line_bands"
        )

    allowed_keys = (
        required_keys
        | OPTIONAL_SIDE_KEYS
    )

    unknown_keys = sorted(
        set(side_config)
        - allowed_keys
    )

    if unknown_keys:
        add_error(
            errors,
            f"{label} HAS UNKNOWN KEY(S) | "
            f"{unknown_keys}",
        )

    missing_keys = sorted(
        required_keys
        - set(side_config)
    )

    if missing_keys:
        add_error(
            errors,
            f"{label} MISSING REQUIRED KEY(S) | "
            f"{missing_keys}",
        )

    if (
        market_name == "moneyline"
        and "line_bands" in side_config
    ):
        add_error(
            errors,
            f"{label}.line_bands "
            "IS NOT VALID FOR MONEYLINE",
        )

    if "enabled" in side_config:
        validate_boolean(
            side_config["enabled"],
            f"{label}.enabled",
            errors,
        )

    band_types = {
        "prob_bands": "prob",
        "my_model_prob_bands": "prob",
        "odds_bands": "odds",
        "line_bands": "line",
        "edge_bands": "edge",
        "ev_bands": "ev",
        "kelly_bands": "kelly",
    }

    for key, band_type in (
        band_types.items()
    ):
        if key in side_config:
            validate_band_list(
                side_config[key],
                f"{label}.{key}",
                band_type,
                errors,
            )


def validate_market(
    market_name: str,
    market_config,
    errors: list[str],
) -> None:
    label = (
        f"markets.nhl.{market_name}"
    )

    if not isinstance(
        market_config,
        dict,
    ):
        add_error(
            errors,
            f"{label} MUST BE A MAPPING",
        )
        return

    expected_sides = (
        EXPECTED_MARKETS[
            market_name
        ]
    )

    allowed_keys = {
        "enabled",
        "pick_preference",
        *expected_sides,
    }

    unknown_keys = sorted(
        set(market_config)
        - allowed_keys
    )

    if unknown_keys:
        add_error(
            errors,
            f"{label} HAS UNKNOWN "
            f"KEY/SIDE NAME(S) | "
            f"{unknown_keys}",
        )

    required_keys = {
        "enabled",
        "pick_preference",
        *expected_sides,
    }

    missing_keys = sorted(
        required_keys
        - set(market_config)
    )

    if missing_keys:
        add_error(
            errors,
            f"{label} MISSING REQUIRED KEY(S) | "
            f"{missing_keys}",
        )

    if "enabled" in market_config:
        validate_boolean(
            market_config[
                "enabled"
            ],
            f"{label}.enabled",
            errors,
        )

    if (
        "pick_preference"
        in market_config
    ):
        pick_preference = (
            market_config[
                "pick_preference"
            ]
        )

        if (
            pick_preference
            not in VALID_PICK_PREFERENCES
        ):
            add_error(
                errors,
                f"{label}.pick_preference INVALID | "
                f"value={pick_preference!r} | "
                f"allowed="
                f"{sorted(VALID_PICK_PREFERENCES)}",
            )

    for side_name in expected_sides:
        if side_name in market_config:
            validate_side(
                market_name,
                side_name,
                market_config[
                    side_name
                ],
                errors,
            )


def validate_secondary_model(
    config,
    errors: list[str],
) -> None:
    label = "markets.nhl.secondary_model"

    if not isinstance(config, dict):
        add_error(errors, f"{label} MUST BE A MAPPING")
        return

    unknown = sorted(
        set(config)
        - SECONDARY_MODEL_REQUIRED_KEYS
    )

    if unknown:
        add_error(
            errors,
            f"{label} HAS UNKNOWN KEY(S) | {unknown}",
        )

    missing = sorted(
        SECONDARY_MODEL_REQUIRED_KEYS
        - set(config)
    )

    if missing:
        add_error(
            errors,
            f"{label} MISSING REQUIRED KEY(S) | {missing}",
        )
        return

    validate_boolean(
        config.get("enabled"),
        f"{label}.enabled",
        errors,
    )

    mode = config.get(
        "selection_mode"
    )

    if (
        mode
        not in VALID_SECONDARY_SELECTION_MODES
    ):
        add_error(
            errors,
            f"{label}.selection_mode INVALID | "
            f"value={mode!r} | "
            f"allowed="
            f"{sorted(VALID_SECONDARY_SELECTION_MODES)}",
        )

    unavailable = config.get(
        "unavailable_behavior"
    )

    if (
        unavailable
        not in VALID_SECONDARY_UNAVAILABLE_BEHAVIORS
    ):
        add_error(
            errors,
            f"{label}.unavailable_behavior INVALID | "
            f"value={unavailable!r} | "
            f"allowed="
            f"{sorted(VALID_SECONDARY_UNAVAILABLE_BEHAVIORS)}",
        )

    min_rows = config.get(
        "min_train_rows"
    )

    if (
        not isinstance(
            min_rows,
            int,
        )
        or isinstance(
            min_rows,
            bool,
        )
        or min_rows < 100
    ):
        add_error(
            errors,
            f"{label}.min_train_rows "
            "MUST BE AN INTEGER >= 100",
        )

    quantile = config.get(
        "disagreement_quantile"
    )

    if (
        not is_finite_number(
            quantile
        )
        or abs(
            float(quantile)
            - 0.75
        ) > 1e-12
    ):
        add_error(
            errors,
            f"{label}.disagreement_quantile "
            "MUST EQUAL 0.75 FOR THE P6 CONTRACT",
        )

    derived = config.get(
        "derived_signal_by_market"
    )

    if not isinstance(
        derived,
        dict,
    ):
        add_error(
            errors,
            f"{label}.derived_signal_by_market "
            "MUST BE A MAPPING",
        )
        return

    expected = set(
        VALID_DERIVED_SIGNAL_BY_MARKET
    )

    unknown_markets = sorted(
        set(derived)
        - expected
    )

    missing_markets = sorted(
        expected
        - set(derived)
    )

    if unknown_markets:
        add_error(
            errors,
            f"{label}.derived_signal_by_market "
            "HAS UNKNOWN MARKET(S) | "
            f"{unknown_markets}",
        )

    if missing_markets:
        add_error(
            errors,
            f"{label}.derived_signal_by_market "
            "MISSING MARKET(S) | "
            f"{missing_markets}",
        )

    for (
        market,
        allowed,
    ) in (
        VALID_DERIVED_SIGNAL_BY_MARKET.items()
    ):
        if market not in derived:
            continue

        value = derived[
            market
        ]

        if value not in allowed:
            add_error(
                errors,
                f"{label}.derived_signal_by_market."
                f"{market} INVALID | "
                f"value={value!r} | "
                f"allowed={sorted(allowed)}",
            )


def validate_config(
    nhl_config,
    errors: list[str],
) -> None:
    actual_keys = set(
        nhl_config
    )

    expected_markets = set(
        EXPECTED_MARKETS
    )

    allowed_keys = (
        expected_markets
        | {"secondary_model"}
    )

    unknown_markets = sorted(
        actual_keys
        - allowed_keys
    )

    if unknown_markets:
        add_error(
            errors,
            f"INVALID MARKET NAME(S) | "
            f"{unknown_markets}",
        )

    missing_markets = sorted(
        expected_markets
        - actual_keys
    )

    if missing_markets:
        add_error(
            errors,
            f"MISSING MARKET(S) | "
            f"{missing_markets}",
        )

    if (
        "secondary_model"
        not in nhl_config
    ):
        add_error(
            errors,
            "MISSING REQUIRED "
            "markets.nhl.secondary_model",
        )
    else:
        validate_secondary_model(
            nhl_config[
                "secondary_model"
            ],
            errors,
        )

    for market_name in EXPECTED_MARKETS:
        if market_name in nhl_config:
            validate_market(
                market_name,
                nhl_config[
                    market_name
                ],
                errors,
            )


def main() -> None:
    reset_log()

    errors: list[str] = []

    log(
        f"CONFIG_PATH={CONFIG_PATH}"
    )

    nhl_config = load_config(
        errors
    )

    if nhl_config is not None:
        validate_config(
            nhl_config,
            errors,
        )

    log(
        f"VALIDATION ERRORS | "
        f"{len(errors)}"
    )

    if errors:
        log(
            "STATUS: FAILED"
        )

        print(
            "NHL markets config validation "
            f"FAILED: {len(errors)} error(s). "
            f"See {LOG_FILE}"
        )

        sys.exit(1)

    log(
        "STATUS: SUCCESS"
    )

    print(
        "NHL markets config validation "
        "complete: 0 errors."
    )


if __name__ == "__main__":
    main()
