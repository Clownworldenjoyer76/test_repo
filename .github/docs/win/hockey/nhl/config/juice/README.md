# NHL Stage 02 Model Calibration

The three CSV files in this directory are **model-probability calibration tables**. They are not sportsbook-vig or de-vig tables.

## `model_calibration_adjustment`

Stage 02 starts from a model fair decimal price and applies the configured adjustment:

`adjusted_decimal = fair_decimal * (1 - model_calibration_adjustment)`

It then converts the adjusted decimal price back to probability and normalizes the opposing probabilities so they sum to 1.

A positive `model_calibration_adjustment` increases that side's relative model probability after normalization. A negative value decreases it.

Sportsbook odds or market lines are used only to select the applicable calibration band. The sportsbook overround is not being removed by this transform.

## Configuration files

- `nhl_moneyline_juice.csv` selects adjustments by sportsbook American-odds band, favorite/underdog status, and venue.
- `nhl_puck_line_juice.csv` selects adjustments by puck-line value, favorite/underdog status, and venue.
- `nhl_total_juice.csv` selects adjustments by total line and side.

## Validation rule

The existing numeric coefficients are preserved as-is. They must not be changed by inspection. Any coefficient changes require out-of-sample / walk-forward revalidation before production use.
