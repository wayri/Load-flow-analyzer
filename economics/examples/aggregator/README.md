# Economic dispatch result

**Optimal schedule.** Solver status: `optimal`.

Role: `aggregator`. Currency/unit label: `CUR`. Full status, accounting, validation and any grid check are in `summary.json`.

`dispatch.csv` records each site meter separately. `battery_soc.csv` has T+1 stored-energy boundary states; its final row is not an extra interval. Generator commitment and flexible-load tables retain all assets.

`cashflow.csv` contains per-interval site costs and revenues. `horizon_charges.csv` contains incremental demand charges and fixed horizon costs; add these to interval total costs to reconcile the summary. Cumulative cash applies those charges at the final boundary.

[Asset overview](topology.txt) shows each PCC and its attached assets.

Plots are bounded overview previews; CSV files retain all intervals. Net profit means modeled horizon revenue minus the supplied costs. Prices and forecasts are supplied inputs. Net value covers the supplied horizon and cost assumptions.

Grid-check status: not requested.
