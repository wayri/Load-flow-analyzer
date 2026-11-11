# Economics module

[Run and manage economic dispatch](../docs/economics.md) for a battery station,
generator, distribution company or aggregator using configurable forecasts,
prices, asset limits and costs.

The [worked examples and plots](examples/README.md) cover all four roles, a
six-hour rolling replan, battery sizing and unbalanced AC replay.

```sh
python -m nrlarge economics example --role battery --out my-battery
python -m nrlarge economics optimize --case my-battery --out dispatch --baseline
python -m nrlarge economics check --case dispatch/inputs --result dispatch
```

See [acknowledgements, disclaimer and waiver](../ACKNOWLEDGEMENTS.md).
