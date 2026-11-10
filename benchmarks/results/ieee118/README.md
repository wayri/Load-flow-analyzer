# IEEE 118-bus

Official MATPOWER-format IEEE case with source generator Q limits.

**Status:** converged (converged).

## Parameters

```json
{
  "source_file": "case118.m",
  "q_limits": "aggregate PV generator limits; SLACK Q free"
}
```

## Outputs

- [Numerical summary](study.json), [CSV summary](summary-00000.csv), [full ASCII network](network.txt).
- [Reloadable inputs](inputs/), [conductor results](nodes-00000.csv), [load results](loads-00000.csv), [element terminals](terminals-00000.csv), [iteration history](convergence-00000.csv).
- [Independent polar voltage comparison](reference-voltage.csv), [source voltage-bound checks](voltage-checks.csv), [generator checks](generator-checks.csv), [branch checks](branch-checks.csv).

## Figures

### Network diagram

![Network diagram](figures/network.png)

[Zoomable SVG](figures/network.svg)

### Phase and neutral voltage

![Phase and neutral voltage](figures/voltage.png)

[Zoomable SVG](figures/voltage.svg)

### Voltage angles

![Voltage angles](figures/angles.png)

[Zoomable SVG](figures/angles.svg)

### Convergence

![Convergence](figures/convergence.png)

[Zoomable SVG](figures/convergence.svg)

### Power balance

![Power balance](figures/power.png)

[Zoomable SVG](figures/power.svg)

[Collection index](../README.md) · [Acknowledgements and disclaimer](../../../ACKNOWLEDGEMENTS.md)
