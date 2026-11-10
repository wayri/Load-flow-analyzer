# Four-wire LV: neutral impedance x4

The EV evening case is repeated with 4 times the neutral self-impedance.

**Status:** solver stopped; displayed values are the final iterate (line_search_failed).

## Parameters

```json
{
  "neutral_self_impedance_multiplier": 4,
  "phase_loads": "same as lv-ev-evening"
}
```

## Outputs

- [Numerical summary](study.json), [CSV summary](summary-00000.csv), [full ASCII network](network.txt).
- [Reloadable inputs](inputs/), [conductor results](nodes-00000.csv), [load results](loads-00000.csv), [element terminals](terminals-00000.csv), [iteration history](convergence-00000.csv).

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
