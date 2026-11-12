# Exploratory same-token DEV diagnostics

Not independent confirmation. P4-generated tokens are identical across evaluations; no natural-generation accuracy claim.

```json
{
  "P3": {
    "records": 192,
    "target": 192,
    "status": "DONE",
    "delta_DTR_vs_P4": {
      "n": 24,
      "mean": -0.010861023389601219,
      "ci": [
        -0.03390182329080922,
        0.010718527238687613
      ]
    },
    "scope": "post-primary DEV only, same P4-generated tokens; exploratory, not generation accuracy"
  },
  "P6": {
    "records": 192,
    "target": 192,
    "status": "DONE",
    "delta_DTR_vs_P4": {
      "n": 24,
      "mean": 0.05671881941170967,
      "ci": [
        0.03466398234050827,
        0.07908391339991191
      ]
    },
    "scope": "post-primary DEV only, same P4-generated tokens; exploratory, not generation accuracy"
  }
}
```
