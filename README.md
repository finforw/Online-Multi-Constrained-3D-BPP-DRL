# Run Unit Tests
python -m env.env_test

# Train model
python -m model.main

# Run evaluation
python -m trained_models.test_model

## Model Results (Updated)

| Method | Configuration | Average Reward | Average Utilization | Average COG Distance |
| --- | --- | ---: | ---: | ---: |
| SOTA | Soft constraints | 8.088 | 71.021% | 1.478 |
| SOTA | Soft + hard constraints | 4.260 | 35.873% | 2.835 |

### SOTA Trained with Constraints Results

| Configuration | Average Reward | Average Utilization | Average COG Distance |
| --- | ---: | ---: | ---: |
| No constraints | 7.102 | 71.021% | 1.478 |
| Soft constraints on COG | 7.686 | 66.809% | 1.396 |
| Soft constraints on COG + hard constraints on ETA | 6.747 | 58.075% | 1.679 |

### Our Model Results

| Configuration | Average Reward | Average Utilization | Average COG Distance |
| --- | ---: | ---: | ---: |
| No constraints | 7.910 | 79.098% | 1.374 |
| Soft constraints on COG (negative reward) | 7.951 | 69.590% | 1.453 |
| Hard constraints on ETA | 6.705 | 64.301% | 1.520 |
| Soft constraints on COG + hard constraints on ETA | ? | ? | ? |