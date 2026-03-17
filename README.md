# Run Unit Tests
python -m env.env_test

# Train model
python -m model.main

# Run evaluation
python -m trained_models.test_model

## Golden Model Results
Cut-1:
Average Reward: 6.387
Average Utilization: 67.990%

Fizz-Fuzz:
Average Reward: 6.496
Average Utilization: 69.095%

## Model Results (Updated)

SOTA with soft constraints:
Average Reward: 8.088
Average Utilization: 71.021%
Average COG Distance: 1.478

SOTA with soft and hard constraints:
Average Reward: 4.260
Average Utilization: 35.873%
Average COG Distance: 2.835

SOTA Trained with Constraints Results:

No constraints:
Average Reward: 7.102
Average Utilization: 71.021%
Average COG Distance: 1.478

Soft constraints on COG:
Average Reward: 7.686
Average Utilization: 66.809%
Average COG Distance: 1.396

Soft constraints on COG plus hard constraints on ETA:
Average Reward: 6.747
Average Utilization: 58.075%
Average COG Distance: 1.679

Our Model Results:

No constraints:
Average Reward: 7.203
Average Utilization: 72.032%
Average COG Distance: 1.489

Soft constraints on COG (negative reward):
Average Reward: 7.951
Average Utilization: 69.590%
Average COG Distance: 1.453

Soft constraints on COG plus hard constraints on ETA:
Average Reward: 6.836
Average Utilization: 58.833%
Average COG Distance: 1.622