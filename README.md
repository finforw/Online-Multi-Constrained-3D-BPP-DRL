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

## Golden Model Results (Updated)

No-Constraints:
Average Reward: 7.102
Average Utilization: 71.021%
Average COG Distance: 1.478

Soft constraints on COG (negative reward)
Average Reward: 7.819
Average Utilization: 68.168%
Average COG Distance: 1.407

Soft constraints on COG (negative reward) plus hard constraints on ETA
Average Reward: 6.822
Average Utilization: 58.771%
Average COG Distance: 1.658