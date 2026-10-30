## Goal
Add a training-load warning: compare the last 7 days of running distance with the
average weekly distance of the last 28 days (acute:chronic workload ratio).

## Changes
1. analytics.py – new function training_load(session, today=None) returning
   {"acute_km", "chronic_km_per_week", "ratio", "warning"}; warning if ratio > 1.3;
   ratio is None when there is no chronic data (no division by zero)
2. api.py – new endpoint GET /analytics/training-load
3. agent.py – register it in TOOLS as "get_training_load"
4. tests – ratio math, empty database, and the API endpoint
5. README – add the endpoint to the table

## Rules
- Follow the existing patterns (read weekly_mileage and its tests first)
- mypy --strict and pytest must pass
- Do not change existing endpoints or behaviour

## Done when
All tests pass and the agent can use the new tool.
