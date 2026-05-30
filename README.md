Sona Power Predict – 2026

College Name

Sona College of Technology

Team Name

CodeForce

Team Members

- Sarvesh – 2nd Year – AIML
- Sivaraman – 2nd Year – AIML
- Maniganda Balaji – 2nd Year – AIML
- Ravikiran – 2nd Year – AIML

Libraries Used

- pandas
- numpy
- xgboost
- logging
- os
- sys

Project Overview

This project predicts the Powerplay score (first 6 overs) of a T20 cricket innings using Machine Learning. The model is trained using historical ball-by-ball cricket data and predicts the expected Powerplay score based on match, team, and player statistics.

Model Approach

Data Preprocessing

- Filtered deliveries from overs 0 to 5 (Powerplay overs).
- Calculated total Powerplay runs for each innings.

Feature Engineering

The following features were used:

- Team batting average in Powerplay
- Venue scoring patterns
- Head-to-head team statistics
- Bowler economy rate
- Batsman scoring performance
- Innings-wise scoring trends

Machine Learning Model

- Algorithm: XGBoost Regressor
- Objective: Powerplay score prediction
- Model trained using engineered cricket statistics and historical match data.

Prediction Process

The model analyzes match conditions and player-related features to estimate the expected Powerplay score. The prediction is generated based on patterns learned from previous matches.

Key Features

- Accurate score prediction using XGBoost
- Efficient handling of structured cricket data
- Feature-engineered inputs for better performance
- Useful for cricket analytics and forecasting

Team CodeForce

Thank you for reviewing our project.
