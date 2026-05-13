Prediction Probability Dffusion Model

Prediction markets will resolve at either 0 or 1. Before the expiration of the contract, the price of the contract represents the market probablility of the event occcuring. In order to model this, I used a general difficusion model, calibrated it to real Polymarket API data, and evaluated the validity of the model. 

Process:
Pulled historical price data from the Polymarket API. Used this to estimate volatility paramter from observed data and simulated possible probability paths. Backtested the model using split training/test data. Finally, I constructed 90% confidence bands on the trajectory of the contract price in order to determine the stability of the model.

Results:
Actual market probability did stay within the 90% confidence interval. However, the estimated volatily paramter was very small and caused my simulated price to not vary as widely. 

Possible Conclusions:
The model does need improvement. First, the model assumes constant volatility over the period and continous movements. However, real market dynamics can be discontinous in nature. Contracts also tend to approach either 1 or 0 as they reach expiration. This model does not take that into account. 

 `Predictions.py` — data pull, calibration, simulation, and backtesting
- `Backtest_Model.pdf` — full write-up of the model and results
