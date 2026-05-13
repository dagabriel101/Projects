# Prediction Market Probability Dynamics  
*A stochastic diffusion model for bounded market probabilities using Polymarket data*

---

## Overview

Prediction market contracts are priced between 0 and 1, where the price represents the market-implied probability of an event occurring. Standard diffusion models are not well-suited for this setting, since they do not naturally respect these bounds.

In this project, I develop a stochastic model for prediction market probabilities that remains bounded in \([0,1]\). I calibrate the model using real data from the Polymarket API and evaluate its performance through simulation and backtesting.

---

## Model

\[
dP_t = \alpha \sqrt{P_t(1 - P_t)} \circ dW_t
\]

This structure ensures:

- Volatility is highest near \(P_t = 0.5\) (maximum uncertainty)  
- Volatility shrinks near 0 and 1 (event becomes more certain)  
- The process remains bounded between 0 and 1  

Using a nonlinear transformation of the probability, the process reduces to Brownian motion, allowing for straightforward calibration and simulation.

---

## Methodology

- Pulled historical market data from the Polymarket API  
- Transformed probabilities into a tractable coordinate system  
- Estimated volatility parameter from observed increments  
- Simulated future probability paths  
- Performed train/test split for backtesting  
- Constructed confidence intervals using Monte Carlo simulation  

---

## Results

- Simulated probability paths remain bounded in \([0,1]\)  
- Realized market probability remained within the **90% simulation band**  
- Uncertainty grows over time consistent with Brownian scaling  

This suggests the model provides a reasonable short-term approximation of probability dynamics.

---

## Key Insights

- Market uncertainty is highest when probabilities are near 0.5  
- Diffusion models can approximate probability dynamics over short horizons  
- Prediction markets exhibit both continuous evolution and discrete jumps  
- Confidence bands provide a useful way to assess model plausibility  

---

## Limitations

- Does not capture jumps due to news or information shocks  
- Assumes constant volatility  
- Does not account for liquidity or market microstructure  
- Near-expiration behavior is not explicitly modeled  

---

## Extensions

- Jump-diffusion models for news-driven moves  
- Time-dependent volatility  
- Drift terms based on information flow  
- Bayesian updating frameworks  

---

## Repository Structure

- `Predictions.py` — data ingestion, calibration, simulation, and backtesting  
- `Backtest_Model.pdf` — full write-up and methodology  

---

## Additional Resources

- Full write-up: `Backtest_Model.pdf`
