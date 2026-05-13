import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt


u = "https://gamma-api.polymarket.com/markets"
r = requests.get(u)
markets = r.json()

newm = markets[5]

print(newm["question"])
print(newm["clobTokenIds"])

tokens = json.loads(newm["clobTokenIds"])
token = tokens[0]   # usually YES token

u = "https://clob.polymarket.com/prices-history"

par = {
    "market": token,
    "interval": "max",
    "fidelity": 60
}

r = requests.get(u, params=par)
data = r.json()

data1 = pd.DataFrame(data["history"])

#data clean up 

data1["time"] = pd.to_datetime(data1["t"], unit="s")
data1["p"] = data1["p"].astype(float)

data1 = data1.sort_values("time").drop_duplicates("time").reset_index(drop=True)

data1["Y"] = np.arcsin(np.sqrt(data1["p"]))
data1["dY"] = data1["Y"].diff()
data1["dt_years"] = data1["time"].diff().dt.total_seconds() / (365*24*3600)

data1 = data1[data1["dt_years"] > 0].copy()

alphaestimate = np.sqrt(
    np.nanmean((data1["dY"] ** 2) / data1["dt_years"])
)

print(data1.head())
print("Alpha Estimate:", alphaestimate)

#path plots
p0 = data1["p"].iloc[-1]
Y0 = np.arcsin(np.sqrt(p0))
T = 0.25
steps = 500
dt_sim = T / steps
Npaths = 20
times = np.linspace(0, T, steps + 1)
plt.figure(figsize=(10, 6))
for _ in range(Npaths):
    W = np.cumsum(np.sqrt(dt_sim) * np.random.randn(steps))
    W = np.insert(W, 0, 0)
    Y = Y0 + alphaestimate * W
    P = np.sin(Y) ** 2
    plt.plot(times, P)
plt.title("Prediction Market Probability Simulations")
plt.xlabel("Years")
plt.ylabel("Probability")
plt.tight_layout()
plt.show()

#backtesting model
split = int(0.5 * len(data1))
train = data1.iloc[:split].copy()
test = data1.iloc[split:].copy()
alpha_train = np.sqrt(np.nanmean((train["dY"]** 2) /train["dt_years"]))
print("Train alpha:", alpha_train)
p0 = train["p"].iloc[-1]
Y0 = np.arcsin(np.sqrt(p0))
test_times = test["time"]
test_t_years = (test_times-test_times.iloc[0]).dt.total_seconds()/(365*24*3600)
Npaths = 5000
simulated_P = []
dt_array = np.diff(test_t_years)
for _ in range(Npaths):
    Z = np.random.randn(len(dt_array))
    dW = np.sqrt(dt_array) * Z
    W = np.insert(np.cumsum(dW), 0, 0)
    Y = Y0 + alpha_train * W
    P = np.sin(Y) ** 2
    simulated_P.append(P)
simulated_P = np.array(simulated_P)
lower = np.percentile(simulated_P, 5, axis=0)
middle = np.np.median(simulated_P, axis=0)
upper = np.percentile(simulated_P, 95, axis=0)

