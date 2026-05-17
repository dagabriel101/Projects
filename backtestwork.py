import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if not os.path.exists("results"):
    os.makedirs("results")

mainToken = "21742633143463906290569050155826241533067272736897614950488156847949938836455"
startDate = "2024-06-01"
endDate = "2024-11-06"

fidelity = 360
window = 48

entryZ = 1.35
cost = 0.002

maxHold = 5
stopLoss = 0.035
takeProfit = 0.055
trailLoss = 0.025


def getHistory(token):
    url = "https://clob.polymarket.com/prices-history"
    start = pd.Timestamp(startDate, tz="UTC")
    end = pd.Timestamp(endDate, tz="UTC")

    rows = []
    now = start

    while now < end:
        nextTime = min(now + pd.Timedelta(days=14), end)

        params = {
            "market": token,
            "startTs": int(now.timestamp()),
            "endTs": int(nextTime.timestamp()),
            "fidelity": fidelity
        }

        answer = requests.get(url, params=params, timeout=30)
        answer.raise_for_status()

        rows += answer.json().get("history", [])
        now = nextTime

    data = pd.DataFrame(rows)
    data["time"] = pd.to_datetime(data["t"], unit="s", utc=True)
    data["price"] = data["p"].astype(float)

    return data[["time", "price"]].drop_duplicates("time").sort_values("time").reset_index(drop=True)


def makeSignal(data):
    tiny = 1e-5

    data["price"] = data["price"].clip(tiny, 1 - tiny)
    data["logit"] = np.log(data["price"] / (1 - data["price"]))
    data["move"] = data["logit"].diff()

    data["meanMove"] = data["move"].rolling(window).mean()
    data["vol"] = data["move"].rolling(window).std()
    data["level"] = data["logit"].rolling(window).mean()

    kappa = [np.nan] * len(data)

    x = data["logit"].values
    dx = data["move"].values

    for i in range(window, len(data)):
        oldX = x[i - window:i - 1]
        oldDx = dx[i - window + 1:i]

        good = np.isfinite(oldX) & np.isfinite(oldDx)

        if good.sum() < window // 2:
            continue

        design = np.column_stack([np.ones(good.sum()), oldX[good]])
        beta = np.linalg.lstsq(design, oldDx[good], rcond=None)[0]
        kappa[i] = max(0, -beta[1])

    data["kappa"] = kappa

    data["expectedMove"] = data["kappa"] * (data["level"] - data["logit"]) + data["meanMove"]
    data["modelLogit"] = data["logit"] + data["expectedMove"]
    data["modelPrice"] = 1 / (1 + np.exp(-data["modelLogit"]))

    data["gap"] = data["modelPrice"] - data["price"]
    data["gapVol"] = data["gap"].rolling(window).std()
    data["z"] = data["gap"] / data["gapVol"]

    data["volMean"] = data["vol"].rolling(window).mean()
    data["volActive"] = data["vol"] > data["volMean"]

    return data


def runBacktest(data):
    position = 0
    entryPrice = np.nan
    entryTime = None
    hold = 0
    bestPnl = 0

    positions = []
    trades = []
    reasons = []
    tradeRows = []

    for i in range(len(data)):
        oldPosition = position
        reason = ""

        price = data["price"].iloc[i]
        time = data["time"].iloc[i]
        z = data["z"].iloc[i]
        active = data["volActive"].iloc[i]

        if position != 0:
            hold += 1
            tradePnl = position * (price - entryPrice)
            bestPnl = max(bestPnl, tradePnl)

            if tradePnl <= -stopLoss:
                reason = "stopLoss"
                tradeRows.append([entryTime, time, oldPosition, entryPrice, price, tradePnl, reason])
                position = 0

            elif tradePnl >= takeProfit:
                reason = "takeProfit"
                tradeRows.append([entryTime, time, oldPosition, entryPrice, price, tradePnl, reason])
                position = 0

            elif bestPnl - tradePnl >= trailLoss:
                reason = "trailExit"
                tradeRows.append([entryTime, time, oldPosition, entryPrice, price, tradePnl, reason])
                position = 0

            elif hold >= maxHold:
                reason = "timeExit"
                tradeRows.append([entryTime, time, oldPosition, entryPrice, price, tradePnl, reason])
                position = 0

        if position == 0:
            hold = 0
            bestPnl = 0

            if np.isfinite(z) and active:
                if z > entryZ:
                    position = -1
                    entryPrice = price
                    entryTime = time
                    reason = "shortEntry"

                elif z < -entryZ:
                    position = 1
                    entryPrice = price
                    entryTime = time
                    reason = "longEntry"

        positions.append(position)
        trades.append(abs(position - oldPosition))
        reasons.append(reason)

    data["position"] = positions
    data["trade"] = trades
    data["reason"] = reasons

    data["priceMove"] = data["price"].diff()
    data["grossPnl"] = data["position"].shift(1) * data["priceMove"]
    data["tradeCost"] = data["trade"] * cost
    data["netPnl"] = data["grossPnl"] - data["tradeCost"]

    data["equity"] = data["netPnl"].fillna(0).cumsum()
    data["peak"] = data["equity"].cummax()
    data["drawdown"] = data["equity"] - data["peak"]

    tradesData = pd.DataFrame(
        tradeRows,
        columns=["entryTime", "exitTime", "position", "entryPrice", "exitPrice", "tradePnl", "exitReason"]
    )

    return data, tradesData


def showResults(data, tradesData):
    pnl = data["netPnl"].dropna()
    sharpe = pnl.mean() / pnl.std() * np.sqrt(365 * 24 / (fidelity / 60))

    print("\nBacktest Results")
    print("----------------")
    print(f"Total PnL:        {data['equity'].iloc[-1]:.4f}")
    print(f"Sharpe-like:      {sharpe:.3f}")
    print(f"Max Drawdown:     {data['drawdown'].min():.4f}")
    print(f"Completed Trades: {len(tradesData)}")
    print(f"Win Rate:         {(tradesData['tradePnl'] > 0).mean():.3f}")
    print(f"Average Trade:    {tradesData['tradePnl'].mean():.4f}")
    print(f"Best Trade:       {tradesData['tradePnl'].max():.4f}")
    print(f"Worst Trade:      {tradesData['tradePnl'].min():.4f}")

    print("\nTrade Log")
    print(tradesData)


def makePlots(data):
    longs = data[data["reason"] == "longEntry"]
    shorts = data[data["reason"] == "shortEntry"]
    exits = data[(data["reason"] == "timeExit") | (data["reason"] == "stopLoss") | (data["reason"] == "takeProfit") | (data["reason"] == "trailExit")]

    plt.figure(figsize=(12, 5))
    plt.plot(data["time"], data["price"], label="Trump price")
    plt.scatter(longs["time"], longs["price"], marker="^", label="Long")
    plt.scatter(shorts["time"], shorts["price"], marker="v", label="Short")
    plt.scatter(exits["time"], exits["price"], marker="x", label="Exit")
    plt.title("Price with Trade Signals")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/priceTrades.png")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(data["time"], data["equity"])
    plt.title("Equity Curve")
    plt.grid(True)
    plt.savefig("results/equityCurve.png")
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(data["time"], data["drawdown"])
    plt.title("Drawdown")
    plt.grid(True)
    plt.savefig("results/drawdown.png")
    plt.show()


def run():
    data = getHistory(mainToken)
    data = makeSignal(data)
    data, tradesData = runBacktest(data)

    showResults(data, tradesData)
    makePlots(data)


run()