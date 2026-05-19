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
        nextTime = now + pd.Timedelta(days=14)
        if nextTime > end:
            nextTime = end
        params = {}
        params["market"] = token
        params["startTs"] = int(now.timestamp())
        params["endTs"] = int(nextTime.timestamp())
        params["fidelity"] = fidelity
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        history = data.get("history", [])
        for h in history:
            rows.append(h)
        now = nextTime
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["price"] = df["p"].astype(float)

    df = df[["time", "price"]]
    df = df.drop_duplicates(subset="time")
    df = df.sort_values("time")
    df = df.reset_index(drop=True)

    return df

def makeSignal(data):
    tiny = 0.00001
    data["price"] = data["price"].clip(tiny, 1 - tiny)
    data["logit"] = np.log(data["price"] / (1 - data["price"]))
    data["move"] = data["logit"].diff()
    data["meanMove"] = data["move"].rolling(window).mean()
    data["vol"] = data["move"].rolling(window).std()
    data["level"] = data["logit"].rolling(window).mean()
    kappa = []
    for i in range(len(data)):
        kappa.append(np.nan)
    x = data["logit"].values
    dx = data["move"].values
    for i in range(window, len(data)):
        oldX = x[i - window : i - 1]
        oldDx = dx[i - window + 1 : i]
        good = np.isfinite(oldX) & np.isfinite(oldDx)
        if np.sum(good) < window // 2:
            continue
        design = np.column_stack((np.ones(np.sum(good)), oldX[good]))
        beta = np.linalg.lstsq(design, oldDx[good], rcond=None)[0]
        value = -beta[1]
        if value < 0:
            value = 0
        kappa[i] = value
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
            hold = hold + 1
            tradePnl = position * (price - entryPrice)
            if tradePnl > bestPnl:
                bestPnl = tradePnl
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

    tradesData = pd.DataFrame(tradeRows ,columns=["entryTime", "exitTime", "position", "entryPrice", "exitPrice", "tradePnl", "exitReason"])
    return data, tradesData


def showResults(data, tradesData):
    pnl = data["netPnl"].dropna()

    sharpe = pnl.mean()/pnl.std()*np.sqrt(365 * 24 / (fidelity / 60))

    print("Total PnL:", data["equity"].iloc[-1])
    print("Sharpe-like:", sharpe)
    print("Max Drawdown:", data["drawdown"].min())
    print("Completed Trades:", len(tradesData))

    if len(tradesData) > 0:
        print("Win %:", (tradesData["tradePnl"] > 0).mean())
        print("Average Trade:", tradesData["tradePnl"].mean())
        print("Best Trade:", tradesData["tradePnl"].max())
        print("Worst Trade:", tradesData["tradePnl"].min())


def run():
    data = getHistory(mainToken)
    data = makeSignal(data)
    data, tradesData = runBacktest(data)
    showResults(data, tradesData)


run()