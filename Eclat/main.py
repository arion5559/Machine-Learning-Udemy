import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


def inspect(results):
    lhs = [tuple(result[0])[0] for result in results]
    rhs = [tuple(result[0])[1] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
dataset_length = len(dataset.iloc[:, :].values)

transactions = [[str(dataset.values[i, j]) for j in range(len(dataset.iloc[i, :].values) - 1)]
                for i in range(dataset_length)]

rules = apriori(transactions=transactions, min_confidence=0.2, min_support=(3*7/dataset_length), min_lift=3,
                min_length=2, max_length=2)

results = list(rules)

resultsinDataFrame = pd.DataFrame(inspect(results), columns=["Product 1", "Product 2", "Support"])

resultsinDataFrame = resultsinDataFrame.nlargest(n=len(results), columns="Support")

print(resultsinDataFrame)
