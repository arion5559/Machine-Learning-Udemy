import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


def inspect(results):
    lhs = [tuple(result[0])[0] for result in results]
    mhs = [tuple(result[0])[1] for result in results]
    rhs = [tuple(result[0])[2] for result in results if len(tuple(result[0])) > 2]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, mhs, rhs, supports, confidences, lifts))


dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
dataset_length = len(dataset.iloc[:, :].values)

transactions = [[str(dataset.values[i, j]) for j in range(len(dataset.iloc[i, :].values) - 1)]
                for i in range(dataset_length)]

rules = apriori(transactions=transactions, min_support=(3*7/dataset_length),
                min_confidence=0.2, min_lift=4, min_lenth=2, max_length=3)

results = list(rules)
for result in results:
    print(result)
    print(tuple(result[0]))

resultsinDataFrame = pd.DataFrame(inspect(results), columns=["Left Hand Side", "Middle Hand Side", "Right Hand Side",
                                                             "Support", "Confidence", "Lift"])

resultsinDataFrame = resultsinDataFrame.nlargest(n=len(results), columns="Lift")

resultsinDataFrame.to_csv("Results.csv", na_rep="nan", index=False)

print(resultsinDataFrame)
