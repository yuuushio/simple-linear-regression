import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation


def extract_variables():
    df = pd.read_csv("Food_Delivery_Times.csv")
    df.columns = [x.lower() for x in df.columns]

    # Array of feature vectors
    x = []

    # List in case we want to add more features
    features = ["distance_km"]

    df = df[features + ["delivery_time_min"]].dropna()

    for f in features:
        x.append(df[f].to_numpy())

    y_vector = df["delivery_time_min"].to_numpy()

    return x, y_vector, df.shape[0]


def linear_regression():
    x_list, yv, n = extract_variables()
    for i, feature_v in enumerate(x_list):
        print(i, feature_v[0], yv[0])


def main():
    linear_regression()


if __name__ == "__main__":
    main()
