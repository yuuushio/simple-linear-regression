import matplotlib

matplotlib.use("Qt5Agg")
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


def calc_gradients(w, b, xv, yv, n):
    tmp_w, tmp_b = 0, 0
    for i in range(n):
        tmp_w += -2 * xv[i] * (yv[i] - (w * xv[i] + b))
        tmp_b += -2 * (yv[i] - (w * xv[i] + b))

    return tmp_w / n, tmp_b / n


def draw_viz(xv, yv, line_eq, w_list, b_list, epoch, animation=False):
    print(matplotlib.get_backend())
    fig, ax = plt.subplots()
    ax.plot(xv, yv, "o")
    (line_plot,) = ax.plot(xv, line_eq, "-")

    ax.set_title("[Food] Delivery Time Prediction")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Time Taken to Deliver (Minutes)")

    if animation:

        def update(frame):
            line_plot.set_ydata(w_list[frame] * xv + b_list[frame])
            return (line_plot,)

        ani = FuncAnimation(
            fig, update, frames=range(0, epoch, 2), interval=10, blit=True
        )
        ani.save("lr.gif", writer="ffmpeg")

    plt.show()


def linear_regression(rate=0.001, epoch=100, animation=False):
    xv_list, yv, n = extract_variables()
    for i, feature_v in enumerate(xv_list):
        print(i, feature_v[0], yv[0])

    w, b = 0, 0
    w_list = []
    b_list = []

    for _ in range(epoch):
        tmp_w, tmp_b = calc_gradients(w, b, xv_list[0], yv, n)
        w += (-1) * rate * tmp_w
        b += (-1) * rate * tmp_b
        w_list.append(w)
        b_list.append(b)

    best_fit_line = w * xv_list[0] + b
    draw_viz(xv_list[0], yv, best_fit_line, w_list, b_list, epoch, animation)


def main():
    linear_regression(0.001, 1000, False)


if __name__ == "__main__":
    main()
