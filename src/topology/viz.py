import matplotlib.pyplot as plt
import numpy as np


def plot_point_cloud(cloud: np.ndarray, title: str = "", path: str | None = None) -> None:
    """
    Scatter a 2D or 3D embedded point cloud, connecting consecutive points
    with a faint line so the trajectory (e.g. a loop) is visible, not just
    a blob of dots.
    """
    dimension = cloud.shape[1]
    fig = plt.figure()

    if dimension == 2:
        ax = fig.add_subplot()
        ax.plot(cloud[:, 0], cloud[:, 1], "-", linewidth=0.5, alpha=0.4, color="C0")
        ax.scatter(cloud[:, 0], cloud[:, 1], s=10, c=np.arange(len(cloud)), cmap="viridis")
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t + delay)")
    elif dimension == 3:
        ax = fig.add_subplot(projection="3d")
        ax.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], "-", linewidth=0.5, alpha=0.4, color="C0")
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=10, c=np.arange(len(cloud)), cmap="viridis")
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t + delay)")
        ax.set_zlabel("x(t + 2*delay)")
    else:
        raise ValueError(f"can only plot 2D or 3D clouds directly, got dimension={dimension}")

    ax.set_title(title or f"Takens embedding (dimension={dimension})")
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_forecast_surface(
    gp,
    X_2d: np.ndarray,
    resolution: int = 60,
    title: str = "",
    path: str | None = None,
) -> None:
    """
    Evaluate a GP's posterior mean/std over a grid spanning X_2d's range,
    rendering the continuous forecast surface a Kriging model actually
    defines -- not just its value at discrete test points. `gp` must have
    been fit on 2D input (e.g. the top-2 whitened/PCA components of the
    topological feature space, for visualization -- the full model can use
    more dimensions than this 2D view shows).
    """
    x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
    y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    mean, std = gp.predict(grid, return_std=True)
    mean = mean.reshape(xx.shape)
    std = std.reshape(xx.shape)
    train_mean = gp.predict(X_2d)

    fig = plt.figure(figsize=(11, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(xx, yy, mean, cmap="viridis", alpha=0.85)
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], train_mean, c="red", s=8)
    ax1.set_xlabel("PC1 (whitened)")
    ax1.set_ylabel("PC2 (whitened)")
    ax1.set_zlabel("predicted forward return")
    ax1.set_title("Kriging mean surface")

    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(xx, yy, std, cmap="magma")
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c="white", s=6, edgecolors="black", linewidths=0.3)
    fig.colorbar(contour, ax=ax2, label="predictive std")
    ax2.set_xlabel("PC1 (whitened)")
    ax2.set_ylabel("PC2 (whitened)")
    ax2.set_title("Kriging uncertainty")

    fig.suptitle(title or "Forecast surface over topological feature space")
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    from datetime import date

    from alpaca.data.enums import DataFeed
    from dotenv import load_dotenv

    from src.ingestion.bar.factory import SamplingMethod, make_bar_builder
    from src.ingestion.bar.stream import sample_bars
    from src.ingestion.date_ranges import DateRange
    from src.sources.alpacha_ingestion import AlpacaTradeSource
    from src.topology.embedding import takens_embedding

    load_dotenv()

    symbol = "SPY"
    date_range = DateRange.single_day(date(2024, 1, 2))
    ticks_per_bar = 5

    source = AlpacaTradeSource(feed=DataFeed.IEX)
    trades = source.iter_trades(symbol, date_range.start, date_range.end)

    builder = make_bar_builder(SamplingMethod.TICK, threshold=ticks_per_bar)
    bars = list(sample_bars(trades, builder))
    print(f"{symbol} {date_range.start.date()}: {len(bars)} tick bars (ticks/bar={ticks_per_bar})")

    closes = [bar.close for bar in bars]
    cloud = takens_embedding(closes, dimension=2, delay=3)

    plot_point_cloud(cloud, title=f"{symbol} {date_range.start.date()} tick-bar Takens embedding")
