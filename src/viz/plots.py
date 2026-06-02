import matplotlib.pyplot as plt


def plot_target_distribution(df, target, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    df[target].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_xlabel(target)
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of {target}")
    return ax


def plot_missingness(df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    pct = df.isna().mean().mul(100).sort_values()
    pct = pct[pct > 0]
    if pct.empty:
        ax.text(0.5, 0.5, "no missing values", ha="center", va="center")
    else:
        pct.plot.barh(ax=ax)
        ax.set_xlabel("% missing")
    ax.set_title("Missingness by column")
    return ax
