"""Customer segmentation of a UK online retailer (UCI Online Retail, 2010-2011).

Pipeline: clean transactions -> build RFM + behaviour features per customer ->
log-transform & scale -> choose k for K-Means with silhouette / elbow ->
profile and name segments -> recommend an action per segment.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

SEED = 42
ROOT = Path(__file__).parent
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
DATA = ROOT / "data" / "online_retail.csv"
URL = ("https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/"
       "master/data/retail-data/all/online-retail-dataset.csv")


def load():
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        pd.read_csv(URL).to_csv(DATA, index=False)
    df = pd.read_csv(DATA, dtype={"InvoiceNo": str, "StockCode": str})
    n0 = len(df)
    df = df[df["CustomerID"].notna()]
    df = df[~df["InvoiceNo"].str.startswith("C")]          # cancellations
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df[~df["StockCode"].isin(["POST", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "CRUK", "D", "PADS"])]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype(int)
    print(f"rows: {n0:,} raw -> {len(df):,} clean; customers: {df.CustomerID.nunique():,}")
    return df


def features(df):
    snap = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    g = df.groupby("CustomerID")
    f = pd.DataFrame({
        "recency_days": (snap - g["InvoiceDate"].max()).dt.days,
        "frequency": g["InvoiceNo"].nunique(),
        "monetary": g["Revenue"].sum(),
        "tenure_days": (snap - g["InvoiceDate"].min()).dt.days,
        "distinct_products": g["StockCode"].nunique(),
    })
    f["avg_order_value"] = f["monetary"] / f["frequency"]
    # remove extreme wholesale outliers from clustering (kept & reported separately)
    cap = f["monetary"].quantile(0.995)
    f["outlier"] = f["monetary"] > cap
    return f


def main():
    df = load()
    f = features(df)
    core = f[~f["outlier"]].copy()
    cols = ["recency_days", "frequency", "monetary", "avg_order_value", "distinct_products"]
    X = StandardScaler().fit_transform(np.log1p(core[cols]))

    ks, inertia, sil = range(2, 9), [], []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=SEED).fit(X)
        inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_, sample_size=3000, random_state=SEED))
    pd.DataFrame({"k": list(ks), "inertia": np.round(inertia, 1), "silhouette": np.round(sil, 4)}).to_csv(OUT / "k_selection.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
    ax[1].plot(list(ks), sil, "o-", color="tab:orange"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
    K = 4; ax[0].axvline(K, ls="--", c="grey"); ax[1].axvline(K, ls="--", c="grey")
    plt.tight_layout(); plt.savefig(OUT / "k_selection.png", dpi=100); plt.close()

    km = KMeans(K, n_init=50, random_state=SEED).fit(X)
    core["cluster"] = km.labels_
    prof = core.groupby("cluster").agg(customers=("frequency", "size"), recency_days=("recency_days", "median"),
                                       orders=("frequency", "median"), revenue=("monetary", "median"),
                                       aov=("avg_order_value", "median"), products=("distinct_products", "median"),
                                       total_revenue=("monetary", "sum"))
    # name clusters by value score (low recency, high frequency/monetary = best)
    score = (-prof["recency_days"].rank() + prof["orders"].rank() + prof["revenue"].rank())
    order = score.sort_values(ascending=False).index
    names = ["Champions", "Loyal Regulars", "Needs Attention", "Lapsed / Lost"]
    actions = {"Champions": "VIP perks, early access, referral asks - protect, don't discount",
               "Loyal Regulars": "Cross-sell bundles and loyalty points to move them up a tier",
               "Needs Attention": "Personalised win-back email within 30 days, time-limited offer",
               "Lapsed / Lost": "Low-cost reactivation campaign only; suppress from paid channels"}
    nm = dict(zip(order, names)); core["segment"] = core["cluster"].map(nm)
    prof.index = prof.index.map(nm); prof = prof.loc[names]
    prof["share_customers"] = (prof["customers"] / prof["customers"].sum()).round(3)
    prof["share_revenue"] = (prof["total_revenue"] / prof["total_revenue"].sum()).round(3)
    prof["action"] = [actions[n] for n in prof.index]
    prof[["recency_days","orders","revenue","aov","products","total_revenue"]] = prof[["recency_days","orders","revenue","aov","products","total_revenue"]].round(1); prof.to_csv(OUT / "segment_profiles.csv"); print(prof.drop(columns="action").to_string())
    print(f"silhouette k={K}: {sil[K-2]:.3f}; outliers excluded: {f['outlier'].sum()}")
    core[["segment"] + cols].to_csv(OUT / "customer_segments.csv")

    colors = {"Champions": "#1f3a5f", "Loyal Regulars": "#4c8bc9", "Needs Attention": "#e0a030", "Lapsed / Lost": "#b0b0b0"}
    # customers vs revenue share
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w/2, prof["share_customers"] * 100, w, label="% of customers", color="#b9c7d8")
    ax.bar(x + w/2, prof["share_revenue"] * 100, w, label="% of revenue", color=[colors[n] for n in names])
    for i, n in enumerate(names):
        ax.text(i + w/2, prof.loc[n, "share_revenue"] * 100 + 1, f"{prof.loc[n,'share_revenue']:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x, names); ax.set_ylabel("%"); ax.set_title("Who drives revenue? Customer share vs revenue share")
    ax.legend(); plt.tight_layout(); plt.savefig(OUT / "segment_revenue_share.png", dpi=100); plt.close()

    # PCA scatter
    pcs = PCA(2, random_state=SEED).fit_transform(X)
    plt.figure(figsize=(6.5, 5))
    for n in names:
        m = (core["segment"] == n).values
        plt.scatter(pcs[m, 0], pcs[m, 1], s=6, alpha=0.6, label=n, c=colors[n])
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Customer segments (PCA of scaled RFM features)")
    plt.legend(markerscale=3); plt.tight_layout(); plt.savefig(OUT / "segments_pca.png", dpi=100); plt.close()


if __name__ == "__main__":
    main()
