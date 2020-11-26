import os
import sys
sys.path.insert(0, "../lib/")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

import bbknn


SAMPLES = pd.read_csv("samples.csv")


def rename_genes(names):
    names = names.str.replace("^GRCh38_+", "")
    names = names.str.replace("^SARS-CoV-2i_", "SARS-CoV-2-")
    names = names.str.replace("SARS-CoV-2-antisense", "Antisense")
    return names


def load_ds(path, sample, cells=None):
    ds = sc.read_10x_h5(path)
    ds.var_names = rename_genes(ds.var_names)
    ds.var_names_make_unique(join=".")
    ds.obs_names = ds.obs_names.str.replace("-\d$", "")
    if cells is not None:
        cells = pd.read_csv(cells)
        cells.x = cells.x.str.replace("-\d$", "")
        ds = ds[cells.x, :].copy()
    ds.obs_names = "0_" + ds.obs_names
    sc.pp.filter_cells(ds, min_genes=200)
    sc.pp.filter_genes(ds, min_cells=3)

    meta = SAMPLES.loc[SAMPLES.Sample_ID == sample, :]
    for c in meta.columns:
        if len(meta[c].values):
            ds.obs[c] = meta[c].values[0]
    return ds


def prepare(path, cells=None, out_file=None, sample=None):
    ds = load_ds(path, sample, cells=cells)
    ds.write(out_file)


def get_markers(anndata, groupby):
    def calc_pct_1(x):
        cells = anndata.obs[groupby] == x.cluster
        gene = anndata.var_names == x.gene
        return (anndata.X[cells, gene] > 0).sum() / cells.sum()

    def calc_pct_2(x):
        cells = anndata.obs[groupby] != x.cluster
        gene = anndata.var_names == x.gene
        return (anndata.X[cells, gene] > 0).sum() / cells.sum()

    markers = pd.concat([
        pd.DataFrame(anndata.uns["rank_genes_groups"]["names"]).melt(),
        pd.DataFrame(anndata.uns["rank_genes_groups"]["pvals_adj"]).melt(),
        pd.DataFrame(anndata.uns["rank_genes_groups"]["logfoldchanges"]).melt()
    ], axis=1)
    markers.columns = ("cluster", "gene", "cluster2", "p_val_adj", "cluster3", "avg_logFC")
    markers = markers.loc[:, ["cluster", "gene", "avg_logFC", "p_val_adj"]]
    markers = markers.loc[markers.avg_logFC > 0, ]
    markers = markers.loc[markers.p_val_adj < 0.05, ]
    markers["pct.1"] = markers.apply(calc_pct_1, axis=1)
    markers["pct.2"] = markers.apply(calc_pct_2, axis=1)
    markers["p_val"] = markers.p_val_adj
    markers = markers.loc[:, ["p_val", "avg_logFC", "pct.1", "pct.2", "p_val_adj", "cluster", "gene"]]
    return markers


def integrate(ds_paths, h5ad_path, out_path, noribo):
    datasets = list(map(sc.read_h5ad, ds_paths))
    ds = datasets[0].concatenate(datasets[1:], join="outer")
    ds.var["mito"] = ds.var_names.str.startswith("MT-")
    ds.var["ribo"] = ds.var_names.str.match("^RP(L|S)")
    sc.pp.calculate_qc_metrics(
        ds,
        qc_vars=["mito", "ribo"],
        percent_top=None,
        log1p=False,
        inplace=True
    )
    if noribo:
        ds = ds[:, ~ds.var["ribo"]]

    sc.pp.normalize_total(ds, target_sum=1e4)
    sc.pp.log1p(ds)
    sc.pp.highly_variable_genes(ds, n_top_genes=3000, batch_key="orig.ident")
    ds.raw = ds

    sc.pp.scale(ds)
    sc.tl.pca(ds, svd_solver="arpack")
    n_neighbors = int(50 / len(datasets))
    bbknn.bbknn(ds, neighbors_within_batch=n_neighbors, n_pcs=30)
    sc.tl.leiden(ds, resolution=0.75)
    sc.tl.umap(ds)
    sc.tl.rank_genes_groups(ds, "leiden", method="wilcoxon", n_genes=0)

    ds.write(h5ad_path)

    os.makedirs(out_path)
    get_markers(ds, "leiden").to_csv(out_path + "/00markers.csv")

    mpl.rcParams["figure.figsize"] = (12, 8)
    ax = sc.pl.umap(ds, color="leiden", size=10, legend_loc="on data", show=False)
    ax.get_figure().savefig(out_path + "/01clusters.pdf")

    ax = sc.pl.umap(ds, color="orig.ident", size=10, show=False)
    ax.get_figure().savefig(out_path + "/02samples.pdf")

    bottom = np.zeros(len(ds.obs["leiden"].unique()))
    fig, ax = plt.subplots()
    for s in ds.obs["orig.ident"].unique():
        cnt = ds.obs["leiden"][ds.obs["orig.ident"] == s].value_counts().sort_index()
        ax.bar(cnt.index, cnt, bottom=bottom, label=s)
        bottom += cnt
    ax.legend()
    fig.suptitle("BBKNN Clusters by sample")
    fig.savefig(out_path + "/03composition.pdf")

    ax = sc.pl.stacked_violin(
        ds,
        ["MRC1", "FABP4", "CCL2", "SPP1", "CCL18", "CXCL10", "G0S2", "MKI67", "CD3E", "CD4", "CD8A", "JCHAIN", "FOXJ1"],
        groupby="leiden",
        rotation=90,
        figsize=(10, 10),
        show=False
    )
    ax[0].get_figure().savefig(out_path + "/04markers.pdf")

    ax = sc.pl.umap(ds, color="total_counts", size=10, show=False)
    ax.get_figure().savefig(out_path + "/05nUMI.pdf")

    ax = sc.pl.umap(ds, color="pct_counts_mito", size=10, show=False)
    ax.get_figure().savefig(out_path + "/06mito.pdf")

    ax = sc.pl.umap(ds, color="pct_counts_ribo", size=10, show=False)
    ax.get_figure().savefig(out_path + "/06ribo.pdf")
