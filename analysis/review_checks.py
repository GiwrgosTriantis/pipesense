"""Reproduces the ML numbers quoted in DECISIONS.md section 1.5.

Usage (after one run of the original starter-kit pipeline):
    python review_checks.py <WORK_DIR with features.gpkg + predictions.csv> <data/pipes.csv>

The failure-path findings F1/F2 were reproduced by hand:
  F1: run run_all.sh once, then drop the `material` column from pipes.csv and run it again
      -> stage 1 fails, stages 2-3 reuse the stale features.gpkg, status SUCCESS, exit code 0.
  F2: put an unreadable features.gpkg in WORK_DIR and run train_model.py
      -> NameError: name 'elapsed' is not defined, no status.json written.
"""
import sys, numpy as np, pandas as pd, geopandas as gpd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
work = sys.argv[1]; raw = sys.argv[2]
g = gpd.read_file(f"{work}/features.gpkg")
pred = pd.read_csv(f"{work}/predictions.csv").merge(g[["pipe_id","has_leak"]], on="pipe_id")
print("predictions.csv in-sample AP/AUC vs labels:", round(ap(pred.has_leak, pred.leak_risk),3), round(auc(pred.has_leak, pred.leak_risk),3))
print("top-25 already-leaking:", pred.head(25).has_leak.sum(), "/25 ; top-100:", pred.head(100).has_leak.sum())
NUM=["age_years","diameter_mm","log_length","num_breaks_5yr","dist_to_center_m","neighbours_500m"]; CAT=["material","soil_type","pressure_zone"]
def X(cols): return pd.get_dummies(g[[c for c in cols if c in NUM]+[c for c in cols if c in CAT]], columns=[c for c in cols if c in CAT])
y=g.has_leak.values
def cv(cols, splitter, groups=None, name=""):
    x=X(cols); s=np.zeros(len(y))
    for tr,te in splitter.split(x,y,groups):
        m=GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=7).fit(x.iloc[tr],y[tr]); s[te]=m.predict_proba(x.iloc[te])[:,1]
    print(f"{name:45s} AP={ap(y,s):.3f} AUC={auc(y,s):.3f}")
ALL=NUM+CAT
cv(ALL, StratifiedKFold(5,shuffle=True,random_state=7), name="random 5-fold, all features")
# spatial blocks ~1km grid
cell=(g.geometry.x//1500).astype(int).astype(str)+"_"+(g.geometry.y//1500).astype(int).astype(str)
print("n spatial blocks", cell.nunique())
cv(ALL, GroupKFold(5), cell, name="spatial-block 5-fold, all features")
cv([c for c in ALL if c!="num_breaks_5yr"], StratifiedKFold(5,shuffle=True,random_state=7), name="random, without num_breaks_5yr")
cv(["age_years"], StratifiedKFold(5,shuffle=True,random_state=7), name="age only")
cv(["age_years","material"], StratifiedKFold(5,shuffle=True,random_state=7), name="age+material")
cv([c for c in ALL if c not in("dist_to_center_m","neighbours_500m")], StratifiedKFold(5,shuffle=True,random_state=7), name="random, no geo features")
print("baseline AP (prevalence):", round(y.mean(),3))
# simple heuristic baseline: rank by age
print("heuristic age rank AP:", round(ap(y,g.age_years),3), " breaks rank AP:", round(ap(y,g.num_breaks_5yr+g.age_years/1000),3))
# missing install year signal
r=pd.read_csv(raw).drop_duplicates("pipe_id"); print("leak rate | install_year missing:", r[r.install_year.isna()].has_leak.mean().round(3), "present:", r[r.install_year.notna()].has_leak.mean().round(3))
print("missing year by material:\n", r.assign(m=r.install_year.isna()).groupby(r.material.str.lower())['m'].mean().round(2).to_string())
# spatial clustering of leaks: leak rate of neighbours of leaking pipes
from sklearn.neighbors import NearestNeighbors
xy=np.c_[g.geometry.x,g.geometry.y]; nn=NearestNeighbors(n_neighbors=6).fit(xy); _,idx=nn.kneighbors(xy)
nb=y[idx[:,1:]].mean(1); print("neighbour leak rate | leak:", nb[y==1].mean().round(3), " | no leak:", nb[y==0].mean().round(3))
print("neighbours_500m range", g.neighbours_500m.min(), g.neighbours_500m.max(), "dist range", g.dist_to_center_m.min(), g.dist_to_center_m.max())
print("age NaN after prep:", g.age_years.isna().sum(), "diam NaN:", g.diameter_mm.isna().sum())
