import numpy as np
import pandas as pd

MIN_DUREE = 3.0
MIN_CLIPS = 10
MAX_CLIPS = 20
NB_GROS_ECARTES = 10
NB_LOCUTEURS = 2000
GRAINE = 42

df = pd.read_csv(
    "data/cv_fr/validated.tsv",
    sep="\t",
    quoting=3,
    usecols=["client_id", "path", "gender", "age"],
    dtype={"gender": str, "age": str},
)
dur = pd.read_csv(
    "data/cv_fr/clip_durations.tsv",
    sep="\t",
    quoting=3,
    usecols=["clip", "duration[ms]"],
)
dur = dur.rename(columns={"clip": "path", "duration[ms]": "duree_ms"})
dur["duree"] = dur["duree_ms"] / 1000
df = df.merge(dur[["path", "duree"]], on="path", how="left")

gros = df.groupby("client_id").size().nlargest(NB_GROS_ECARTES).index
df = df[~df["client_id"].isin(gros)]
df = df[df["duree"] >= MIN_DUREE]

nb_clips = df.groupby("client_id").size()
eligibles = nb_clips[nb_clips >= MIN_CLIPS].index
print("locuteurs eligibles :", len(eligibles))

rng = np.random.default_rng(GRAINE)
n = min(NB_LOCUTEURS, len(eligibles))
choisis = rng.choice(eligibles, size=n, replace=False)
df = df[df["client_id"].isin(choisis)]

df = df.sample(frac=1, random_state=GRAINE)
df = df.groupby("client_id").head(MAX_CLIPS)

print("locuteurs retenus :", df["client_id"].nunique())
print("clips retenus :", len(df))
print("heures d'audio :", df["duree"].sum() / 3600)
print(df.groupby("client_id").size().describe())

par_loc = df.groupby("client_id")[["gender", "age"]].first()
print(par_loc["gender"].value_counts(dropna=False))
print(par_loc["age"].value_counts(dropna=False))

df.to_csv("data/selection.csv", index=False)
print("ecrit : data/selection.csv")

assert (df.groupby("client_id").size() >= MIN_CLIPS).all()
assert (df.groupby("client_id").size() <= MAX_CLIPS).all()
assert (df["duree"] >= MIN_DUREE).all()
assert not df["client_id"].isin(gros).any()
print("verifications OK")
