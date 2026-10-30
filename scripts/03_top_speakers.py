import pandas as pd

df = pd.read_csv(
    "data/cv_fr/validated.tsv",
    sep="\t",
    quoting=3,
    usecols=["client_id", "path"],
)

dur = pd.read_csv(
    "data/cv_fr/clip_durations.tsv",
    sep="\t",
    quoting=3,
    usecols=["clip", "duration[ms]"],
)
dur = dur.rename(columns={"clip": "path", "duration[ms]": "duree_ms"})
dur["duree"] = dur["duree_ms"] / 1000

print("clips dans validated :", len(df))
print("lignes dans clip_durations :", len(dur))
print("doublons dans clip_durations :", dur["path"].duplicated().sum())

df = df.merge(dur[["path", "duree"]], on="path", how="left")
print("clips apres fusion :", len(df))
print("clips sans duree :", df["duree"].isna().sum())

print(df["duree"].describe())
print("heures totales :", df["duree"].sum() / 3600)

stats = df.groupby("client_id").agg(
    clips=("path", "size"),
    secondes=("duree", "sum"),
    duree_moy=("duree", "mean"),
)
stats["heures"] = stats["secondes"] / 3600
stats["part_corpus_%"] = 100 * stats["clips"] / len(df)
stats = stats.sort_values("clips", ascending=False)

top = stats.head(10).copy()
top.index = top.index.str[:8]
print(top[["clips", "heures", "duree_moy", "part_corpus_%"]])
print("part des 10 plus gros :", top["part_corpus_%"].sum(), "%")
