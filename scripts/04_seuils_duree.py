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
df = df.merge(dur[["path", "duree"]], on="path", how="left")

for seuil in [1, 2, 3, 4]:
    gardes = df[df["duree"] >= seuil]
    par_loc = gardes.groupby("client_id").size()
    part = 100 * len(gardes) / len(df)
    nb_loc = (par_loc >= 10).sum()
    print(f"seuil {seuil} s : {len(gardes)} clips gardes ({part:.1f} %), {nb_loc} locuteurs avec au moins 10 clips")
