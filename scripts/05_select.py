import numpy as np
import pandas as pd

MIN_DUREE = 3.0
MIN_CLIPS = 10
MAX_CLIPS = 20
NB_GROS_ECARTES = 10
NB_LOCUTEURS = 2000
NB_FEMMES = 400
NB_HOMMES = 400
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
df = df[df["client_id"].isin(eligibles)]
print("locuteurs eligibles :", len(eligibles))

nb_genres = df.groupby("client_id")["gender"].nunique()
print("locuteurs avec plusieurs genres declares :", (nb_genres > 1).sum())

genre = df.groupby("client_id")["gender"].first()
age = df.groupby("client_id")["age"].first()
df["genre_loc"] = df["client_id"].map(genre)
df["age_loc"] = df["client_id"].map(age)

femmes = genre[genre == "female_feminine"].index
hommes = genre[genre == "male_masculine"].index
autres = genre[~genre.isin(["female_feminine", "male_masculine"])].index

rng = np.random.default_rng(GRAINE)
n_f = min(NB_FEMMES, len(femmes))
n_h = min(NB_HOMMES, len(hommes))
n_a = min(max(NB_LOCUTEURS - n_f - n_h, 0), len(autres))
choisies = rng.choice(femmes, size=n_f, replace=False)
choisis_h = rng.choice(hommes, size=n_h, replace=False)
choisis_autres = rng.choice(autres, size=n_a, replace=False)
choisis = np.concatenate([choisies, choisis_h, choisis_autres])
print("tires : femmes", n_f, "/ hommes", n_h, "/ autres", n_a)
if len(choisis) < NB_LOCUTEURS:
    print("ATTENTION : seulement", len(choisis), "locuteurs au lieu de", NB_LOCUTEURS)
df = df[df["client_id"].isin(choisis)]

df = df.sample(frac=1, random_state=GRAINE)
df = df.groupby("client_id").head(MAX_CLIPS)

print("locuteurs retenus :", df["client_id"].nunique())
print("clips retenus :", len(df))
print("heures d'audio :", df["duree"].sum() / 3600)
print(df.groupby("client_id").size().describe())

par_loc = df.groupby("client_id")[["genre_loc", "age_loc"]].first()
print(par_loc["genre_loc"].value_counts(dropna=False))
print(par_loc["age_loc"].value_counts(dropna=False))

assert (df.groupby("client_id").size() >= MIN_CLIPS).all()
assert (df.groupby("client_id").size() <= MAX_CLIPS).all()
assert (df["duree"] >= MIN_DUREE).all()
assert not df["client_id"].isin(gros).any()
assert df["client_id"].nunique() == len(choisis)
assert (par_loc["genre_loc"] == "female_feminine").sum() == n_f
assert (par_loc["genre_loc"] == "male_masculine").sum() == n_h
print("verifications OK")

df.to_csv("data/selection.csv", index=False)
print("ecrit : data/selection.csv")
