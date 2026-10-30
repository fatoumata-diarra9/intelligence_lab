import pandas as pd
df = pd.read_csv(
    "data/cv_fr/validated.tsv",
    sep="\t",
    quoting=3,
    usecols=["client_id", "path", "up_votes", "down_votes", "age", "gender"],
)
print("clips :", len(df))
print("locuteurs (client_id) :", df["client_id"].nunique())
par_locuteur = df.groupby("client_id").size()
print(par_locuteur.describe())
for n in [2, 5, 10, 20, 50]:
    print(f"locuteurs avec au moins {n} clips :", (par_locuteur >= n).sum())

print(df["gender"].value_counts(dropna=False))
print(df["age"].value_counts(dropna=False))
