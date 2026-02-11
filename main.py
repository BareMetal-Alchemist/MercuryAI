import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Clean and Process Data
df = pandas.read_csv("geochem/geochem.csv", low_memory=False)

hg_col = "HG_ICP10"

df = df.dropna(subset=[hg_col])
df = df[df[hg_col] >= 0]

df["HG_log"] = np.log1p(df[hg_col])

#print(df["HG_log"].describe())
#print(df[hg_col].sort_values().tail(10))





def classify_hg(x):
    if x < 50:
        return "Low"
    elif x < 200:
        return "Medium"
    else:
        return "High"

df["risk"] = pandas.qcut(
    df["HG_log"],
    q=3,
    labels=["Low", "Medium", "High"]
)

#print(df["risk"].value_counts())

print([col for col in df.columns if "lat" in col.lower() or "lon" in col.lower()])

#print(df["risk"].value_counts())
# print([col for col in df.columns if "hg" in col.lower()])

# Training

features = ["LATITUDE", "LONGITUDE"]

X = df[features]
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))
