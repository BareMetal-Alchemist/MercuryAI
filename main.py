import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



# Clean and Process Data
df = pandas.read_csv("geochem/geochem.csv", low_memory=False)

hg_col = "HG_ICP10"

df = df.dropna(subset=[hg_col])
df = df[df[hg_col] >= 0]

df["HG_log"] = np.log1p(df[hg_col])

print(df["HG_log"].describe())
print(df["HG_log"].sort_values().tail(10))
print(df[hg_col].describe())
print(df[hg_col].sort_values().tail(10))

df["risk"] = pandas.qcut(
    df["HG_log"],
    q=3,
    labels=["Low", "Medium", "High"]
)

#bins = pandas.qcut(df["HG_log"], q=3, retbins=True)[1]
#print("Log-space bin edges:", bins)

#print("Original Hg bin edges:", np.expm1(bins))



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

df["predicted"] = model.predict(df[["LATITUDE", "LONGITUDE"]])


print(classification_report(y_test, pred))



# Visualize

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = {"Low": "green", "Medium": "orange", "High": "red"}



for r in ["Low", "Medium", "High"]:
    subset = df[df["risk"] == r]
    axes[0].scatter(
        subset["LONGITUDE"],
        subset["LATITUDE"],
        c=colors[r],
        label={
            "Low": "0.001–0.027 ppm",
            "Medium": "0.027–0.055 ppm",
            "High": "> 0.055 ppm"
        }[r],
        s=5,
        alpha=0.6
    )

axes[0].set_title("Ground Truth Mercury Risk")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].legend()



for r in ["Low", "Medium", "High"]:
    subset = df[df["predicted"] == r]
    axes[1].scatter(
        subset["LONGITUDE"],
        subset["LATITUDE"],
        c=colors[r],
        label={
            "Low": "0.001–0.027 ppm",
            "Medium": "0.027–0.055 ppm",
            "High": "> 0.055 ppm"
        }[r],
        s=5,
        alpha=0.6
    )


axes[1].set_title("Model Predicted Risk")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].legend()


plt.tight_layout()
plt.show()


