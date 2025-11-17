import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

weather_fg = fs.get_feature_group(name="weather", version=2)

print("👉 READING weather v2 ...")
df = weather_fg.read()

print(df.head(20))
print("\n👉 Date range:", df["date"].min(), "→", df["date"].max())
print("\n👉 Unique dates in last 20 rows:")
print(df["date"].sort_values().tail(20))
