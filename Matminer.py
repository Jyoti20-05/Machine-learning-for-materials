import requests
import pandas as pd
import time
import re
# === Constants ===
base_url = "http://oqmd.org/oqmdapi/formationenergy"
limit = 100
max_entries = 1000
fields = "name,entry_id,composition,spacegroup,ntypes,natoms,volume,delta_e,band_gap,stability"

cation_elements = {"Zn", "In", "Ga", "Sn"}

def fetch_and_filter(anion, label):
print(f"\n🔍 Fetching binary '{label}' compounds...")
all_data = []

element_filter = f"element_set=(Zn-In-Ga-Sn),{anion}"

for offset in range(0, max_entries, limit):
params = {
"fields": fields,
"filter": element_filter,
"limit": limit,
"offset": offset,
"noduplicate": "True",
"desc": "False"
}

response = requests.get(base_url, params=params)
if response.status_code != 200:
print(f"❌ Error {response.status_code} at offset {offset}")
return pd.DataFrame() # Safely return empty DataFrame

data = response.json().get("data", [])
if not data:
break

all_data.extend(data)
print(f"✅ Fetched {len(data)} entries (offset {offset})")

if len(data) < limit:
break

time.sleep(0.3)

if not all_data:
print("⚠️ No data fetched.")
return pd.DataFrame()

df = pd.DataFrame(all_data)

def is_valid_binary(comp_str):
elements = re.findall(r'[A-Z][a-z]?', comp_str)
unique_elements = set(elements)
return (
len(unique_elements) == 2 and
anion in unique_elements and
len(cation_elements & unique_elements) == 1
)

df_filtered = df[df["composition"].apply(is_valid_binary)].reset_index(drop=True)
print(f"🧼 Filtered to {len(df_filtered)} valid binary {label} compounds.")
return df_filtered

df_binary_oxides = fetch_and_filter("O", "oxides")
df_binary_nitrides = fetch_and_filter("N", "nitrides")

# Save if not empty
if not df_binary_oxides.empty:
df_binary_oxides.to_csv("oqmd_binary_oxides.csv", index=False)
if not df_binary_nitrides.empty:
df_binary_nitrides.to_csv("oqmd_binary_nitrides.csv", index=False)

print("\n🧪 Oxides:")
print(df_binary_oxides.head())

print("\n🧪 Nitrides:")
print(df_binary_nitrides.head())

pip install pandas matplotlib seaborn tqdm requests pymatgen matminer
pip install pandas requests

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element

# === Load datasets ===
oxides = pd.read_csv("oqmd_binary_oxides.csv")
nitrides = pd.read_csv("oqmd_binary_nitrides.csv")

# === Define common functions ===
cation_elements = ['Zn', 'Ga', 'In', 'Sn']
anion_map = {'oxide': 'O', 'nitride': 'N'}

def get_cation(comp):
elements = re.findall(r'[A-Z][a-z]?', comp)
for el in elements:
if el in cation_elements:
return el
return 'Unknown'

def parse_composition_ratio(comp, cation, anion):
el_amounts = re.findall(r'([A-Z][a-z]*)([0-9\.]*)', comp)
count_dict = {el: float(amt) if amt else 1.0 for el, amt in el_amounts}
return (
count_dict.get(anion, 0) / count_dict.get(cation, 1),
count_dict.get(cation, 0),
count_dict.get(anion, 0)
)

def get_ionic_radii_diff(comp_str):
comp = Composition(comp_str)
try:
oxi_guess = comp.oxi_state_guesses()[0]
comp = Composition(oxi_guess)
except:
return None
radii = []
for el in comp.elements:
ox = comp.get_el_amt_dict().get(str(el), 1)
try:
r = el.ionic_radii[ox]
radii.append(r)
except:
continue
if len(radii) == 2:
return abs(radii[0] - radii[1])
return None

def analyze_dataset(df, system_label):
anion = anion_map[system_label]
df['cation'] = df['composition'].apply(get_cation)
df[['anion_to_cation_ratio', 'cation_count', 'anion_count']] = df.apply(
lambda row: pd.Series(parse_composition_ratio(row['composition'], row['cation'], anion)),
axis=1
)
df['ionic_density'] = df['natoms'] / df['volume']
df['approx_coordination'] = df['anion_count'] / df['cation_count']
df['composition_obj'] = df['composition'].apply(Composition)
df['ionic_radius_diff'] = df['composition'].apply(get_ionic_radii_diff)

# === Magpie Features ===
magpie_featurizer = ElementProperty.from_preset("magpie")
df = magpie_featurizer.featurize_dataframe(df, 'composition_obj')

# === Plots ===
sns.set(style="whitegrid", font_scale=1.1)

# 1. Correlation Matrix
plt.figure(figsize=(10, 6))
# Clean version: select only key features
core_features = ['delta_e', 'band_gap', 'volume', 'natoms', 'ntypes',
'ionic_density', 'approx_coordination', 'ionic_radius_diff']


plt.figure(figsize=(10, 6))
sns.heatmap(df[core_features].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title(f"[{system_label.upper()}] Correlation Matrix (Key Features)")
plt.tight_layout()
plt.show()
# 2. Formation Energy by Cation
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="cation", y="delta_e")
plt.title(f"[{system_label.upper()}] Formation Energy by Cation")
plt.tight_layout()
plt.show()

# 3. Band Gap by Cation
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="cation", y="band_gap")
plt.title(f"[{system_label.upper()}] Band Gap Distribution")
plt.tight_layout()
plt.show()

# 4. Stability vs Band Gap
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="stability", y="band_gap", hue="cation", s=60)
plt.title(f"[{system_label.upper()}] Stability vs Band Gap")
plt.tight_layout()
plt.show()

# 5. Ionic Density Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="ionic_density", element="step", stat="density", common_norm=False)
plt.title(f"[{system_label.upper()}] Ionic Density Distribution")
plt.tight_layout()
plt.show()

# 6. Approx Coordination Number
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="cation", y="approx_coordination")
plt.title(f"[{system_label.upper()}] Anion:Cation Ratio")
plt.tight_layout()
plt.show()

# 7. Space Group Frequencies
top_spacegroups = df['spacegroup'].value_counts().head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_spacegroups.values, y=top_spacegroups.index)
plt.title(f"[{system_label.upper()}] Top 10 Space Groups")
plt.xlabel("Compound Count")
plt.tight_layout()
plt.show()

# 8. Compound Count by Cation
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="cation")
plt.title(f"[{system_label.upper()}] Compound Count by Cation")
plt.tight_layout()
plt.show()

return df

# === Apply analysis ===
oxides_processed = analyze_dataset(oxides.copy(), system_label="oxide")
nitrides_processed = analyze_dataset(nitrides.copy(), system_label="nitride")
import os
import requests
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN
from matminer.featurizers.structure import SiteStatsFingerprint, GlobalSymmetryFeatures
from matminer.featurizers.site import CrystalNNFingerprint
from pymatgen.core.structure import Structure
import tempfile
from tqdm import tqdm

# === Featurizers ===
structure_fp = SiteStatsFingerprint(CrystalNNFingerprint.from_preset("ops"))
symmetry_fp = GlobalSymmetryFeatures()

# === CIF fetch function ===
def fetch_cif(entry_id):
url = f"http://oqmd.org/oqmdapi/structure/{entry_id}/cif"
response = requests.get(url)
if response.status_code == 200:
return response.text
return None

# === Featurization function ===
def featurize_from_cif(cif_str):
with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as temp_cif:
temp_cif.write(cif_str.encode())
temp_cif_path = temp_cif.name
try:
parser = CifParser(temp_cif_path)
structure = parser.get_structures()[0]
site_features = structure_fp.featurize(structure)
symm_features = symmetry_fp.featurize(structure)
return site_features + symm_features
except:
return None

def process_structures(df, label="oxides"):
all_features = []
failed_ids = []
for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {label}"):
entry_id = row['entry_id']
cif = fetch_cif(entry_id)
if cif:
features = featurize_from_cif(cif)
if features:
all_features.append(features)
else:
failed_ids.append(entry_id)
all_features.append([None] * (len(structure_fp.feature_labels()) + len(symmetry_fp.feature_labels())))
else:
failed_ids.append(entry_id)
all_features.append([None] * (len(structure_fp.feature_labels()) + len(symmetry_fp.feature_labels())))

feature_labels = structure_fp.feature_labels() + symmetry_fp.feature_labels()
feature_df = pd.DataFrame(all_features, columns=feature_labels)
result_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
return result_df, failed_ids

oxides_structured, oxide_failures = process_structures(oxides_processed, label="oxides")
nitrides_structured, nitride_failures = process_structures(nitrides_processed, label="nitrides")

# === Save results ===
oxides_structured.to_csv("oqmd_oxides_structural_features.csv", index=False)
nitrides_structured.to_csv("oqmd_nitrides_structural_features.csv", index=False)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# === Load datasets ===
oxides = pd.read_csv("oqmd_oxides_structural_features.csv")
nitrides = pd.read_csv("oqmd_nitrides_structural_features.csv")

oxides['system'] = 'oxide'
nitrides['system'] = 'nitride'

# === Target properties ===
properties = [
"band_gap", "delta_e", "volume", "ionic_density",
"approx_coordination", "ionic_radius_diff", "coordination_number"
]

# === Structure features (limit to a few for readability) ===
structure_features = [col for col in oxides.columns if col.startswith("site") or col.startswith("symmetry")]
structure_features_short = structure_features[:6]

# === Final list of features for ANOVA/plots ===
analysis_features = properties + structure_features_short

# === Style settings ===
sns.set(style="whitegrid", font_scale=1.1)

# === Function to analyze per system ===
def analyze_per_system(df, system_name):
print(f"\n\n=== {system_name.upper()} SYSTEM ===")

# === Patch: Ensure coordination_number exists ===
if 'coordination_number' not in df.columns:
if 'actual_coordination' in df.columns:
df['coordination_number'] = df['actual_coordination']
elif 'mean_coordination_number' in df.columns:
df['coordination_number'] = df['mean_coordination_number']
elif 'approx_coordination' in df.columns:
#print("⚠️ 'coordination_number' not found, using 'approx_coordination' as proxy.")
df['coordination_number'] = df['approx_coordination']
else:
print("❌ ERROR: No coordination number column found. Skipping analysis.")
return

# Drop missing key values
df = df.dropna(subset=["cation", "band_gap", "delta_e", "volume", "ionic_density"])

# Filter to cations with at least 2 samples
valid_cations = df["cation"].value_counts()[lambda x: x >= 2].index.tolist()
df = df[df["cation"].isin(valid_cations)]

print(f"Cations included in ANOVA (≥2 samples): {valid_cations}")

anova_results = []

for prop in analysis_features:
try:
plot_df = df[["cation", prop]].dropna()

if plot_df[prop].nunique() <= 1:
print(f"Skipping {prop}: Not enough variation.")
continue

# --- Boxplot ---
plt.figure(figsize=(8, 5))
sns.boxplot(data=plot_df, x="cation", y=prop)
plt.title(f"[{system_name.upper()}] Effect of Cation on '{prop}'")
plt.tight_layout()
plt.show()

# --- ANOVA ---
groups = [group[prop].dropna().values for _, group in df.groupby("cation")]
if all(len(g) > 1 for g in groups):
stat, pval = f_oneway(*groups)
if pval < 0.001:
note = "Strongly significant"
elif pval < 0.01:
note = "Significant"
elif pval < 0.05:
note = "Moderately significant"
else:
note = "Not significant"
print(f"{prop:25s} → F = {stat:.2f}, p = {pval:.2e} → {note}")
anova_results.append((prop, stat, pval, note))
else:
print(f"{prop:25s} → Skipped: Not enough data per group.")
anova_results.append((prop, None, None, "Insufficient data per group"))

except Exception as e:
print(f"{prop:25s} → Error during processing: {e}")
anova_results.append((prop, None, None, f"Error: {e}"))

# === Save Results ===
anova_df = pd.DataFrame(anova_results, columns=["property", "F_stat", "p_value", "inference"])
output_file = f"{system_name}_cation_effect_anova.csv"
anova_df.to_csv(output_file, index=False)
print(f"Saved ANOVA results to: {output_file}")


# === Run for oxides and nitrides separately ===
analyze_per_system(oxides, "oxides")
analyze_per_system(nitrides, "nitrides")

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")

# === Load your data ===
oxides = pd.read_csv("oqmd_oxides_structural_features.csv")
nitrides = pd.read_csv("oqmd_nitrides_structural_features.csv")
oxides["system"] = "oxide"
nitrides["system"] = "nitride"

targets = ["band_gap", "delta_e"]
results = {}

def predict_rf_with_shap(df, system_name):
print(f"\n=== {system_name.upper()} ===")

df = df.dropna(subset=targets)

# --- Detect feature columns ---
excluded = ["compound", "formula", "system", "cation"] + targets
features = df.select_dtypes(include=[np.number]).columns.difference(excluded).tolist()

for target in targets:
print(f"\n➡️ Predicting {target} for {system_name}")

df_target = df.dropna(subset=[target])
X = df_target[features]
y = df_target[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Hyperparameter tuning ---
param_grid = {
"n_estimators": [100, 200],
"max_depth": [None, 10, 20],
"min_samples_split": [2, 5]
}

grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Best Params: {grid.best_params_}")
print(f"📉 MSE: {mse:.3f}, R²: {r2:.3f}")

# --- SHAP (CPU-friendly tree explainer) ---
explainer = shap.Explainer(model, X_train, algorithm="tree")
shap_values = explainer(X_test)

# --- SHAP Plot ---
plt.figure()
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.title(f"SHAP Summary - {system_name.upper()} - {target}")
plt.tight_layout()
plt.savefig(f"shap_{system_name}_{target}.png")
plt.close()
print(f"📊 SHAP plot saved to shap_{system_name}_{target}.png")

results[(system_name, target)] = {
"model": model,
"mse": mse,
"r2": r2,
"shap": shap_values,
"features": features
}

# === Run for both systems ===
predict_rf_with_shap(oxides, "oxides")
predict_rf_with_shap(nitrides, "nitrides")
