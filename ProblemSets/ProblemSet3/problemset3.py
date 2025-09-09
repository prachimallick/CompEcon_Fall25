# %%
import pandas as pd
import os

# Set your path
base_path = r"C:\Users\Prachi Mallick\OneDrive - University of South Carolina\3rd year Research\EOBZIP_2025_06\\"

# File paths
products_file = os.path.join(base_path, "products.txt")
patent_file = os.path.join(base_path, "patent.txt")
exclusivity_file = os.path.join(base_path, "exclusivity.txt")

# Load just the headers using sep="~"
products = pd.read_csv(products_file, sep="~", nrows=0)   # only header row
patents = pd.read_csv(patent_file, sep="~", nrows=0)
exclusivity = pd.read_csv(exclusivity_file, sep="~", nrows=0)

# Print out the variable names
print("Products file columns:")
print(products.columns.tolist(), "\n")

print("Patents file columns:")
print(patents.columns.tolist(), "\n")

print("Exclusivity file columns:")
print(exclusivity.columns.tolist(), "\n")


# %%
import pandas as pd
import os

# ----------------------------------------
# 1. Load the three Orange Book datasets
# ----------------------------------------
base_path = r"C:\Users\Prachi Mallick\OneDrive - University of South Carolina\3rd year Research\EOBZIP_2025_06\\"

products_file = os.path.join(base_path, "products.txt")
patent_file = os.path.join(base_path, "patent.txt")
exclusivity_file = os.path.join(base_path, "exclusivity.txt")


products = pd.read_csv(products_file, sep="~", dtype=str)
patents = pd.read_csv(patent_file, sep="~", dtype=str)
exclusivity = pd.read_csv(exclusivity_file, sep="~", dtype=str)

print("Loaded data")
print("Products shape:", products.shape)
print("Patents shape:", patents.shape)
print("Exclusivity shape:", exclusivity.shape)

# ----------------------------------------
# 2. Merge the datasets
# ----------------------------------------
# Merge products with patents
products_patents = pd.merge(
    products,
    patents,
    how="left",              # keep all products, even if no patents
    on=["Appl_No", "Product_No"]
)

# Merge with exclusivity
merged = pd.merge(
    products_patents,
    exclusivity,
    how="left",              # keep all products, even if no exclusivity
    on=["Appl_No", "Product_No"]
)

# ----------------------------------------
# 3. Clean dates
# ----------------------------------------
merged["Patent_Expire_Date_Text"] = pd.to_datetime(
    merged["Patent_Expire_Date_Text"], errors="coerce"
)
merged["Exclusivity_Date"] = pd.to_datetime(
    merged["Exclusivity_Date"], errors="coerce"
)

# ----------------------------------------
# 4. Quick summary checks
# ----------------------------------------
print("\n Merged dataset created")
print("Merged shape:", merged.shape)

# How many products have at least one patent?
print("Products with patents:", merged["Patent_No"].notna().sum())

# How many products have exclusivity?
print("Products with exclusivity:", merged["Exclusivity_Code"].notna().sum())

# Example: look at all rows for a single drug
print("\nExample: all rows for AMZEEQ")
print(merged[merged["Trade_Name"].str.contains("AMZEEQ", na=False)].head(10))

# ----------------------------------------
# 5. Save for later
# ----------------------------------------
merged.to_csv("merged_orangebook.csv", index=False)
print("\n Saved merged dataset as merged_orangebook.csv")


# %%
import os
print(os.listdir(base_path))


# %%
import pandas as pd
import os

# Your base path
base_path = r"C:\Users\Prachi Mallick\OneDrive - University of South Carolina\3rd year Research\EOBZIP_2025_06\\"

# File paths
products_file = os.path.join(base_path, "products.txt")
patent_file = os.path.join(base_path, "patent.txt")
exclusivity_file = os.path.join(base_path, "exclusivity.txt")

# Load with correct delimiter (~)
products = pd.read_csv(products_file, sep="~", dtype=str)
patents = pd.read_csv(patent_file, sep="~", dtype=str)
exclusivity = pd.read_csv(exclusivity_file, sep="~", dtype=str)

print("Products shape:", products.shape)
print("Patents shape:", patents.shape)
print("Exclusivity shape:", exclusivity.shape)

print("\nProducts columns:", products.columns.tolist())
print("Patents columns:", patents.columns.tolist())
print("Exclusivity columns:", exclusivity.columns.tolist())

# ---------------------------
# Merge datasets
# ---------------------------
# Merge products with patents
products_patents = pd.merge(
    products,
    patents,
    how="left",
    on=["Appl_No", "Product_No"]
)

# Merge with exclusivity
merged = pd.merge(
    products_patents,
    exclusivity,
    how="left",
    on=["Appl_No", "Product_No"]
)

# Convert patent expiration date
merged["Patent_Expire_Date_Text"] = pd.to_datetime(
    merged["Patent_Expire_Date_Text"], errors="coerce"
)

# Save merged dataset
merged.to_csv("merged_orangebook.csv", index=False)
print("Merged dataset saved as merged_orangebook.csv")
print(merged.head())


# %%
import matplotlib.pyplot as plt

# Drop missing dates
expiry_data = merged.dropna(subset=["Patent_Expire_Date_Text"]).copy()

# Extract year
expiry_data["Patent_Year"] = expiry_data["Patent_Expire_Date_Text"].dt.year

plt.figure(figsize=(10,6))
expiry_data["Patent_Year"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution of Patent Expiration Years (Orange Book)")
plt.xlabel("Year")
plt.ylabel("Number of Patents Expiring")
plt.tight_layout()
plt.savefig("fig1.png") 
plt.show()


# %%
# Plot 1: Distribution of Patent Expiration Years
# ----------------------------------------------
# This figure shows how many drug patents expire in each year.
# Years with a higher bar indicate a "wave" of expirations — in these years,
# many generics can enter at once, putting extra pressure on brand-name firms.
# In my dataset, 2027 stands out as a peak year for expirations.Year 2026 and year 2031 also stand out
# This is precisely when companies may feel the strongest pressure
# to advertise directly to patients to slow switching to generics.


# %%
# Plot 2: Patent Expiration Trends by "Treatment" vs. "Control"
# -------------------------------------------------------------
# In my research question, I eventually want to compare drug company that did patient-directed advertising as treatment group
# to those that did not use patient directed advertising as control group, before and after patent expiry.
# However, since I do not yet have advertising data yet, here I create a placeholder treatment definition for illustration: 
# drugs made by large pharmaceutical firms (Pfizer, Merck, Novartis, Johnson & Johnson) are coded as "treatment", 
# while all other firms are "control".

import numpy as np

# Example: Define "treatment" as drugs from big pharma companies
merged["Treatment"] = np.where(
    merged["Applicant_Full_Name"].str.contains("Pfizer|Merck|Novartis|Johnson", case=False, na=False),
    1, 0
)

# Average number of expirations per year by group
trend = (
    merged.dropna(subset=["Patent_Expire_Date_Text"])
    .groupby([merged["Patent_Expire_Date_Text"].dt.year, "Treatment"])
    .size()
    .reset_index(name="Count")
)

plt.figure(figsize=(10,6))
for t, label in [(1,"Treatment"), (0,"Control")]:
    subset = trend[trend["Treatment"]==t]
    plt.plot(subset["Patent_Expire_Date_Text"], subset["Count"], marker="o", label=label)

plt.title("Patent Expiration Trends by Treatment vs Control")
plt.xlabel("Year")
plt.ylabel("Number of Expiring Patents")
plt.legend()
plt.tight_layout()
plt.savefig("fig2.png") 
plt.show()


# %%
# Count how many products per ingredient per year
ingredient_counts = (
    merged.dropna(subset=["Patent_Expire_Date_Text"])
    .groupby([merged["Ingredient"], merged["Patent_Expire_Date_Text"].dt.year])
    .size()
    .reset_index(name="NumProducts")
)

# Take average number of products per ingredient each year
avg_counts = ingredient_counts.groupby("Patent_Expire_Date_Text")["NumProducts"].mean()

plt.figure(figsize=(10,6))
avg_counts.plot(marker="o")
plt.title("Average Number of Products per Ingredient Over Time")
plt.xlabel("Year")
plt.ylabel("Avg Number of Products (Proxy for Generic Entry)")
plt.tight_layout()
plt.savefig("fig3.png")
plt.show()


# %%



