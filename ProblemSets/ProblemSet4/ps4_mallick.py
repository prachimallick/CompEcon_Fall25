#!/usr/bin/env python
# coding: utf-8

# ### Problem Set 4
# #### ECON 833
# **Name: Prachi Mallick**

# In[13]:


"""
MLE vs OLS estimation of log-wage model
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

# Load data
df = pd.read_stata("PSID_data.dta")

# Create hourly wage = income / hours
df["hr_wage"] = df["hlabinc"] / df["hannhrs"]

# Filter conditions
df = df[
    (df["hsex"] == 1) &               # male
    (df["age"].between(25, 60)) &     # age 25–60
    (df["hr_wage"] > 7)               # hourly wage > $7
].copy()

# Add log wage
df["ln_wage"] = np.log(df["hr_wage"])

print(df.head())




# #### Regression Model
# 
# The model we estimate is:
# 
# $$
# \ln(w_{i,t}) = \alpha + \beta_1 Educi,t + \beta_2 Agei,t + \beta_3 Agei,t^2
# + \beta_4 Blacki,t + \beta_5 Hispanici,t + \beta_6 OtherRacei,t + \varepsilon_{i,t}
# $$
# 
# 

# In[19]:


# Define variables
df["educ"] = df["hyrsed"]
df["age2"] = df["age"]**2
df["black"] = (df["hrace"] == 2).astype(int)
df["hispanic"] = (df["hrace"] == 5).astype(int)
df["other"] = (~df["hrace"].isin([1, 2, 5])).astype(int)

# Drop missing rows
reg_vars = ["ln_wage", "educ", "age", "age2", "black", "hispanic", "other"]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=reg_vars)

# Build X and y
X = df[["educ", "age", "age2", "black", "hispanic", "other"]]
X = sm.add_constant(X)
y = df["ln_wage"]

# OLS
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

# MLE
init_params = np.append(ols_model.params.values, ols_model.resid.std())
bounds = [(None, None)]*X.shape[1] + [(1e-6, None)]







# #### Interpretation of OLS Results 
# 
# 
#  **Education (0.0764)**: Each additional year of education increases log wages by about **7.6%** on average, holding other factors constant. This is highly significant (p < 0.000).
# 
#  **Age (0.0603) and Age² (-0.0006)**: Wages rise with age, but at a decreasing rate. The positive age coefficient and negative age² coefficient indicate a concave wage–age profile.
# 
#  **Black (-0.1627)**: Being Black is associated with about **16% lower wages** compared to otherwise similar White males, and the result is significant.
# 
#  **Hispanic (0.0000)**: The Hispanic dummy shows no effect in this sample .Because the sample doesnt have any hispanic male.
# 
#  **Other (0.0085)**: No significant difference in wages for the "Other" race category relative to White.
# 
# 
# 

# In[20]:


def neg_loglike(params, y, X):
    """
    Negative log-likelihood for linear regression with normal errors.

    Parameters
    ----------
    params : array-like
        Model parameters. The first k elements are regression coefficients (beta),
        and the last element is the standard deviation (sigma).
    y : array-like
        Dependent variable (n x 1).
    X : array-like
        Design matrix of regressors (n x k).

    Returns
    -------
    float
        Negative log-likelihood value to be minimized.
    """
    beta = params[:-1]
    sigma = params[-1]
    if sigma <= 0:
        return np.inf
    resid = y - X @ beta
    n = len(y)
    ll = -0.5*n*np.log(2*np.pi*sigma**2) - 0.5*np.sum((resid/sigma)**2)
    return -ll

result = minimize(neg_loglike, init_params, args=(y, X),
                  method="L-BFGS-B", bounds=bounds)

mle_params = result.x
print("MLE coefficients:", mle_params[:-1])
print("MLE sigma:", mle_params[-1])


# In[8]:


years = [1971, 1980, 1990, 2000]
results = []

for year in years:
    df_year = df[df["year"] == year].copy()
    df_year = df_year.dropna(subset=["ln_wage"] + reg_vars)

    X = df_year[["educ", "age", "age2", "black", "hispanic", "other"]]
    X = sm.add_constant(X)
    y = df_year["ln_wage"]

    # OLS
    ols_model = sm.OLS(y, X).fit()
    beta1_ols = ols_model.params["educ"]

    # MLE
    init_params = np.append(ols_model.params.values, ols_model.resid.std())
    bounds = [(None, None)]*X.shape[1] + [(1e-6, None)]
    result = minimize(neg_loglike, init_params, args=(y, X),
                      method="L-BFGS-B", bounds=bounds)
    beta1_mle = result.x[X.columns.get_loc("educ")]

    results.append({
        "year": year,
        "OLS_beta1": beta1_ols,
        "MLE_beta1": beta1_mle
    })

results_df = pd.DataFrame(results)
results_df


# #### Interpretation of β₁ (Education)
# 
# The coefficient on education (β₁) indicates the **percentage increase in wages** from an additional year of schooling.  
# 
# - In **1971**, returns were about **6.7%**.  
# - In **1980**, they remained stable at about **6.6%**.  
# - By **1990**, returns increased to roughly **9.6%**.  
# - In **2000**, they rose further to around **11.0%**.  
# 
# So we see that returns to education were relatively flat in the 1970s–1980s but rose significantly in the 1990s and 2000s. This suggests that education has become increasingly rewarded in the labor market over time.
# 
# 

# 
