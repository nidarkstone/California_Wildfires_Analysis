import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('/Users/nidakaratas/Desktop/mapdataall.csv')

df_clean = df[
    (df['incident_latitude'] >= 32) & 
    (df['incident_latitude'] <= 42.5) & 
    (df['incident_longitude'] >= -125) & 
    (df['incident_longitude'] <= -114)
].copy()

df_clean = df_clean.dropna(subset=['incident_acres_burned', 'incident_administrative_unit'])
df_clean = df_clean[df_clean['incident_acres_burned'] > 0] #remove 0 

# North vs South (37. Enlem Referans Alındı)
LAT_SPLIT = 37.0
df_clean['Region'] = np.where(df_clean['incident_latitude'] > LAT_SPLIT, 'North', 'South')


# Visualization

# Graph 1: Study Area (Map) 
fig1, ax1 = plt.subplots(figsize=(10, 12))

# California coords 
ca_border_coords = [(42.0, -124.2), (42.0, -120.0), (39.0, -120.0), (35.0, -114.6),
                    (34.3, -114.1), (32.7, -114.7), (32.5, -117.1), (34.4, -120.5),
                    (36.6, -122.0), (38.0, -123.0), (40.4, -124.4), (42.0, -124.2)]
bx = [x[1] for x in ca_border_coords]
by = [x[0] for x in ca_border_coords]
ax1.plot(bx, by, 'k--', alpha=0.3, zorder=1, label='State Border')

sns.scatterplot(
    data=df_clean, 
    x='incident_longitude', 
    y='incident_latitude',
    hue='Region', 
    palette={'North': '#3498db', 'South': '#e74c3c'}, # North (Kuzey), South (Güney)
    size='incident_acres_burned', 
    sizes=(10, 200), 
    alpha=0.6, 
    ax=ax1
)

ax1.set_title('Fig 1: Study Area & Spatial Distribution (North vs. South)', fontsize=14)
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend(loc='upper right', title='Region')
ax1.set_aspect(1.25) 
plt.tight_layout()
plt.savefig('Fig1_Study_Area_Map.png', dpi=300)
print("Grafik 1 Kaydedildi: Fig1_Study_Area_Map.png")


#Graph 2: Top Administrative Units (Bar Chart) ---
agency_counts = df_clean['incident_administrative_unit'].value_counts().head(15)

fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.barplot(x=agency_counts.values, y=agency_counts.index, palette='viridis', ax=ax2)

ax2.set_title('Fig 2: Top Administrative Units by Operational Load (Incident Count)', fontsize=14)
ax2.set_xlabel('Number of Incidents')
ax2.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Fig2_Top_Agencies.png', dpi=300)
print("Grafik 2 Kaydedildi: Fig2_Top_Agencies.png")


#Graph 3: Risk Quadrandt (Scatter)
unit_stats = df_clean.groupby('incident_administrative_unit').agg({
    'incident_id': 'count',
    'incident_acres_burned': 'sum',
    'incident_latitude': 'mean'
}).rename(columns={'incident_id': 'Frequency', 'incident_acres_burned': 'Total_Severity', 'incident_latitude': 'Avg_Latitude'})

# choose top 30 units
top_units = unit_stats.sort_values('Frequency', ascending=False).head(30).copy()
top_units['Region'] = np.where(top_units['Avg_Latitude'] > LAT_SPLIT, 'North', 'South')

fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    data=top_units, 
    x='Frequency', 
    y='Total_Severity', 
    hue='Region',
    style='Region', 
    s=150, 
    palette={'North': '#3498db', 'South': '#e74c3c'},
    edgecolor='black', 
    ax=ax3
)


ax3.set_yscale('log')
ax3.axvline(top_units['Frequency'].median(), color='gray', ls='--', alpha=0.5)
ax3.axhline(top_units['Total_Severity'].median(), color='gray', ls='--', alpha=0.5)

ax3.text(top_units['Frequency'].max(), top_units['Total_Severity'].max(), 'CRITICAL ZONE\n(High Freq / High Sev)', ha='right', va='top', color='darkred', fontsize=10, fontweight='bold')
ax3.text(top_units['Frequency'].max(), top_units['Total_Severity'].min(), 'OPERATIONAL FATIGUE\n(High Freq / Low Sev)', ha='right', va='bottom', color='orange', fontsize=10, fontweight='bold')

ax3.set_title('Fig 3: Risk Profiles: Frequency vs. Severity Trade-off', fontsize=14)
ax3.set_xlabel('Operational Load (Frequency)')
ax3.set_ylabel('Catastrophic Risk (Total Acres - Log Scale)')
ax3.grid(True, which="both", ls="--", alpha=0.2)
plt.tight_layout()
plt.savefig('Fig3_Risk_Quadrant.png', dpi=300)
print("Grafik 3 Kaydedildi: Fig3_Risk_Quadrant.png")


#Graph 4: Latitude Trend (Trend Line) ---
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.regplot(
    data=top_units, 
    x='Avg_Latitude', 
    y='Total_Severity',
    scatter_kws={'s':100, 'alpha':0.6}, 
    line_kws={'color':'red'}, 
    ax=ax4
)

ax4.set_yscale('log')
ax4.set_title('Fig 4: Geographic Determinism (Latitude vs. Fire Severity)', fontsize=14)
ax4.set_xlabel('Latitude (South -> North)')
ax4.set_ylabel('Total Acres Burned (Log Scale)')
ax4.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig('Fig4_Latitude_Trend.png', dpi=300)
print("Grafik 4 Kaydedildi: Fig4_Latitude_Trend.png")

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as stats

# Creation of model 
y = np.log10(top_units['Total_Severity'])
X = top_units['Avg_Latitude']
X_with_const = sm.add_constant(X) 

model = sm.OLS(y, X_with_const).fit()
residuals = model.resid

# assumption check needed
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# normality check (Q-Q Plot)
stats.probplot(residuals, dist="norm", plot=axes[0])
axes[0].set_title('Normal Q-Q Plot (Hataların Normalliği)')

# Equal Variance check (Residuals vs Fitted)
axes[1].scatter(model.fittedvalues, residuals)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel('Tahmin Edilen Değerler')
axes[1].set_ylabel('Artıklar (Residuals)')
axes[1].set_title('Residuals vs Fitted (Eş Varyans)')

# Distribution of Residuals (Histogram)
sns.histplot(residuals, kde=True, ax=axes[2])
axes[2].set_title('Hataların Dağılım Histogramı')

plt.tight_layout()
plt.show()


# Shapiro-Wilk 
shapiro_test = stats.shapiro(residuals)
print(f"Normality (Shapiro-Wilk) p-value: {shapiro_test.pvalue:.4f}")

# Breusch-Pagan 
_, p_val_bp, _, _ = het_breuschpagan(residuals, X_with_const)
print(f"Equal Variance (Breusch-Pagan) p-value: {p_val_bp:.4f}")

# Model summary
print("\n--- Regression REsults ---")
print(f"R-Kare: {model.rsquared:.4f}")
print(f"Latitude Coefficient (Slope): {model.params['Avg_Latitude']:.4f}")
