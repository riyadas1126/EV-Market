# Data Manipulation and Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Configure Display Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Load dataset
df = pd.read_csv('Electric_Vehicle_Population_Data.csv')

# Display basic information
df.head()
df.info()

# Check for missing values
df.isnull().sum()

# Drop rows with missing values
df = df.dropna()

# Count EV registrations by model year
ev_adoption = df['Model Year'].value_counts().sort_index()

# Visualize EV adoption over time
plt.figure(figsize=(12, 8))
sns.barplot(x=ev_adoption.index, y=ev_adoption.values, palette="viridis")
plt.xlabel('Model Year')
plt.ylabel('Number of EVs Registered')
plt.title('EV Adoption Over Time')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Analyze geographical distribution at the county level
EV_country_distribution = df['County'].value_counts()
top_counties = EV_country_distribution.head(3).index

# Filter dataset for these top counties
top_counties_data = df[df['County'].isin(top_counties)]

# Analyze EV distribution within the cities of these top counties
ev_city_distribution_top_counties = top_counties_data.groupby(['County', 'City']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')

# Visualize the top 10 cities across these counties
top_cities = ev_city_distribution_top_counties.head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x='Number of Vehicles', y='City', hue='County', data=top_cities, palette="magma")
plt.title('Top Cities in Top Counties by EV Registrations')
plt.xlabel('Number of Vehicles Registered')
plt.ylabel('City')
plt.legend(title='County')
plt.tight_layout()
plt.show()

# Analyze the distribution of electric vehicle types
ev_type_distribution = df['Electric Vehicle Type'].value_counts()

plt.figure(figsize=(12, 8))
sns.barplot(x=ev_type_distribution.values, y=ev_type_distribution.index, palette='coolwarm')
plt.xlabel('Number of Vehicles')
plt.ylabel('EV Types')
plt.title('Distribution of Electric Vehicle Types')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Count vehicles by make
make_popularity = df['Make'].value_counts().head(10)

# Visualize the top 10 most popular makes
plt.figure(figsize=(12, 6))
sns.barplot(x=make_popularity.values, y=make_popularity.index, palette="viridis")
plt.title('Top 10 Most Popular Makes')
plt.xlabel('Number of Vehicles')
plt.ylabel('Make')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Select the top 3 manufacturers based on registrations
top_3_makes = make_popularity.head(3).index
top_makes_data = df[df['Make'].isin(top_3_makes)]

# Analyze EV model popularity within top manufacturers
ev_model_distribution_top_makes = top_makes_data.groupby(['Make', 'Model']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')

# Visualize the top 10 models
top_models = ev_model_distribution_top_makes.head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x='Number of Vehicles', y='Model', hue='Make', data=top_models, palette='viridis')
plt.title('Top Models in Top 3 Makes by EV Registrations')
plt.xlabel('Number of Vehicles Registered')
plt.ylabel('Model')
plt.legend(title='Make', loc='center right')
plt.tight_layout()
plt.show()

#Distribution of Electric Range
plt.figure(figsize=(12, 8))
sns.histplot(df['Electric Range'], bins=30, kde=True, color='blue')
plt.title('Distribution of Electric Vehicle Ranges')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Number of Vehicles')
plt.axvline(df['Electric Range'].mean(), color='red', linestyle='--', label=f'Mean Range: {df["Electric Range"].mean():.2f} miles')
plt.legend()
plt.show()

#Average Electric Range by Model Year
# Calculate the average electric range by Model Year
avg_range_by_year = df.groupby('Model Year')['Electric Range'].mean().reset_index()

# Visualize the trend
plt.figure(figsize=(12, 8))
sns.lineplot(data=avg_range_by_year, x='Model Year', y='Electric Range', marker='o', color='green')
plt.title('Average Electric Range by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Average Electric Range (miles)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Electric Range by Top Manufacturers and Models
# Calculate average electric range by Make and Model
range_by_make_mode = top_makes_data.groupby(['Make', 'Model'])['Electric Range'].mean().reset_index()
top_range_models = range_by_make_mode.sort_values(by='Electric Range', ascending=False).head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x='Electric Range', y='Model', hue='Make', data=top_range_models, palette='cool')
plt.title('Top 10 Models by Average Electric Range in Top Makes')
plt.xlabel('Average Electric Range (miles)')
plt.ylabel('Model')
plt.legend(title='Make', loc='center right')
plt.show()

# Calculate EV registrations by year
ev_counts = df['Model Year'].value_counts().sort_index()
filtered_years = ev_counts[ev_counts.index <= 2023]

# Define exponential growth function
def exp_growth(x, a, b):
    return a * np.exp(b * x)

# Prepare data for curve fitting
x_data = filtered_years.index - filtered_years.index.min()
y_data = filtered_years.values

# Fit data
params, covariance = curve_fit(exp_growth, x_data, y_data)

# Forecast future growth (2024–2029)
forecast_years = np.arange(2024, 2024 + 6) - filtered_years.index.min()
forecasted_values = exp_growth(forecast_years, *params)

plt.figure(figsize=(12, 8))
plt.plot(filtered_years.index, filtered_years.values, 'bo-', label='Historical Data')
plt.plot(forecast_years + filtered_years.index.min(), forecasted_values, 'ro--', label='Forecasted Data')
plt.title('Market Size Growth and Projection for Electric Vehicles (2024–2029)')
plt.xlabel('Year')
plt.ylabel('Number of Vehicles Registered')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
