# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:42:02 2024

@author: HP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the CSV files
co2_data = pd.read_csv("co2data.csv")  # Read CO2 data
income_data = pd.read_csv("incomedata.csv")  # Read income data

# Filling the blank spaces in co2_data with 0.0
co2_data.fillna(0.0, inplace=True)

# Filtering co2_data based on indicators chosen
filtered_co2_data = co2_data[co2_data["Indicator Name"].isin([
   "Urban population (% of total population)",
   "Forest area (sq. km)",
   "Agricultural land (% of land area)",
   "Mortality rate, under-5 (per 1,000 live births)",
   "CO2 emissions (metric tons per capita)"
])]

# Resetting the index
filtered_co2_data = filtered_co2_data.reset_index()

# Removing non-numeric first lines
filtered_co2_data = filtered_co2_data.drop(filtered_co2_data.columns[[0]], axis='columns')

# Merging the filtered co2_data with the IncomeGroup column from income_data
merged_data = pd.merge(filtered_co2_data, income_data[["Country Code", "IncomeGroup"]], on="Country Code", how="left")

# Reordering the columns to place IncomeGroup at the last
merged_data = merged_data[[col for col in merged_data.columns if col != "IncomeGroup"] + ["IncomeGroup"]]

# Saving the modified data to a new CSV file
merged_data.to_csv("updated_co2data.csv", index=False)

# Transposing dataset
transposed_dataset = merged_data.T

# Reading the updated CSV file
updated_data = pd.read_csv("updated_co2data.csv")

# Displaying the contents of the DataFrame 
print(updated_data)

# Descriptive statistics
print(updated_data.describe())

# Selecting countries from rich, middle income (now) and poor
rich =  ['United Kingdom']
upper_middle_income = ['Indonesia'] 
lower_middle_income = ['India']
poor = ['Afghanistan']

first20_years = [str(year) for year in range(1960, 1986)]
last20_years = [str(year) for year in range(1990, 2024)]

# Extract data for the first third (1960 to 1985) and last third (1990 to 2023)
first_20yr_data = updated_data[["Country Name", "Country Code", "Indicator Name", "Indicator Code", "IncomeGroup"] + first20_years[0:26]]
last_20yr_data = updated_data[["Country Name", "Country Code", "Indicator Name", "Indicator Code", "IncomeGroup"] + last20_years[0:34]]

# Calculate the correlation matrix for the first 20 years
first_20yr_correlation_matrix = first_20yr_data.drop(["Country Name", "Country Code", "Indicator Name", "Indicator Code", "IncomeGroup"], axis=1).corr()

# Calculate the correlation matrix for the recent 20 years
last_20yr_correlation_matrix = last_20yr_data.drop(["Country Name", "Country Code", "Indicator Name", "Indicator Code", "IncomeGroup"], axis=1).corr()

# Calculate the average rate of change for each indicator over the 20-year intervals
avg_rate_of_change = {}
for indicator in first_20yr_data["Indicator Name"].unique():
    indicator_values = first_20yr_data[first_20yr_data["Indicator Name"] == indicator][first20_years].values
    # Exclude zero and missing values from calculations
    valid_values_mask = np.logical_and(indicator_values != 0, ~np.isnan(indicator_values))
    rate_of_change = np.diff(indicator_values, axis=1) / indicator_values[:, :-1]
    rate_of_change[~valid_values_mask[:, :-1]] = np.nan  # Set invalid values to NaN
    avg_rate_of_change[indicator] = np.nanmean(rate_of_change)

# Output the correlation matrices and average rate of change
print("Correlation Matrix for the first 20 years (1960-1985):")
print(first_20yr_correlation_matrix)

print("\nAverage Rate of Change:")
for indicator, rate in avg_rate_of_change.items():
    print(f"{indicator}: {rate}")

# Define a function to perform bootstrapping and compute the confidence interval
def bootstrap_mean_confidence_interval(data, num_samples=1000, alpha=0.05):
    means = []
    for _ in range(num_samples):
        # Generate a bootstrap sample by sampling with replacement from the original dataset
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        mean = np.mean(bootstrap_sample)
        means.append(mean)
    
    # Compute the confidence interval using percentiles
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    lower_bound = np.percentile(means, lower_percentile)
    upper_bound = np.percentile(means, upper_percentile)
    
    return lower_bound, upper_bound


# Select data for the first 20 years and filter by income group
first_20yr_co2_emissions = first_20yr_data[first20_years].values.flatten()
rich_co2_emissions_first_20yr = first_20yr_data[first_20yr_data["Country Name"].isin(rich)][first20_years].values.flatten()
upper_middle_income_co2_emissions_first_20yr = first_20yr_data[first_20yr_data["Country Name"].isin(upper_middle_income)][first20_years].values.flatten()
lower_middle_income_co2_emissions_first_20yr = first_20yr_data[first_20yr_data["Country Name"].isin(lower_middle_income)][first20_years].values.flatten()
poor_co2_emissions_first_20yr = first_20yr_data[first_20yr_data["Country Name"].isin(poor)][first20_years].values.flatten()

# Compute bootstrapped confidence intervals for the mean CO2 emissions for the first 20 years
rich_ci_first_20yr = bootstrap_mean_confidence_interval(rich_co2_emissions_first_20yr)
upper_middle_income_ci_first_20yr = bootstrap_mean_confidence_interval(upper_middle_income_co2_emissions_first_20yr)
lower_middle_income_ci_first_20yr = bootstrap_mean_confidence_interval(lower_middle_income_co2_emissions_first_20yr)
poor_ci_first_20yr = bootstrap_mean_confidence_interval(poor_co2_emissions_first_20yr)

# Select data for the last 20 years and filter by income group
last_20yr_co2_emissions = last_20yr_data[last20_years].values.flatten()
rich_co2_emissions_last_20yr = last_20yr_data[last_20yr_data["Country Name"].isin(rich)][last20_years].values.flatten()
upper_middle_income_co2_emissions_last_20yr = last_20yr_data[last_20yr_data["Country Name"].isin(upper_middle_income)][last20_years].values.flatten()
lower_middle_income_co2_emissions_last_20yr = last_20yr_data[last_20yr_data["Country Name"].isin(lower_middle_income)][last20_years].values.flatten()
poor_co2_emissions_last_20yr = last_20yr_data[last_20yr_data["Country Name"].isin(poor)][last20_years].values.flatten()

# Compute bootstrapped confidence intervals for the mean CO2 emissions for the last 20 years
rich_ci_last_20yr = bootstrap_mean_confidence_interval(rich_co2_emissions_last_20yr)
upper_middle_income_ci_last_20yr = bootstrap_mean_confidence_interval(upper_middle_income_co2_emissions_last_20yr)
lower_middle_income_ci_last_20yr = bootstrap_mean_confidence_interval(lower_middle_income_co2_emissions_last_20yr)
poor_ci_last_20yr = bootstrap_mean_confidence_interval(poor_co2_emissions_last_20yr)

# Print the results for the first 20 years
print("\nBootstrapped 95% confidence intervals for mean CO2 emissions (metric tons per capita) for the first 20 years:")
print("Rich countries:", rich_ci_first_20yr)
print("Upper middle-income countries:", upper_middle_income_ci_first_20yr)
print("Lower middle-income countries:", lower_middle_income_ci_first_20yr)
print("Poor countries:", poor_ci_first_20yr)

# Print the results for the last 20 years
print("\nBootstrapped 95% confidence intervals for mean CO2 emissions (metric tons per capita) for the last 20 years:")
print("Rich countries:", rich_ci_last_20yr)
print("Upper middle-income countries:", upper_middle_income_ci_last_20yr)
print("Lower middle-income countries:", lower_middle_income_ci_last_20yr)
print("Poor countries:", poor_ci_last_20yr)

# Plotting for last 20 year data 

# Select relevant columns for CO2 emissions and forest area
co2_data = updated_data[updated_data['Indicator Name'] == 'CO2 emissions (metric tons per capita)'][['Country Name', 'IncomeGroup'] + last20_years]
forest_data = updated_data[updated_data['Indicator Name'] == 'Forest area (sq. km)'][['Country Name', 'IncomeGroup'] + last20_years]

# Set 'Country Name' and 'IncomeGroup' as index
co2_data.set_index(['Country Name', 'IncomeGroup'], inplace=True)
forest_data.set_index(['Country Name', 'IncomeGroup'], inplace=True)

# Select relevant columns for CO2 emissions and forest area for rich countries
rich_co2_data = co2_data.loc[rich]
rich_forest_data = forest_data.loc[rich]

poor_co2_data = co2_data.loc[poor]
poor_forest_data = forest_data.loc[poor]

lower_middle_income_co2_data = co2_data.loc[lower_middle_income]
lower_middle_income_forest_data = forest_data.loc[lower_middle_income]

upper_middle_income_co2_data = co2_data.loc[upper_middle_income]
upper_middle_income_forest_data = forest_data.loc[upper_middle_income]

# Calculate correlation using corrwith() for each income group
rich_correlation_results = rich_co2_data.corrwith(rich_forest_data, axis=1)
poor_correlation_results = poor_co2_data.corrwith(poor_forest_data, axis=1)
lower_middle_income_correlation_results = lower_middle_income_co2_data.corrwith(lower_middle_income_forest_data, axis=1)
upper_middle_income_correlation_results = upper_middle_income_co2_data.corrwith(upper_middle_income_forest_data, axis=1)

# Print correlation results for each income group
print("\nCorrelation between CO2 emissions and Forest area for rich countries for last 20 year data:")
print(rich_correlation_results)

print("\nCorrelation between CO2 emissions and Forest area for poor countries for last 20 year data:")
print(poor_correlation_results)

print("\nCorrelation between CO2 emissions and Forest area for lower middle-income countries for last 20 year data:")
print(lower_middle_income_correlation_results)

print("\nCorrelation between CO2 emissions and Forest area for upper middle-income countries for last 20 year data:")
print(upper_middle_income_correlation_results)

# Extract correlation values
rich_corr_value = rich_correlation_results.values[0]
poor_corr_value = poor_correlation_results.values[0]
lower_middle_income_corr_value = lower_middle_income_correlation_results.values[0]
upper_middle_income_corr_value = upper_middle_income_correlation_results.values[0]

# Extract country names
rich_country = rich_correlation_results.index[0][0]
poor_country = poor_correlation_results.index[0][0]
lower_middle_income_country = lower_middle_income_correlation_results.index[0][0]
upper_middle_income_country = upper_middle_income_correlation_results.index[0][0]

# Plotting the bar chart for correlation between CO2 emissions and Forest area
plt.figure(figsize=(10, 6))

countries = [rich_country, poor_country, lower_middle_income_country, upper_middle_income_country]
correlation_values = [rich_corr_value, poor_corr_value, lower_middle_income_corr_value, upper_middle_income_corr_value]

plt.bar(countries, correlation_values, color=['blue', 'red', 'green', 'orange'])

plt.xlabel('Countries')
plt.ylabel('Correlation')
plt.title('Correlation between CO2 Emissions and Forest area for last 20 year data')

plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.grid(axis='y')

plt.tight_layout()
plt.show()

# Select relevant columns for CO2 emissions and Urban population
Urban_population_data = updated_data[updated_data['Indicator Name'] == 'Urban population (% of total population)'][['Country Name', 'IncomeGroup'] + last20_years]

# Set 'Country Name' and 'IncomeGroup' as index
Urban_population_data.set_index(['Country Name', 'IncomeGroup'], inplace=True)

rich_Urban_population_data = Urban_population_data.loc[rich]
poor_Urban_population_data = Urban_population_data.loc[poor]
lower_middle_income_Urban_population_data = Urban_population_data.loc[lower_middle_income]
upper_middle_income_Urban_population_data = Urban_population_data.loc[upper_middle_income]

# Calculate correlation using corrwith()
rich_correlation_results = rich_co2_data.corrwith(rich_Urban_population_data, axis=1)
poor_correlation_results = poor_co2_data.corrwith(poor_Urban_population_data, axis=1)
lower_middle_income_correlation_results = lower_middle_income_co2_data.corrwith(lower_middle_income_Urban_population_data, axis=1)
upper_middle_income_correlation_results = upper_middle_income_co2_data.corrwith(upper_middle_income_Urban_population_data, axis=1)

# Print correlation results for each income group
print("\nCorrelation between CO2 emissions and Urban population (% of total population) for rich countries for last 20 year data:")
print(rich_correlation_results)

print("\nCorrelation between CO2 emissions and Urban population (% of total population) for poor countries for last 20 year data:")
print(poor_correlation_results)

print("\nCorrelation between CO2 emissions and Urban population (% of total population) for lower middle-income countries for last 20 year data:")
print(lower_middle_income_correlation_results)

print("\nCorrelation between CO2 emissions and Urban population (% of total population) for upper middle-income countries for last 20 year data:")
print(upper_middle_income_correlation_results)

# Extract correlation values
rich_corr_value = rich_correlation_results.values[0]
poor_corr_value = poor_correlation_results.values[0]
lower_middle_income_corr_value = lower_middle_income_correlation_results.values[0]
upper_middle_income_corr_value = upper_middle_income_correlation_results.values[0]

# Extract country names
rich_country = rich_correlation_results.index[0][0]
poor_country = poor_correlation_results.index[0][0]
lower_middle_income_country = lower_middle_income_correlation_results.index[0][0]
upper_middle_income_country = upper_middle_income_correlation_results.index[0][0]

# Plotting the bar chart for correlation between CO2 emissions and Urban population
plt.figure(figsize=(10, 6))

countries = [rich_country, poor_country, lower_middle_income_country, upper_middle_income_country]
correlation_values = [rich_corr_value, poor_corr_value, lower_middle_income_corr_value, upper_middle_income_corr_value]

plt.bar(countries, correlation_values, color=['blue', 'red', 'green', 'orange'])

plt.xlabel('Countries')
plt.ylabel('Correlation')
plt.title('Correlation between CO2 Emissions and Urban population (% of total population) for last 20 year data')

plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.grid(axis='y')

plt.tight_layout()
plt.show()