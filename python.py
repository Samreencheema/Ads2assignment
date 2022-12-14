
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 7 21:16:40 2022

@author: Samreen Safdar 
"""


# Library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

'''
The aim is to Explore and understand any correlations 
between population growth and energy consumption indicators using graph.
First load the dataset using Pandas Library.
Then clean the dataset like remove None value.
Make a function of Line, Bar and heatmap graph function.
'''
# Load Dataset
df = pd.read_csv("P_Data_Extract_From_World_Development_Indicators_Data.csv")
df.head()

# Remove None Value
df = df.dropna(axis=0) 


# Rename Column of Dataset
df.rename(columns={'Country Name': 'country'}, inplace=True)
df.head(1)

# Divide dataset for Population growth & Energy Consumption

population_growth  = df.loc[df['Series Code'] == "SP.POP.GROW"]
energy_consumption = df.loc[df['Series Code'] == "EG.USE.ELEC.KH.PC"]

# Remove unnecessary column

population_growth.drop(['Country Code','Series Name', 'Series Code'], axis=1, inplace=True)
energy_consumption.drop(['Country Code','Series Name', 'Series Code'], axis=1, inplace=True)

population_growth.set_index('country', inplace=True)
energy_consumption.set_index('country', inplace=True)

'''
Graph Plot Functions
- These function get different parameters
- Dataframe 
- Countries Name 
- Years 
- Title of graph
- filename that saved in local file
'''

def graph_plot(df, countries, years, title, filename):
    # figure size is the graph image size
    # This is a simple graph.
    df.loc[countries, years].T.plot(figsize=(14, 8)) 
    title = plt.title(title) 
    plt.savefig(filename)
    plt.show()

def graph_plot2(df, countries, years, title,  filename):
    # figure size is the graph image size
    # This graph type is Area.
    df.loc[countries, years].T.plot(kind='area', figsize=(14, 8))
    title = plt.title(title)
    plt.savefig(filename)
    plt.show()

def graph_bar(df, country, years, title, filename):
    # figure size is the graph image size
    # This is Bar graph
    df.loc[country, years].plot(kind='bar',figsize=(14, 8))
    title = plt.title(title)
    plt.savefig(filename)
    plt.show()

def graph_barh(df, countries, years, title, filename):
    # figure size is the graph image size
    # This is Barh graph
    df.loc[countries, years].transpose().plot(kind='barh', figsize=(20, 14), stacked=False)
    title = plt.title(title)
    plt.savefig(filename)
    plt.show()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):

# Create a heatmap from a numpy array and two lists of labels.
#     - data - A 2D numpy array of shape (M, N).
#     - row_labels - A list or array of length M with the labels for the rows.
#     - col_labels - A list or array of length N with the labels for the columns.

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def graph_heatmap(df, countries, years, title, filename):
    fig, ax = plt.subplots(figsize=(7, 7))
    
    im, cbar = heatmap(df[years].T, df[years], countries, ax=ax,
                       cmap="YlGn", cbarlabel=title)
    plt.savefig(filename)
    fig.tight_layout()
    plt.show()



'''
Years & Country Information
- Year used 1980-2021
- Countries used 'Croatia', 'Ireland', 'Norway', 'Austria', 'Netherlands'
'''


# years = list(population_growth.columns)
years = ['1980 [YR1980]', '1985 [YR1985]', '1990 [YR1990]', '1995 [YR1995]', '2000 [YR2000]', 
         '2005 [YR2005]', '2010 [YR2010]', '2015 [YR2015]', '2020 [YR2020]',]
countries = ['Norway', 'Austria', 'Netherlands', 'Ireland']


# Population Growth Line Graph 
graph_plot(population_growth, countries, years, "Population Growth (Annual %)", "pg_1.jpg")

# Population Growth Bar Graph
graph_bar(population_growth, countries, years, 'Population Growth (Annual %) of Ireland', "pg_2.jpg")

# Population Growth HeatMap
graph_heatmap(population_growth, countries, years, "Population Growth (Annual %)", "pg_3.jpg")

# Energy Consumption Line Graph
graph_plot(energy_consumption, countries, years, 'Electric power consumption (kWh per capita)', "ec_1.jpg")

# Energy Consumption Bar Graph
graph_bar(energy_consumption, countries, years, 'Electric power consumption (kWh per capita)', "ec_2.jpg")

# Energy Consumption Heat Graph
graph_heatmap(energy_consumption, countries, years, "Electric power consumption (kWh per capita)", "ec_3.jpg")



population_growth[years].T

energy_consumption[years].T