%matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()

# Converting to dataframe for nice printing and use the "feature_names" attribute

pd.DataFrame({'feature name': iris.feature_names})
#1
pd.DataFrame({'target name': iris.target_names})
#2

# Convert to pandas df (adding real column names) to use some pandas functions (head, describe...)
iris.df = pd.DataFrame(iris.data, 
                       columns = iris.feature_names)

# First few rows
iris.df.head()
#3

# Summary stats
iris.df.describe()
#4



iris.df['target'] = iris.target

# A bit of rearrangement for plotting
df = iris.df.loc[:, ['sepal length (cm)', 'target']]

# Add an index column which indicates index within a class
df['idx'] = list(range(0, 50)) * 3

# Rearrange to be rows of class values rather feature values for a sample
df = df.pivot(index = 'idx', columns = 'target')

# Convert back to an array
df = np.array(df)

# Plot a boxplot!
plt.boxplot(df, labels = iris.target_names)
plt.title('sepal length (cm)')
#5


# Using pairplot from seaborn is a quick way to see which features separate out our data
# Draw scatterplots for joint relationships and histograms for univariate distributions:
import seaborn as sns; sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g = sns.pairplot(iris)
#6


#Show different levels of a categorical variable by the color of plot elements
g = sns.pairplot(iris, hue="species")
#7

#Use a different color palette:
g = sns.pairplot(iris, hue="species", palette="husl")
#8

