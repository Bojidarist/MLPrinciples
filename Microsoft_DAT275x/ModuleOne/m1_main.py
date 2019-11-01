import pandas as pd # To make data frame
from sklearn.preprocessing import scale # To scale the data
import seaborn as sns # For plotting
import matplotlib.pyplot as plt # For plotting
from sklearn import datasets # Get dataset from sklearn
from sklearn.model_selection import train_test_split # For splitting data
import numpy as np # Numpy
from sklearn.neighbors import KNeighborsClassifier # For using a model

# Import the dataset from sklearn.datasets
iris = datasets.load_iris()

# Create data frame from the dictionary
species = []
# iris.target is [0, 1, 2, 1, 2 ...]
for x in iris.target:
    # We select a specie like iris.target_names[0]
    species.append(iris.target_names[x])

# Makes the data from iris["data"] and maked DataFrame like in R
iris = pd.DataFrame(iris["data"], columns = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"])
# Add a new column named Species with the data from species
iris["Species"] = species
# Print the data types (like str() in R)
# Sepal_Length    float64
# Sepal_Width     float64
# Petal_Length    float64
# Petal_Width     float64
# Species          object
# dtype: object
print(iris.dtypes)

#             Count
# Species
# setosa         50
# versicolor     50
# virginica      50
iris["Count"] = 1
print(iris[["Species", "Count"]].groupby("Species").count())

# Plots the data from iris
def plot_iris(iris, col1, col2):
    sns.lmplot(x = col1, y = col2, data = iris, hue = "Species", fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("Iris species shown by color")
    plt.show()

# Plot the data
plot_iris(iris, "Petal_Width", "Sepal_Length")
plot_iris(iris, "Sepal_Width", "Sepal_Length")

# Scale the numeric values of the features.
# It is important that numeric features used to train
# machine learning models have a similar range of values.
# Otherwise, features which happen to have large numeric
# values may dominate model training, even if other features
# with smaller numeric values are more informative.
# In this case Zscore normalization is used.
# This normalization process scales each feature so that
# the mean is 0 and the variance is 1.0.

# Split the dataset into randomly sampled training and evaluation data sets.
# The random selection of cases seeks to limit the leakage of information
# between the training and evaluation cases.

# Scale the data
num_cols = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
iris_scaled = scale(iris[num_cols])
iris_scaled = pd.DataFrame(iris_scaled, columns = num_cols)
#        Sepal_Length  Sepal_Width  Petal_Length  Petal_Width
# count       150.000      150.000       150.000      150.000
# mean         -0.000       -0.000        -0.000       -0.000
# std           1.003        1.003         1.003        1.003
# min          -1.870       -2.434        -1.568       -1.447
# 25%          -0.901       -0.592        -1.227       -1.184
# 50%          -0.053       -0.132         0.336        0.133
# 75%           0.675        0.559         0.763        0.791
# max           2.492        3.091         1.786        1.712
print(iris_scaled.describe().round(3))

# The methods in the scikit-learn package requires numeric numpy arrays as arguments.
# Therefore, the strings indicting species must be re-coded as numbers.
# The code in the cell below does this using a dictionary lookup.
levels = { "setosa": 0, "versicolor": 1, "virginica": 2 }
iris_scaled["Species"] = [levels[x] for x in iris["Species"]]
#    Sepal_Length  Sepal_Width  Petal_Length  Petal_Width  Species
# 0     -0.900681     1.019004     -1.340227    -1.315444        0
# 1     -1.143017    -0.131979     -1.340227    -1.315444        0
# 2     -1.385353     0.328414     -1.397064    -1.315444        0
# 3     -1.506521     0.098217     -1.283389    -1.315444        0
# 4     -1.021849     1.249201     -1.340227    -1.315444        0
print(iris_scaled.head())

# Split the data into a training and test set by Bernoulli sampling
# Set the numpy random seed to something so we get consistent results
np.random.seed(3456)
# Split the data into two matrices with size 75 each
iris_split = train_test_split(np.asmatrix(iris_scaled), test_size = 75)
# The first matrix is for training
iris_train_features = iris_split[0][:, :4]
# numpy.ravel makes multiple arrays into one -> [1,2] & [3,4] => [1,2,3,4]
iris_train_labels = np.ravel(iris_split[0][:, 4])
# The second matrix is for testing
iris_test_features = iris_split[1][:, :4]
iris_test_labels = np.ravel(iris_split[1][:, 4])

# Make a KNeighborsClassifier and 'fit' it
KNN_model = KNeighborsClassifier(n_neighbors = 3)
KNN_model.fit(iris_train_features, iris_train_labels)

# Compute accuracy
iris_test = pd.DataFrame(iris_test_features, columns = num_cols)
iris_test['predicted'] = KNN_model.predict(iris_test_features)
iris_test['correct'] = [1 if x == z else 0 for x, z in zip(iris_test['predicted'], iris_test_labels)]
accuracy = 100.0 * float(sum(iris_test['correct'])) / float(iris_test.shape[0])
print(accuracy)

# Output a prediction
species_names = ["setosa", "versicolor", "virginica"]
iris_result = KNN_model.predict(iris_test_features[2])
iris_result_index = int(iris_result[0])
print(species_names[iris_result_index])