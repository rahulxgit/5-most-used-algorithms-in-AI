\documentclass{article}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{color}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}

\lstset{style=mystyle}

\title{5 Most Used Algorithms in AI}
\author{harsh tripathi [ 22111020]}
\date{19 july 2024}

\begin{document}

\maketitle

\section{Introduction}
This document explains five of the most commonly used algorithms in Artificial Intelligence, complete with figures and Python code examples.

\section{1. Linear Regression}
Linear regression is a simple yet powerful algorithm used for predicting a continuous outcome based on one or more input features.

\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{linear_regression_figure.png}
\caption{Linear Regression}
\label{fig:linear_regression}
\end{figure}

Python implementation:

\begin{lstlisting}[language=Python]
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_new = np.array([[6]])
y_pred = model.predict(X_new)

print(f"Prediction for X=6: {y_pred[0]}")
\end{lstlisting}

\section{2. K-Means Clustering}
K-Means is an unsupervised learning algorithm used for clustering data points into K groups based on their similarities.

\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{kmeans_figure.png}
\caption{K-Means Clustering}
\label{fig:kmeans}
\end{figure}

Python implementation:

\begin{lstlisting}[language=Python]
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
X = np.random.rand(100, 2)

# Create and fit the model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.show()
\end{lstlisting}

\section{3. Decision Trees}
Decision Trees are versatile algorithms used for both classification and regression tasks, making decisions based on asking a series of questions.

\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{decision_tree_figure.png}
\caption{Decision Tree}
\label{fig:decision_tree}
\end{figure}

Python implementation:

\begin{lstlisting}[language=Python]
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
\end{lstlisting}

\section{4. Neural Networks}
Neural Networks, inspired by biological neural networks, are powerful algorithms capable of learning complex patterns in data.

\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{neural_network_figure.png}
\caption{Neural Network}
\label{fig:neural_network}
\end{figure}

Python implementation (using TensorFlow):

\begin{lstlisting}[language=Python]
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate dummy data
import numpy as np
X = np.random.random((1000, 10))
y = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
\end{lstlisting}

\section{5. Support Vector Machines (SVM)}
SVMs are powerful algorithms used for classification and regression, particularly effective in high-dimensional spaces.

\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{svm_figure.png}
\caption{Support Vector Machine}
\label{fig:svm}
\end{figure}

Python implementation:

\begin{lstlisting}[language=Python]
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a random dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
\end{lstlisting}

\end{document}
