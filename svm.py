import random

import numpy as np
from sklearn import svm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def generate_points(n=20):
    return [(random.randint(-20, 0), random.randint(-20, 20)) for _ in range(n // 2)] + \
           [(random.randint(0, 20), random.randint(-20, 20)) for _ in range(n // 2)], \
           [0 for _ in range(n // 2)] + [1 for _ in range(n // 2)]


def print_lines(X, y, clf):

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none')
    plt.show()


def main():
    points, points_class = generate_points()
    plt.scatter([i[0] for i in points], [i[1] for i in points], c=points_class)
    plt.show()

    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(points, points_class)

    print_lines(np.array(points), np.array(points_class), clf)

    new_point = [10, 10]
    points.append(new_point)
    y = clf.predict([new_point])
    points_class.append(y[0])
    print_lines(np.array(points), np.array(points_class), clf)


if __name__ == '__main__':
    main()
