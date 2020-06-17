import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

def view_data_segments(xs, ys):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

def polynomial_least_squares(x, y, n):
    X = np.column_stack([x**k for k in range(0,n+1)])
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    x_new = np.linspace(x.min(),x.max(),1000)
    y_new = np.polyval(np.flip(A),x_new)
    y_hat = np.polyval(np.flip(A),x)
    s = square_error(ys, y_hat)
    return x_new, y_new, s

def sin_least_squares(x, y):
    X = np.column_stack((np.ones(x.shape), np.sin(x)))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = A[0] + A[1]*np.sin(x_new)
    y_hat = A[0] + A[1]*np.sin(x)
    s = square_error(ys, y_hat)
    return x_new, y_new, s

def square_error(y, y_hat):
    return np.sum((y-y_hat)**2)

file = sys.argv[1]
xss, yss = load_points_from_file(file)
iteratable_x = [xss[i:i + 20] for i in range(0, len(xss), 20)]
iteratable_y = [yss[i:i + 20] for i in range(0, len(yss), 20)]
err_t = 0

fig, ax = plt.subplots()
for xs,ys in zip(iteratable_x, iteratable_y):
    xs2, ys2, err_sin = sin_least_squares(xs, ys)
    err_min = 10000000
    for i in range(1,4,2):
        xs1, ys1, err_poly = polynomial_least_squares(xs, ys, i)
        if i==1:
            xl1 = xs1
            yl1 = ys1
            err_lin = err_poly
        if err_poly < err_min:
            xp1 = xs1
            yp1 = ys1
            err_min = err_poly
    if err_lin < 1.2*err_poly and (err_lin < err_sin or err_poly < err_sin):
        err_t = err_t + err_lin
        if len(sys.argv) == 3:
            if sys.argv[2]=="--plot":
                ax.plot(xl1, yl1, 'b', lw=1)
    else:
        if err_min < err_sin:
            err_t = err_t + err_min
            if len(sys.argv) == 3:
                if sys.argv[2]=="--plot":
                    ax.plot(xp1, yp1, 'r', lw=1)
        else:
            err_t = err_t + err_sin
            if len(sys.argv) == 3:
                if sys.argv[2]=="--plot":
                    ax.plot(xs2, ys2, 'y', lw=1)

print(err_t)
if len(sys.argv) == 3:
    if sys.argv[2]=="--plot":
        view_data_segments(xss, yss)
