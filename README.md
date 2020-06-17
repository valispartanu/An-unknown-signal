# An unknown signal
###This python program uses the matrix form of least squares regression in order to estimate a model given a set of CSV train files. Each file contains several line segments (each made up of 20 points). The program determines the function type of each line segment (eg. linear/polynomial/unknown), produces the total reconstruction error for that file and presents a figure showing the reconstructed line from the points if an optional argument is given. The main challenge was to find the right polynomial degree in order to avoid overfitting. 
###Cross validation might have been a sensible solution, but having a relatively small number of datapoints, splitting them in train and test points would have resulted in a wrong residual error. After some testing, a threshold was imposed. This forces the program to choose a smaller polynomial degree over the one that was overfitting. 
###More info can be found in the report.
#Usage
###Command to use for file eg. adv_3:
###python code.py train_data/adv_3.csv --plot
