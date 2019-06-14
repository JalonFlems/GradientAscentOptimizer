#!/usr/bin/env python
# coding: utf-8

# # MAT232 - Computer Programing Challenge (Week #5) 

# #### Created by: Micah Flemming 

# ##### Imports 

import sys
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import sympy as sy
import math, random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify, implemented_function

if __name__ == "__main__":
    
# ##### Initialization

# Initialize the symbols
    x, y = sy.symbols('x y')

    while True:    
        
        # Retrieve the function from the user
        user_inp = input("Enter a function in terms of x and y. Don't forget to put any operations in between the variables (e.g. x**2 + y**2): ") 
        
        # Parse the function
        f_xy = parse_expr(user_inp)
        
        
        # ##### Gradient Ascent Optimization
        
        values = input("Enter the integer values for x and y seperated by spaces (e.g. 1 2): ")
        inp_x, inp_y = values.split(" ")
        
        # 1 - Use the given point on the curve, p
        x_o, y_o = int(inp_x), int(inp_y)
        point = np.array([x_o, y_o], dtype=float)
        
        # initialize increase and step (used for updates)
        step, increase = 0, 0.001
        
        # initialize arrays to keep track of points
        x_points, y_points, z_points = [], [], []
        
        # initialize grad_p_x and grad_p_y 
        grad_p_x, grad_p_y = float("inf"), float("inf")
        
        
        while abs(float(grad_p_x)) > 0.001 and abs(float(grad_p_y)) > 0.001:
            
            # 2 - Find the gradient vector
            f_xy_x = sy.diff(f_xy, x) # derivative with respect to x
            f_xy_y = sy.diff(f_xy, y) # derivative with respect to y
        
            # 3 - Evaluate the gradient vector at point, p.
            grad_p_x = f_xy_x.subs([(x, point[0]), (y, point[1])])
            grad_p_y = f_xy_y.subs([(x, point[0]), (y, point[1])])
            
            grad_point = np.array([float(grad_p_x), float(grad_p_y)])
            
            # 4 - Find the direction vector of grad_point
            dir_vec = (1 / np.linalg.norm(grad_point)) * grad_point
            
            # 5 - Travel in the direction of dir_vec
            point += dir_vec * increase
            
            step += 1
            
            if step % 50 == 0:
                x_points.append(point[0])
                y_points.append(point[1])
                z_points.append(f_xy.subs([(x, point[0]), (y, point[1])]))
                
            if step > 100000: break
                
        # ##### Graphing 
        
        print("Function: {} | Initial Point: {} | Approximate Maximum: {}".format(f_xy, values, point))
        
        # Convert the function into a lambda function
        f_xy_lambda = lambdify([x, y], f_xy, 'numpy')
        
        h, g = np.linspace(-10, 10, 30), np.linspace(-10, 10, 30)
        
        X, Y = np.meshgrid(h, g)
        
        Z = f_xy_lambda(X, Y)
        
        
        # Below is the path that was travelled by executing the given algorithm and the graph of the function 
        
        
        # Plot the line
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1, projection = '3d')
        ax.plot3D(x_points, y_points, z_points)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        # Plot the function 
        ax = fig.add_subplot(2, 1, 2, projection = '3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.show()
        
        #del x_o, y_o, f_xy, f_xy_x, f_xy_y, f_xy_lambda
        
        user_inp = input("Would you like to enter another function?")
        
        if user_inp.lower() == 'no':
            sys.exit()
        
        # ###### References
        
        # 3D plots as subplots: https://matplotlib.org/3.1.0/gallery/mplot3d/subplot3d.html
        # 
        # Three-dimensional Plotting: https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
        
        