# MAT232 - Computer Programing Challenge (Week #5) 

#------------------------ Imports -----------------------------

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
from mpl_toolkits.mplot3d import Axes3D
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify, implemented_function

#--------------------------------------------------------------

#---------------------- Global Variables ----------------------

AXIS_SPAN = 10
INCREASE = 0.01
MAX_ITERS = 10000

#--------------------------------------------------------------

#------------------------- Functions --------------------------
def Gradient_Ascent():
    pass


#--------------------------------------------------------------

if __name__ == "__main__":

#---------------------- Initialization ------------------------

    x, y = sy.symbols('x y')  # Initialize the symbols
    
#--------------------------------------------------------------

#------------------------ User Input  -------------------------

    while True:    
        
        # Retrieve the function from the user
        user_inp = input("Enter a function in terms of x and y. Don't forget to put any operations in between the variables (e.g. x**2 + y**2): ") 
        
        # Parse the function
        f_xy = parse_expr(user_inp)
         
        values = input("Enter the integer values for x and y seperated by spaces (e.g. 1 2): ")
        inp_x, inp_y = values.split(" ")
 
#--------------------------------------------------------------        

#-------------  Gradient Ascent Optimization  -----------------

        # 1 - Use the given point on the curve, p
        x_o, y_o = int(inp_x), int(inp_y)
        point = np.array([x_o, y_o], dtype = float)
        
        
        step = 0  # initialize step to help keep track of iterations 
        
        x_points, y_points, z_points = [], [], []  # initialize arrays to keep track of points
        
        grad_p_x, grad_p_y = float("inf"), float("inf")  # initialize grad_p_x and grad_p_y 
        
        
        while abs(float(grad_p_x)) > 0.001 and abs(float(grad_p_y)) > 0.001:
            
            # 2 - Find the gradient vector (i.e. the derivatives with respect to x and y)
            f_xy_x, f_xy_y = sy.diff(f_xy, x), sy.diff(f_xy, y) 
        
            # 3 - Evaluate the gradient vector at point, p.
            grad_p_x, grad_p_y = f_xy_x.subs([(x, point[0]), (y, point[1])]), f_xy_y.subs([(x, point[0]), (y, point[1])])

            grad_point = np.array([float(grad_p_x), float(grad_p_y)])
            
            # 4 - Find the direction vector of grad_point
            dir_vec = (1 / np.linalg.norm(grad_point)) * grad_point
            
            # 5 - Travel in the direction of dir_vec
            point += dir_vec * INCREASE
           
            step += 1
            
            if step % 50 == 0:
                x_points.append(point[0])
                y_points.append(point[1])
                z_points.append(f_xy.subs([(x, point[0]), (y, point[1])]))
                
            if step > MAX_ITERS: break         
#--------------------------------------------------------------

#-------------------------- Graphing  -------------------------

        print("Function: {} | Initial Point: {} | Approximate Maximum: {}".format(f_xy, values, point))
        
        # Convert the function into a lambda function
        f_xy_lambda = lambdify([x, y], f_xy, 'numpy')
        
        h, g = np.linspace(-AXIS_SPAN, AXIS_SPAN, 30), np.linspace(-AXIS_SPAN, AXIS_SPAN, 30)
        
        X, Y = np.meshgrid(h, g)
        
        Z = f_xy_lambda(X, Y)
        
        
        # Below is the path that was travelled by executing the given algorithm and the graph of the function 
        
        # TODO: Name the plot

        # Plot the line and the function
        fig = plt.figure()         
        ax = fig.add_subplot(1, 1, 1, projection = '3d')
        ax.plot3D(x_points, y_points, z_points, 'red')
        #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='magma', edgecolor='none')
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.show()

#--------------------------------------------------------------

#------------------------- User Input -------------------------
        user_inp = input("Would you like to enter another function?")
        
        if user_inp.lower() == 'no':
            sys.exit() 

#--------------------------------------------------------------
