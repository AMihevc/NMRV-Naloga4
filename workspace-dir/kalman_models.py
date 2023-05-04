import numpy as np
import math
import matplotlib.pyplot as plt
from ex4_utils import kalman_step, gaussian_prob, derive_input_matrike

# function that returns the matrices based on the model type
def get_matrices(model_type, r, q):
    # INPUTS:
    # model_type - a string that can be 'RW', 'NCV', or 'NCA'
    # r - the number of dimensions of the state vector??? (not sure)
    # q - the number of dimensions of the observation vector??? (not sure)
    # OUTPUTS:
    # Fi_matrika - the system matrix
    # H - the observation matrix
    # Q - the system noise covariance
    # R - the observation noise covariance

    # Random Walk (RW)
    if model_type == 'RW':
        F = np.array([[0, 0], 
                       [0, 0]], dtype=np.float32)

        L = np.array([[1, 0], 
                      [0, 1]], dtype=np.float32)

        H = np.array([[1, 0], 
                      [0, 1]], dtype=np.float32)

        R = r * np.array([[1, 0], 
                      [0, 1]], dtype=np.float32)

    # Nearly-Constant Velocity (NCV) 
    elif model_type == 'NCV':
        F = np.array([[0, 0, 1, 0], 
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]], dtype=np.float32)
        
        L = np.array([[0, 0],
                      [0, 0],
                      [1, 0], 
                      [0, 1]], dtype=np.float32)

        H = np.array([[1, 0, 0, 0], 
                      [0, 1, 0, 0]], dtype=np.float32)

        R = r * np.array([[1, 0], 
                          [0, 1]], dtype=np.float32)

    # Nearly-Constant Acceleration (NCA) 
    elif model_type == 'NCA':
        F = np.array([ [0, 0, 1, 0, 0, 0], 
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]], dtype=np.float32)
        
        L = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [1, 0], 
                      [0, 1]], dtype=np.float32)

        H = np.array([[1, 0, 0, 0, 0, 0], 
                      [0, 1, 0, 0, 0, 0]], dtype=np.float32)

        R = r * np.array([[1, 0], 
                          [0, 1]], dtype=np.float32)
    else:
        raise ValueError('Unknown model type')
    
    #derive Fi and L to get the transition matrix
    Fi_matrika, Q = derive_input_matrike(F, L, q)

    # return the matrices   
    return Fi_matrika, H, Q, R

# function implements Random Walk (RW), 
# Nearly-Constant Velocity (NCV), 
# and Nearly-Constant Acceleration (NCA) models using Kalman filter 
def plot_smoothing(model, r, q, ax):
    # INPUTS:
    # model - a string that can be 'RW', 'NCV', or 'NCA'
    # r - measurement uncertainty
    # q - model uncertainty
    # ax - axes to plot on

    #trajectory generation
    N = 50                               # number of time steps in trajectory
    v = np.linspace(5*math.pi, 0, N)    # angular velocity

    # Trajectory 1
    x = np.cos(v)*v
    y = np.sin(v)*v               

    # Trajectory 2 (rectangle)
    # x = np.array([0, 10, 10, 0, 0])
    # y = np.array([0, 0, 5, 5, 0])

    # Trajectory 3 (rectangular "man")
    # x = np.array([0, 5, 5, 3, 3, 7, 7, 5, 5, 10, 10, 8, 8, 12, 12, 7, 7, 0])
    # y = np.array([0, 0, 2, 2, 5, 5, 7, 7, 9, 9, 7, 7, 5, 5, 2, 2, 0, 0])

    # Trajectory 4 (P)
    x = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    y = np.array([0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7])

    # Trajectory 7 (some random jaged edges)
    # t = np.linspace(0, 2*np.pi, 100)
    # x = np.concatenate([np.linspace(0, 10, 20), 10 + 2*np.sin(t), np.linspace(10, 0, 20), 0 + 2*np.sin(t)])
    # y = np.concatenate([np.zeros(20), np.linspace(0, 10, 50), 10 + 2*np.cos(t), np.linspace(10, 0, 50)])

    # # Make sure x and y have the same length
    # x = x[:len(y)]

    #Trajectory 8 (heart)

    # # Define the x values for the heartbeat shape
    # x = np.linspace(0, 4*np.pi, 1000)

    # # Define the y values for the heartbeat shape
    # y = np.sin(x) + np.sin(3*x) + np.sin(5*x)

    # # Scale and shift the y values to create a more interesting path
    # y = 5*y + 10

    # # Add some random noise to the path
    # y += np.random.normal(scale=0.5, size=len(x))

    # # Add some jaggedness to the path
    # y += np.random.normal(scale=0.25, size=len(x)).cumsum()

    # # Make sure the y values stay within a certain range
    # y = np.clip(y, 0, 20)


    # Get the matrices
    A, C, Q, R = get_matrices(model, r, q)

    #Code from instructions 
    sx = np.zeros(( x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros(( y.size, 1), dtype=np.float32).flatten()
    sx[0] = x[0]
    sy[0] = y[0]
    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x[0]
    state[1] = y[0]
    covariance = np.eye (A.shape[0] , dtype=np . float32 )

    for j in range(1, x.size):
        #print(j)
        state, covariance, _, _ = kalman_step( A, C, Q, R, np.reshape( np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)

        sx[j] = state[0]
        sy[j] = state[1]

    #Plotting
    # set the title and labels of the plot acording to the model q and r
    ax.set_title('Model: ' + model + ', q: ' + str(q) + ', r: ' + str(r))

    # plot the trajectories
    ax.plot(x, y, 'r', label='original')
    ax.plot(sx, sy, 'b', label='smoothed')


# TODO 
# Test the implementations using some artificially generated trajectories 
# and experiment with Kalman filter parameters.

figure, axes = plt.subplots(3, 4, figsize=(17, 14))
#title for figure
figure.suptitle('Kalman filter smoothing', fontsize=16)

# Random Walk (RW)
plot_smoothing('RW', 1, 100, axes[0,0])
plot_smoothing('RW', 1, 10, axes[0,1])
plot_smoothing('RW', 10, 1, axes[0,2])
plot_smoothing('RW', 100, 1, axes[0,3])

# Nearly-Constant Velocity (NCV)
plot_smoothing('NCV', 1, 100, axes[1,0])
plot_smoothing('NCV', 1, 10, axes[1,1])
plot_smoothing('NCV', 10, 1, axes[1,2])
plot_smoothing('NCV', 100, 1, axes[1,3])

# Nearly-Constant Acceleration (NCA)
plot_smoothing('NCA', 1, 100, axes[2,0])
plot_smoothing('NCA', 1, 10, axes[2,1])
plot_smoothing('NCA', 10, 1, axes[2,2])
plot_smoothing('NCA', 100, 1, axes[2,3])

#show the legend in top left corrner
axes[0,0].legend(loc='upper left')



plt.show()
