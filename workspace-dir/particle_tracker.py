import numpy as np
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram
from ex4_utils import sample_gauss, derive_input_matrike
from utils.tracker import Tracker
import cv2
import random
import math
import sympy as sp

# run the tracker by running the following command from the toolkit-dir:
# python evaluate_tracker.py --workspace_path ../workspace-dir --tracker particle_tracker

def get_matrices_fixed(model_type, r, q):
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
        T = sp.symbols('T')
        F = sp.Matrix([[0, 0], 
                       [0, 0]])
        
 

        L = sp.Matrix([[1, 0], 
                      [0, 1]])
        # intergrate L to get Q

        H = np.array([[1, 0], 
                      [0, 1]], dtype=np.float32)

        R = r * np.array([[1, 0], 
                      [0, 1]], dtype=np.float32)
        
        #intergrate F to get Fi
        Fi = sp.exp(F * T)
        Fi = Fi.subs(T, 1)
        
        # intergrate L to get Q
        Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
        Q = Q.subs(T, 1)

        Fi = np.array(Fi, dtype="float")
        Q = np.array(Q, dtype="float")

    # Nearly-Constant Velocity (NCV) 
    elif model_type == 'NCV':
        T = sp.symbols('T')
        F = sp.Matrix([[0, 0, 1, 0], 
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        
        L = sp.Matrix([[0, 0],
                      [0, 0],
                      [1, 0], 
                      [0, 1]])

        H = np.array([[1, 0, 0, 0], 
                      [0, 1, 0, 0]], dtype="float")

        R = r * np.array([[1, 0], 
                          [0, 1]], dtype="float")

        #intergrate F to get Fi
        Fi = sp.exp(F * T)
        Fi = Fi.subs(T, 1)
       
        # intergrate L to get Q
        Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
        Q = Q.subs(T, 1)

        Fi = np.array(Fi, dtype="float")
        Q = np.array(Q, dtype="float")

    # Nearly-Constant Acceleration (NCA) 
    elif model_type == 'NCA':
        T = sp.symbols('T')
        F = sp.Matrix([ [0, 0, 1, 0, 0, 0], 
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]])

        
        L = sp.Matrix([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1],
                      [0, 0], 
                      [0, 0]])
      

        H = np.array([[1, 0, 0, 0, 0, 0], 
                      [0, 1, 0, 0, 0, 0]], dtype=np.float32)

        R = r * np.array([[1, 0], 
                          [0, 1]], dtype=np.float32)

        #intergrate F to get Fi
        Fi = sp.exp(F * T)
        Fi = Fi.subs(T, 1)

        # intergrate L to get Q
        Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
        Q = Q.subs(T, 1)

        
        # convert Fi and Q to numpy arrays
        Fi = np.array(Fi, dtype="float")
        Q = np.array(Q, dtype="float")

    else:
        raise ValueError('Unknown model type')
    
    #derive Fi and L to get the transition matrix
    # Fi_matrika, Q = derive_input_matrike(F, L, q)

    # return the matrices   
    return Fi, Q, H, R

class ParticleTracker(Tracker):

    #name 
    def name(self):
        return "ParticleTracker"

    # Constructor
    def __init__(
            self,
            enlargement_factor=1.0,
            alpha_update=0.05,
            sigma_kernel=1,
            sigma_distance = 0.1,
            histogram_bins=16,
            number_of_particles=30,
            q_model_noise = 10,
            model='NCA',
        ):
        self.enlargement_factor = enlargement_factor
        self.alpha_update = alpha_update
        self.sigma_kernel = sigma_kernel
        self.sigma_distance = sigma_distance
        self.histogram_bins = histogram_bins
        self.number_of_particles = number_of_particles
        self.q_model_noise = q_model_noise
        self.model = model

        #matrices for motion model and kalman filter
        self.Fi_matrika, self.Q_covariance = None , None
        
        self.particles = None
        self.particles_state = None
        self.weights = None

        #template, search window, position and size
        self.template = None
        self.template_histogram = None
        self.position = None
        self.size = None
        self.original_size = None

        #kernel
        self.kernel = None
        self.kernel_size = None

    #function for getting the size of the search window
    def get_size(self, region, en_factor):
        x_size = round(region[2] * en_factor)
        y_size = round(region[3] * en_factor)
        if x_size % 2 == 0:
            x_size += 1
        if y_size % 2 == 0:
            y_size += 1
        return  (x_size, y_size)

    #function for initializing the tracker  
    def initialize(self, image, region: list):

        # # region = np.array(region).astype(np.int64)

        region[2] = math.floor(region[2])
        if region[2] % 2 == 0:
            region[2] += 1

        region[3] = math.floor(region[3])
        if region[3] % 2 == 0:
            region[3] += 1
        
        gor = max(region[1], 0)
        levo = max(region[0], 0)
        dol = int(region[1] + region[3])
        desno = int(region[0] + region[2])

        self.template = image[int(gor):int(dol), int(levo):int(desno)]
        
        #set original size of the search window
        self.original_size = self.get_size(region, 1)

        #get the size of the search window with the enlargement factor
        self.size = (region[2], region[3])

        #get position and size of the search window
        # self.search_window_position = (region[0] + region[2] / 2, region[1] + region[3] / 2) #set to center of search window
        self.position = np.array([int(region[0] + region[2] / 2), int(region[1] + region[3] / 2)])
        
        patch, _ = get_patch(image, self.position, self.size)
        #create the kernel for visual model
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], 1)
        self.kernel_size = self.kernel.shape

        self.template = extract_histogram(patch, self.histogram_bins, self.kernel)
        self.template = self.template / np.sum(self.template)

        #set the matrices for the motion model
        self.Fi_matrika, self.Q_covariance, self.C, self.R = get_matrices_fixed(self.model, 1, self.q_model_noise)

        #initialize particles using gausian distribution 
        self.particles = np.zeros((self.number_of_particles, self.Q_covariance.shape[0]))
        self.particles[:, :2] = sample_gauss(self.position, self.Q_covariance[:2, :2], self.number_of_particles)
        self.weights = np.ones(self.number_of_particles) # weights are initialized to 1

        #Print all the settings
        # print("initialized")
        # print(f"model: {self.model}")
        # print(f"enlargement_factor: {self.enlargement_factor}")
        # print(f"alpha_update: {self.alpha_update}")
        # print(f"sigma_kernel: {self.sigma_kernel}")
        # print(f"sigma_distance: {self.sigma_distance}")
        # print(f"histogram_bins: {self.histogram_bins}")
        # print(f"number_of_particles: {self.number_of_particles}")
        # print(f"q_model_noise: {self.q_model_noise}")
        # print(f"Fi_matrika: {self.Fi_matrika}")
        # print(f"Q_covariance: {self.Q_covariance}")
        # print(f"kernel: {self.kernel}")
        # print(f"kernel_size: {self.kernel_size}")
        # print(f"template: {self.template}")
        # print(f"template_histogram: {self.template_histogram}")
        # print(f"search_window_position: {self.search_window_position}")
        # print(f"search_window_size: {self.search_window_size}")
        # print(f"original_size: {self.original_size}")
        # print(f"particles_state: {self.particles_state}")
        # print(f"particles: {self.particles}")
        # print(f"weights: {self.weights}")
        # print("----------")

    #code from instructions 
    def resample_particles(self):
        weights_normalized = self.weights / np.sum(self.weights)
        weights_cumsum = np.cumsum(weights_normalized)
        random_samples = np.random.rand(self.number_of_particles, 1 )
        sampled_indexes = np.digitize(random_samples, weights_cumsum)
        new_particles = self.particles[sampled_indexes.flatten(), :]
        return new_particles

    #hellinger distance function 
    def hellinger_distance(self, p, q):
        return 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))

    #function for Histogram extraction and normalization
    def extract_norm_histogram(self, patch):

        #get the histogram of the patch        
        histogram = extract_histogram(patch, self.histogram_bins, self.kernel)

        #normalize histogram
        histogram = histogram / np.sum(histogram)

        return histogram
    
    #function for istogram extraction
    def extract_histogram_from_image(self, image):
        #get the patch from the image
        patch, _ = get_patch(image, self.position, self.size)

        #get the histogram of the patch
        histogram = extract_histogram(patch, self.histogram_bins, self.kernel)

        #normalize histogram
        histogram = histogram / np.sum(histogram)

        return histogram
    
    #function for calculating the weights
    def calculate_weights(self, image):

        for i in range(self.number_of_particles):
            x_coordinate = self.particles[i, 0]
            y_coordinate = self.particles[i, 1]
            
            #check if particle is out of bounds
            if x_coordinate > image.shape[1] or y_coordinate > image.shape[0] or x_coordinate < 0 or y_coordinate < 0:
                self.weights[i] = 0
                # print("particle out of bunds")
            else:
                temp = self.template / np.sum(self.template)
                # print("this is temp: ", temp.shape)
                # print("position, size", (x_direction, y_direction), self.size)
                patch, _ = get_patch(image, (x_coordinate, y_coordinate), self.size)

                #get the normilized histogram of the patch
                histogram = self.extract_norm_histogram(patch)

                # hellinger = np.linalg.norm(np.sqrt(histogram) - np.sqrt(temp)) / np.sqrt(2)
                hellinger_fun = self.hellinger_distance(histogram, temp)
                self.weights[i] = np.exp(- 0.5 * (hellinger_fun ** 2 / self.sigma_distance ** 2))
        
        #normalize weights
        self.weights = self.weights / np.sum(self.weights) + 0.0001

    #function for tracking the object
    def track(self, image):

            
        # replace the particles by sampling new particles based on the weight distribution
        new_particles = self.resample_particles()

        # predict the particles using the motion model
        noise = sample_gauss(np.zeros(self.Q_covariance.shape[0]), self.Q_covariance, self.number_of_particles)
        
        self.particles = (self.Fi_matrika @ new_particles.T).T + noise
        #self.particles = np.matmul(self.Fi_matrika, new_particles.T).T + noise

        # recalculating the weights based on visiual model similarity
        self.calculate_weights(image)
        
        # COMPUTE NEW STATE ESTIMATE

        #self.particles_state = np.matmul( np.transpose(self.particles), weights_normalized)
        self.position = self.weights.T @ self.particles[:, :2]    

        patch, _ = get_patch(image, self.position, self.size) #flag

        histogram_fun = self.extract_norm_histogram(patch)

        # Update the template
        self.template = (1 - self.alpha_update) * self.template + self.alpha_update * histogram_fun

        left = max(self.position[0] - self.size[0] / 2, 0)
        top = max(self.position[1] - self.size[1] / 2, 0)

        return [left, top, self.size[0], self.size[1]]
    
#define same tracker but with different parameters
class ParticleTrackerRW(ParticleTracker):

    #override the name function 
    def name(self):
        return "ParticleTrackerRW"
    
    #override the init function
    # Constructor
    def __init__(
            self,
            enlargement_factor=1.0,
            alpha_update=0.05,
            sigma_kernel=1,
            sigma_distance = 0.1,
            histogram_bins=16,
            number_of_particles=30,
            q_model_noise = 10,
            model='RW',
        ):
        self.enlargement_factor = enlargement_factor
        self.alpha_update = alpha_update
        self.sigma_kernel = sigma_kernel
        self.sigma_distance = sigma_distance
        self.histogram_bins = histogram_bins
        self.number_of_particles = number_of_particles
        self.q_model_noise = q_model_noise
        self.model = model

        #matrices for motion model and kalman filter
        self.Fi_matrika, self.Q_covariance = None , None
        
        self.particles = None
        self.particles_state = None
        self.weights = None

        #template, search window, position and size
        self.template = None
        self.template_histogram = None
        self.position = None
        self.size = None
        self.original_size = None

        #kernel
        self.kernel = None
        self.kernel_size = None


class ParticleTrackerNCV(ParticleTracker):

    #override the name function 
    def name(self):
        return "ParticleTrackerNCV"
    
    #override the init function
    # Constructor
    def __init__(
            self,
            enlargement_factor=1.0,
            alpha_update=0.05,
            sigma_kernel=1,
            sigma_distance = 0.1,
            histogram_bins=16,
            number_of_particles=30,
            q_model_noise = 10,
            model='NCV',
        ):
        self.enlargement_factor = enlargement_factor
        self.alpha_update = alpha_update
        self.sigma_kernel = sigma_kernel
        self.sigma_distance = sigma_distance
        self.histogram_bins = histogram_bins
        self.number_of_particles = number_of_particles
        self.q_model_noise = q_model_noise
        self.model = model

        #matrices for motion model and kalman filter
        self.Fi_matrika, self.Q_covariance = None , None
        
        self.particles = None
        self.particles_state = None
        self.weights = None

        #template, search window, position and size
        self.template = None
        self.template_histogram = None
        self.position = None
        self.size = None
        self.original_size = None

        #kernel
        self.kernel = None
        self.kernel_size = None