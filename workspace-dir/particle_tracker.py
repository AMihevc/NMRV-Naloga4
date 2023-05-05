import numpy as np
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram
from ex4_utils import sample_gauss, derive_input_matrike
from utils.tracker import Tracker



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

class ParticleTracker(Tracker):

    #name 
    def name(self):
        return "ParticleTracker"

    # Constructor
    def __init__(
            self,
            enlargement_factor=2.0,
            alpha_update=0.5,
            sigma_kernel=0.5,
            sigma_distance = 0.11,
            histogram_bins=6,
            number_of_particles=150,
            q_model_noise = 1,
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
        self.search_window_position = None
        self.search_window_size = None
        self.original_size = None

        #kernel
        self.kernel = None
        self.kernel_size = None

    def extract_norm_histogram(self, patch):
        
        histogram = extract_histogram(patch, self.histogram_bins, self.kernel)

        #normalize histogram
        histogram = histogram / np.sum(histogram)

        return histogram
    #function for istogram extraction
    def extract_histogram_from_image(self, image):
        patch, _ = get_patch(image, self.search_window_position, self.search_window_size)

        histogram = extract_histogram(patch, self.histogram_bins, self.kernel)

        #normalize histogram
        histogram = histogram / np.sum(histogram)

        return histogram

    def initialize(self, image, region):

        region = np.array(region).astype(np.int)

        #make sure the region is odd sized 
        if region[2] % 2 == 0:
            region[2] += 1
        if region[3] % 2 == 0:
            region[3] += 1
        
        #get position and size of the search window
        self.search_window_position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        #set original size of the search window
        self.original_size = (region[2], region[3])

        #get the size of the search window with the enlargement factor
        x_pos = int(region[2] * self.enlargement_factor)
        y_pos = int(region[3] * self.enlargement_factor)
        # make sure the search window is odd sized
        if x_pos % 2 == 0:
            x_pos += 1
        if y_pos % 2 == 0:
            y_pos += 1
      
        self.search_window_size = (x_pos, y_pos)
        



        #define the matrix for the motion model
        self.Fi_matrika, _, self.Q_covariance, _ = get_matrices(self.model, self.q_model_noise, 1)


        #create the kernel for visual model
        self.kernel = create_epanechnik_kernel(self.search_window_size[0], self.search_window_size[1], self.sigma_kernel)
        self.kernel_size = self.kernel.shape
        
        #get histogram of the image in the search window
        self.template, _ = get_patch(image, self.search_window_position, self.search_window_size)
        
        self.template_histogram = self.extract_norm_histogram(self.template)

        #depending on the model, initialize particles state
        self.particles_state = np.array([self.search_window_position[0], self.search_window_position[1]])
        if self.model == 'NCV':
            self.particles_state = np.append(self.particles_state, [0, 0])
        if self.model == 'NCA':
            self.particles_state = np.append(self.particles_state, [0, 0, 0, 0])

        #initialize particles using gausian distribution
        self.particles = sample_gauss(self.particles_state, self.Q_covariance, self.number_of_particles) # majbe tuki kej zaokroziš??
        self.particles[:, 0:2] = np.around(self.particles[:, 0:2])
        self.weights = np.ones(self.number_of_particles) # weights are initialized to 1
    
    #code from instructions 
    def resample_particles(self):
        weights_normalized = self.weights / np.sum(self.weights)
        weights_cumsum = np.cumsum(weights_normalized)
        random_samples = np.random.rand(self.number_of_particles)
        sampled_indexes = np.digitize(random_samples, weights_cumsum)
        self.particles[sampled_indexes.flatten(), :]


    def hellinger_distance(self, p, q):
        return 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))
    
    def calculate_weights(self, image):
        #print("calculate weights")
        for i in range(self.number_of_particles):
            try: 
                #get the patch from the image
                patch, _ = get_patch(image, (self.particles[i, 0], self.particles[i, 1]), self.kernel_size) #flag
                
                # print(f"patch: {patch.shape}")
                # print(f"kernel: {self.kernel_size}")
                # print(f"search_window: {self.search_window_size}")

                # #get the histogram of the patch
                patch_histogram = self.extract_norm_histogram(patch)

                #calculate the similarity between the patch and the template using hellinger distance
                hellinger_distance = self.hellinger_distance(patch_histogram, self.template_histogram)
                probabilty = np.exp(-0.5 * hellinger_distance ** 2 / self.sigma_distance ** 2)
            
            except Exception:
                probabilty = 0
                #sometimes the patch is of by 1 in one dimension and I am not sure why that happens
                #I think it is because of the way I calculate the patch position
                #I tried to fix it but I could not figure it out
                #I am not sure if this is the best way to fix it but it works for now 
            
            #apply the new weight of the particle
            self.weights[i] = probabilty

    def track(self, image):
            
        #replace the particles by sampling new particles absed on the weight distribution
        self.resample_particles()

        #predict the particles using the motion model
        self.particles = np.matmul(self.Fi_matrika, self.particles.T).T

        #dont forget to add noise to the particles
        self.particles = self.particles + sample_gauss(np.zeros(self.Fi_matrika.shape[0]), self.Q_covariance, self.number_of_particles)
        self.particles[:, 0:2] = np.around(self.particles[:, 0:2])

        # recalculating the weights based on visiual model similarity
        self.calculate_weights(image)
        
        # Compute the new state estimate
        weights_normalized = self.weights / np.sum(self.weights)
        self.particles_state = np.matmul( np.transpose(self.particles), weights_normalized)

        #update the template
        self.template, _ = get_patch(image, self.particles_state[0:2], self.search_window_size) #flag 
        hist = self.extract_norm_histogram(self.template)
        self.template_histogram = self.alpha_update * hist + (1 - self.alpha_update) * self.template_histogram

        # return the new state estimate
        return [int(self.particles_state[0] - self.particles_state[0]/2), int(self.particles_state[1]- self.particles_state[1]/2) , self.search_window_size[0], self.search_window_size[1]]






