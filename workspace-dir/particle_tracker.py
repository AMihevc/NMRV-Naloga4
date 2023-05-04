import numpy as np
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram
from ex4_utils import sample_gauss
from tracker import Tracker
from kalman_models import get_matrices

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
            lamda = 0.000001,
            histogram_bins=6,
            number_of_particles=150,
            q_model_noise = 1,
            r_measurement_noise = 1,
            model='NCV',
        ):
        self.enlargement_factor = enlargement_factor
        self.alpha_update = alpha_update
        self.sigma_kernel = sigma_kernel
        self.sigma_distance = sigma_distance
        self.lamda = lamda
        self.histogram_bins = histogram_bins
        self.number_of_particles = number_of_particles
        self.q_model_noise = q_model_noise
        self.r_measurement_noise = r_measurement_noise
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

    #function for istogram extraction
    def extract_histogram_from_image(self, image):

        histogram = extract_histogram(image, self.histogram_bins, self.kernel)

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
        self.search_window_size = (int(region[2] * self.enlargement_factor), int(region[3] * self.enlargement_factor))

        #define the matrix for the motion model
        self.Fi_matrika, _, self.Q, _ = get_matrices(self.model, self.q_model_noise, self.r_measurement_noise)

        #create the kernel for visual model
        self.kernel = create_epanechnik_kernel(self.search_window_size[0], self.search_window_size[1], self.sigma_kernel)
        
        #get histogram of the image in the search window
        self.template, _ = get_patch(image, self.search_window_position, self.search_window_size)
        
        self.template_histogram = self.extract_histogram_from_image(self.template)

        #depending on the model, initialize particles state
        self.particles_state = np.array([self.search_window_position[0], self.search_window_position[1]])
        if self.model == 'NCV':
            self.particles_state = np.append(self.particles_state, [0, 0])
        if self.model == 'NCA':
            self.particles_state = np.append(self.particles_state, [0, 0, 0, 0])

        #initialize particles using gausian distribution
        self.particles = sample_gauss(self.particles_state, self.Q_covariance, self.number_of_particles) # majbe tuki kej zaokroziš??
        self.weights = np.ones(self.number_of_particles) # weights are initialized to 1
    
    #code from instructions 
    def resample_particles(self):
        weights_normalized = self.weights / np.sum(self.weights)
        weights_cumsum = np.cumsum(weights_normalized)
        random_samples = np.random.rand(self.number_of_particles)
        sampled_indexes = np.digitize(random_samples, weights_cumsum)
        self.particles[sampled_indexes.flatten(), :]

    def hellinger_distance(p, q):
        return 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))
    
    def calculate_weights(self, image):
        for i in range(self.number_of_particles):
            #get the patch from the image
            patch, _ = get_patch(image, (self.particles[i, 0], self.particles[i, 1]), self.search_window_size) #flag

            #get the histogram of the patch
            patch_histogram = self.extract_histogram_from_image(patch)

            #calculate the similarity between the patch and the template using hellinger distance
            hellinger_distance = self.hellinger_distance(patch_histogram, self.template_histogram)

            #calculate the weight of the particle
            self.weights[i] = np.exp(-0.5 * hellinger_distance ** 2 / self.sigma_distance ** 2)

    def track(self, image):
            
        #replace the particles by sampling new particles absed on the weight distribution
        self.resample_particles()

        #predict the particles using the motion model
        self.particles = np.matmul(self.Fi_matrika, self.particles.T).T

        #dont forget to add noise to the particles
        self.particles = self.particles + sample_gauss(np.zeros(len(self.Fi_matrika.shape[0])), self.Q_covariance, self.number_of_particles)
        # majbe zaokrožiš?

        # recalculating the weights based on visiual model similarity
        self.calculate_weights(self, image)
        
        # Compute the new state estimate
        weights_normalized = self.weights / np.sum(self.weights)
        self.particles_state = np.matmul( np.transpose(self.particles), weights_normalized)

        #update the template
        self.template, _ = get_patch(image, self.particles_state[0:2], self.search_window_size) #flag 
        hist = self.extract_histogram_from_image(self.template)
        self.template_histogram = self.alpha_update * hist + (1 - self.alpha_update) * self.template_histogram

        # return the new state estimate
        return [int(self.particles_state[0] - self.particles_state[0]/2), int(self.particles_state[1]- self.particles_state[1]/2) , self.search_window_size[0], self.search_window_size[1]]






