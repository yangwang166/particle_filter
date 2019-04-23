#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "particle_filter.h"
#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using namespace std;


void ParticleFilter::init(double x, 
                          double y, 
                          double theta, 
                          double std[]) {

  // start

  num_particles = 50;

  // Create Gaussian Distribution
  normal_distribution<double> t_g_dist_x(x, std[0]);
  normal_distribution<double> t_g_dist_y(y, std[1]);
  normal_distribution<double> t_g_dist_theta(theta, std[2]);

  // Init Particles
  for (int i = 0; i < num_particles; i++) {
    Particle t_p;
    t_p.id = i;
    t_p.weight = 1;
    t_p.x = t_g_dist_x(gen);
    t_p.y = t_g_dist_y(gen);
    t_p.theta = t_g_dist_theta(gen);
    particles.push_back(t_p);
  }

  is_initialized = true;
  
  // end

}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  // start

  unsigned int t_num_predictions = predicted.size();
  unsigned int t_num_observations = observations.size();

  for (int i = 0; i < t_num_observations; i++) { 

    // Init  
    double t_min_distance = numeric_limits<double>::max();
    int t_mapId = -1;
    for (int j = 0; j < t_num_predictions; j++ ) { 
      double t_x_distance = observations[i].x - predicted[j].x;
      double t_y_distance = observations[i].y - predicted[j].y;
      double t_distance = t_x_distance * t_x_distance + t_y_distance * t_y_distance;

      if (t_distance < t_min_distance) {
        t_min_distance = t_distance;
        t_mapId = predicted[j].id;
      }
    }

    // Update the observations
    observations[i].id = t_mapId;
  }
  
  // end

}


void ParticleFilter::prediction(double delta_t, 
                                double std_pos[], 
                                double velocity, 
                                double yaw_rate) {

  // start

  // Create Gaussian Distribution
  normal_distribution<double> t_g_delta_x(0, std_pos[0]);
  normal_distribution<double> t_g_delta_y(0, std_pos[1]);
  normal_distribution<double> t_g_delta_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    // Update new state for particles
    if (fabs(yaw_rate) > 0.00001) {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }

    // Add noise to particles
    particles[i].x += t_g_delta_x(gen);
    particles[i].y += t_g_delta_y(gen);
    particles[i].theta += t_g_delta_theta(gen);
  }
  
  // end
}



void ParticleFilter::resample() {

  // start

  // Init
  vector<double> t_weights;
  double t_max_weight = numeric_limits<double>::min();

  for(int i = 0; i < num_particles; i++) {
    t_weights.push_back(particles[i].weight);

    if(particles[i].weight > t_max_weight) {
      t_max_weight = particles[i].weight;
    }
  }

  uniform_real_distribution<double> t_u_dist_double(0.0, t_max_weight);
  uniform_int_distribution<int> t_u_dist_int(0, num_particles - 1);

  int t_idx = t_u_dist_int(gen);
  double beta = 0.0;

  vector<Particle> t_resampled_particles;

  for(int i = 0; i < num_particles; i++) {
    beta += t_u_dist_double(gen) * 2.0;

    while(beta > t_weights[t_idx]) {
      t_idx = (t_idx + 1) % num_particles;
      beta -= t_weights[t_idx];
    }

    t_resampled_particles.push_back(particles[t_idx]);
  }
  particles = t_resampled_particles;
  
  // end
  
}


void ParticleFilter::updateWeights(double sensor_range, 
                                   double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  // start

  for (int i = 0; i < num_particles ; i++) {

    double t_p_theta = particles[i].theta;
    double t_p_x = particles[i].x;
    double t_p_y = particles[i].y;
    
    std::vector<LandmarkObs> t_valid_landmarks;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double t_x_f = map_landmarks.landmark_list[j].x_f;
      double t_y_f = map_landmarks.landmark_list[j].y_f;
      int t_id = map_landmarks.landmark_list[j].id_i;
      
      bool t_use_distance = false;

      if (t_use_distance) {

        double distance = dist(t_p_x, t_p_y, t_x_f , t_y_f);

        if (distance <= sensor_range) {
          t_valid_landmarks.push_back(LandmarkObs{ t_id, t_x_f, t_y_f });
        }

      } else {

        if (fabs(t_x_f - t_p_x) <= sensor_range && fabs(t_y_f - t_p_y) <= sensor_range) {
          t_valid_landmarks.push_back(LandmarkObs{ t_id, t_x_f, t_y_f });
        }

      }
    }
    
    std::vector<LandmarkObs> t_transformed_obs;

    for (int j = 0; j < observations.size(); j++) {
      double t_x = cos(t_p_theta) * observations[j].x - sin(t_p_theta) * observations[j].y + t_p_x;
      double t_y = sin(t_p_theta) * observations[j].x + cos(t_p_theta) * observations[j].y + t_p_y;
      t_transformed_obs.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }
    
    // Call dataAssociation
    dataAssociation(t_valid_landmarks, t_transformed_obs);
    
    particles[i].weight = 1.0;

    for (int j = 0; j < t_transformed_obs.size(); j++) {
      double t_o_x = t_transformed_obs[j].x;
      double t_o_y = t_transformed_obs[j].y;
      int t_landmark_id = t_transformed_obs[j].id;

      double weight = EPS;

      for (int k = 0; k < t_valid_landmarks.size(); k++) {

        if (t_valid_landmarks[k].id == t_landmark_id) {

          double t_dX = t_o_x - t_valid_landmarks[k].x;
          double t_dY = t_o_y - t_valid_landmarks[k].y;
          double t_s_x = std_landmark[0];
          double t_s_y = std_landmark[1];
          weight = (1 / (2 * M_PI * t_s_x * t_s_y)) * exp(-( t_dX * t_dX / (2 * t_s_x * t_s_x) + (t_dY * t_dY / (2 * t_s_y * t_s_y))));

          if (weight == 0) {
            weight = EPS;
          }
          break;
        }

      }

      particles[i].weight = particles[i].weight * weight;

    }
  }
  
  // end
}





void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}




string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}




string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
