#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from visualization_msgs.msg import MarkerArray
from smarc_msgs.msg import ThrusterFeedback
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D
from sensor_msgs.msg import Imu
import numpy as np
from scipy.stats import norm


class ParticleFilter:
    def __init__(self, num_particles, num_states, process_noise,
                 measurement_noise):
        self.robot_name = self._get_robot_name()
        self.num_particles = num_particles
        self.num_states = num_states
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.target_frame = 'utm'
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.particles = self._init_particles_for_tracking()

        self.particles_msg = self._init_particles_msg()
        self.particles_topic = '/{}/localization/particles'.format(
            self.robot_name)
        self.particles_pub = rospy.Publisher(self.particles_topic,
                                             Float64MultiArray,
                                             queue_size=5)

        # Motion model related
        self.thrusts = {1: 0, 2: 0}
        self.thruster1_sub = self._setup_thruster_sub(1)
        self.thruster2_sub = self._setup_thruster_sub(2)
        self.coeff = .0005
        self.imu = Imu()
        self.imu_sub = self._setup_imu_sub()

        # Measurement model related
        self.weights = self._init_weights()
        self.landmarks = self._read_landmarks()
        self.num_landmarks = self.landmarks.shape[0]
        self.measurements = np.array([[]])
        self.measurement_sub = self._setup_measurement_sub()
        self.has_new_measurements = False

        # Update
        self.dt = .1
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.run)

    def _setup_measurement_sub(self):
        obs_topic = '/{}/sim/sidescan/detection_hypothesis'.format(
            self.robot_name)
        obs_sub = rospy.Subscriber(obs_topic, Detection2DArray,
                                   self._update_measurement)
        return obs_sub

    def _update_measurement(self, msg):
        self.has_new_measurements = True

        self.measurements = np.array([[d.results[i].pose.pose.position.y]
                                      for d in msg.detections
                                      for i in range(len(d.results))])

    def _read_landmarks(self):
        """Wait for /{robot_name}/sim/marked_positions to publish its first
        message, use it to initialize landmarks array."""

        marked_pos_topic = '/{}/sim/marked_positions'.format(self.robot_name)
        msg = rospy.wait_for_message(marked_pos_topic, MarkerArray)

        landmarks = []
        for marker in msg.markers:
            landmarks.append([
                marker.pose.position.x, marker.pose.position.y,
                marker.pose.position.z
            ])
        return np.array(landmarks)

    def _setup_imu_sub(self):
        imu_topic = '/{}/core/sbg_imu'.format(self.robot_name)
        imu_sub = rospy.Subscriber(imu_topic, Imu, self._update_imu)
        return imu_sub

    def _update_imu(self, msg):
        self.imu = msg

    def _setup_thruster_sub(self, i):
        topic = '/{}/core/thruster{}_fb'.format(self.robot_name, i)
        thruster_sub = rospy.Subscriber(topic, ThrusterFeedback,
                                        self._update_thrust, i)
        return thruster_sub

    def _update_thrust(self, msg, i):
        self.thrusts[i] = msg.rpm.rpm

    def _get_robot_name(self):
        """Get robot name if exist, else use default = sam"""
        if rospy.has_param('~robot_name'):
            return rospy.get_param('~robot_name')
        return 'sam'

    def _init_particles_msg(self):
        dim0 = MultiArrayDimension(label='particle_index',
                                   size=self.num_particles,
                                   stride=self.num_particles * self.num_states)
        dim1 = MultiArrayDimension(label='particle_state',
                                   size=self.num_states,
                                   stride=self.num_states)
        layout = MultiArrayLayout(data_offset=0, dim=[dim0, dim1])
        particle_msg = Float64MultiArray(layout=layout,
                                         data=self.particles.flatten())
        return particle_msg

    def _init_particles_for_tracking(self):

        particles = np.zeros((self.num_particles, self.num_states))

        # Get init pose in utm
        odom_topic = '/{}/sim/odom'.format(self.robot_name)
        msg = rospy.wait_for_message(odom_topic, Odometry)
        print('odom msg: {}'.format(msg))
        pose = msg.pose

        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.target_frame, msg.header.frame_id, rospy.Time(),
                    rospy.Duration(1.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))

        init_pose = tf2_geometry_msgs.do_transform_pose(pose, trans).pose
        (init_roll, init_pitch, init_yaw) = euler_from_quaternion([
            init_pose.orientation.x, init_pose.orientation.y,
            init_pose.orientation.z, init_pose.orientation.w
        ])
        mean_state = [
            init_pose.position.x, init_pose.position.y, init_pose.position.z,
            init_roll, init_pitch, init_yaw
        ]
        particles = np.array(mean_state * self.num_particles).reshape(
            self.num_particles, self.num_states)

        #TODO: set spread at the proper place and to the proper value
        particles += np.random.uniform(low=-.1, high=.1, size=particles.shape)
        # Angles should be between (-pi, pi)
        particles = self._normalize_angles(particles)
        return particles

    def _normalize_angles(self, particles):
        particles[:, -3:] = (particles[:, -3:] + np.pi) % (2 * np.pi) - np.pi
        return particles

    def _init_weights(self):
        return np.array([1.0 / self.num_states] * self.num_states)

    def run(self, timer):
        self.motion_model()
        if self.has_new_measurements:
            print('Resampling!')
            self.weights = self.measurement_model()
            print(self.weights)
            self.particles = self.systematic_resampling()
        self.particles_msg.data = self.particles.flatten()
        self.particles_pub.publish(self.particles_msg)

    def motion_model(self):
        thrust = (self.thrusts[1] + self.thrusts[2]) * self.coeff
        linear_velocity = np.array([thrust, 0, 0]).reshape(3, 1)
        (roll, pitch, yaw) = euler_from_quaternion([
            self.imu.orientation.x, self.imu.orientation.y,
            self.imu.orientation.z, self.imu.orientation.w
        ])
        rotation = self._compute_rotation(roll, pitch, yaw)
        velocity = np.matmul(rotation, linear_velocity)
        print('thrust: {}, velocity: {}'.format(thrust, velocity))

        # TODO: set proper process noise for displacement and orientation
        # separately?
        self.particles[:, :3] += (velocity * self.dt).reshape(1, 3)
        self.particles[:, -3:] = [roll, pitch, yaw]
        # Diffusion
        self.particles += np.random.randn(
            *self.particles.shape) * self.process_noise
        self.particles = self._normalize_angles(self.particles)

    def _compute_rotation(self, roll, pitch, yaw):
        rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotation = np.matmul(rz, np.matmul(ry, rx))
        return rotation

    def measurement_model(self):
        self.has_new_measurements = False
        # prediction: (num_landmarks, num_particles)
        prediction = self.predicted_measurement()

        # likelihood: (num_measurements, num_landmarks, num_particles)
        likelihood = np.stack([
            norm(prediction,
                 self.measurement_noise).pdf(self.measurements[i, :])
            for i in range(len(self.measurements))
        ])

        # association: (num_measurements, num_particles)
        # association = np.argmax(likelihood, axis=1)
        ml_likelihood = np.max(likelihood, axis=1)

        # assumes i.i.d. measurements
        weights = np.sum(ml_likelihood, axis=0)
        weights += np.finfo(float).eps
        weights /= np.sum(weights)
        return weights

    def predicted_measurement(self):
        """Compute predicted sidescan measurement for all landmarks from all
        particles. Note that the sidescan measurement only consists of a range
        measure in meters. Only landmarks within a certain threshold of
        angles are deemed observable, the others have predicted measurement set
        to inf.
        Returns:
            - dist: (num_landmarks, num_particles)
        """
        #TODO: correct predicted measurement
        # vectors pointing from particles to landmarks: (num_landmarks, num_particles, 3)
        particle_to_landmark_vec = np.stack([
            self.landmarks[i, :] - self.particles[:, :3]
            for i in range(self.num_landmarks)
        ])
        # distance between particles and landmarks: (num_landmarks, num_particles)
        dist = np.linalg.norm(particle_to_landmark_vec, axis=2)

        # convert particles to landmark vector into unit vectors
        particle_to_landmark_vec_normalized = particle_to_landmark_vec / dist.reshape(
            self.num_landmarks, self.num_particles, 1)

        # compute heading
        heading = np.stack([
            np.cos(self.particles[:, -1]) * np.cos(self.particles[:, -2]),
            np.sin(self.particles[:, -1]) * np.cos(self.particles[:, -2]),
            np.sin(self.particles[:, -2])
        ])
        # normalize heading
        heading_normalized = heading / np.linalg.norm(heading, axis=0)

        # compute cos angle between heading and particle_to_landmark_vec
        dot_prod = np.dot(particle_to_landmark_vec_normalized,
                          heading_normalized)
        # cos: (num_landmarks, num_particles)
        cos = np.stack(
            [dot_prod[i, :].diagonal() for i in range(self.num_landmarks)])

        #TODO: set threshold at the proper place to the proper value
        thresh = .05
        observable = np.abs(cos) <= thresh
        dist[~observable] = np.inf
        return dist

    def systematic_resampling(self):
        cumsum = np.cumsum(self.weights)
        random_nr = np.random.rand() / self.num_particles
        thresh = np.arange(start=random_nr,
                           stop=1.,
                           step=1. / self.num_particles)

        indices = []
        i, j = 0, 0
        while i < len(cumsum) and j < len(thresh):
            if cumsum[i] >= thresh[j]:
                indices.append(i)
                j += 1
            else:
                i += 1
        return self.particles[indices, :]


def main():
    rospy.init_node('particle_filter', anonymous=True)
    rospy.Rate(5)

    num_particles = 100
    num_states = 6
    process_noise = .1
    measurement_noise = .1

    particle_filter = ParticleFilter(num_particles, num_states, process_noise,
                                     measurement_noise)

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
