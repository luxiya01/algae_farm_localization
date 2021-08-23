#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
import numpy as np


class ParticleFilter:
    def __init__(self, num_particles, num_states, process_noise):
        self.robot_name = self._get_robot_name()
        self.num_particles = num_particles
        self.num_states = num_states
        self.process_noise = process_noise

        self.target_frame = 'utm'
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.particles = self._init_particles_for_tracking()
        self.weights = self._init_weights()

        self.particles_msg = self._init_particles_msg()
        self.particles_topic = '/{}/localization/particles'.format(
            self.robot_name)
        self.particles_pub = rospy.Publisher(self.particles_topic,
                                             Float64MultiArray,
                                             queue_size=5)

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
        particles[:, -3:] = (particles[:, -3:] + np.pi) % (2 * np.pi) - np.pi
        return particles

    def _init_weights(self):
        return np.array([1.0 / self.num_states] * self.num_states)

    def run(self):
        self.particles[:, 0] += np.random.uniform(low=-.01,
                                                  high=.01,
                                                  size=self.num_particles)
        self.particles[:, 1] += np.random.uniform(low=-.01,
                                                  high=.01,
                                                  size=self.num_particles)

        self.particles_msg.data = self.particles.flatten()
        self.particles_pub.publish(self.particles_msg)


def main():
    rospy.init_node('particle_filter', anonymous=True)
    rospy.Rate(5)

    num_particles = 100
    num_states = 6
    process_noise = [.1, .1, .1, .1, .1]

    particle_filter = ParticleFilter(num_particles, num_states, process_noise)

    while not rospy.is_shutdown():
        particle_filter.run()


if __name__ == '__main__':
    main()
