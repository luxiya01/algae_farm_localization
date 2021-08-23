#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
import numpy as np


class ParticleFilter:
    def __init__(self, num_particles, num_states):
        self.robot_name = self._get_robot_name()
        self.num_particles = num_particles
        self.num_states = num_states
        self.particles = np.zeros((self.num_particles, self.num_states))

        self.particles_msg = self._init_particle_msg()
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

    def _init_particle_msg(self):
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

    def run(self):
        random_xs = np.random.randint(643785, 643815, size=self.num_particles)
        random_ys = np.random.randint(6459246,
                                      6459276,
                                      size=self.num_particles)
        self.particles[:, 0] = random_xs
        self.particles[:, 1] = random_ys

        self.particles_msg.data = self.particles.flatten()
        self.particles_pub.publish(self.particles_msg)


def main():
    rospy.init_node('particle_filter', anonymous=True)
    rospy.Rate(5)

    num_particles = 100
    num_states = 6

    particle_filter = ParticleFilter(num_particles, num_states)

    while not rospy.is_shutdown():
        particle_filter.run()


if __name__ == '__main__':
    main()
