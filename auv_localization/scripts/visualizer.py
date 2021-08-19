#!/usr/bin/env python

import rospy
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np


class Vizualizer:
    """Class for visualization of the environment, ground truth robot position
    and particle filter state estimates."""
    def __init__(self,
                 ax_lims={
                     'x': None,
                     'y': None
                 },
                 boundary={
                     'x': 10,
                     'y': 10
                 }):
        self.robot_name = self._get_robot_name()
        self.boundary = boundary
        self.map_objects = self._init_map_objects()
        self.env_lims = self._set_envlims(ax_lims)
        self.fig, self.ax = self.init_plot()
        plt.show()

        self.particles = None
        self.particle_color = 'g'
        self._setup_particle_subscriber()

    def _get_robot_name(self):
        """Get robot name if exist, else use default = sam"""
        if rospy.has_param('~robot_name'):
            return rospy.get_param('~robot_name')
        return 'sam'

    def _setup_particle_subscriber(self):
        particles_topic = '/{}/localization/particles'.format(self.robot_name)
        self.particle_sub = rospy.Subscriber(particles_topic,
                                             Float64MultiArray,
                                             self._update_particles)

    def _update_particles(self, msg):
        # Initialize particles for the first time
        if self.particles is None:
            self._init_particles(msg)

        offset = msg.layout.data_offset
        self.particles = np.array(msg.data).reshape(self.particles.shape)

    def _init_particles(self, msg):
        particles_dim = [x.size for x in msg.layout.dim]
        self.particles = np.zeros((particles_dim))
        line, = self.ax.plot(self.particles[0, :],
                             self.particles[1, :],
                             label='particles',
                             c=self.particle_color,
                             marker='o',
                             ls='')
        animation.FuncAnimation(self.fig,
                                self._animate_particles,
                                fargs=(line, ),
                                interval=50,
                                blit=True)

    def _animate_particles(self, i, line):
        """
        This method is periodically called by FuncAnimation to update the
        environment plot with the new particle states.

        Args:
            - i: frame id
            - line: line plot object to be updated
        """
        line.set_xdata(self.particles[0, :])
        line.set_ydata(self.particles[1, :])

        return line,

    def _init_map_objects(self):
        """Wait for /{robot_name}/sim/marked_positions to publish its first
        message, use it to initialize the map with object positions."""

        marked_pos_topic = '/{}/sim/marked_positions'.format(self.robot_name)
        msg = rospy.wait_for_message(marked_pos_topic, MarkerArray)

        map_objects = {}
        for marker in msg.markers:
            map_objects[marker.ns] = {
                'x':
                marker.pose.position.x,
                'y':
                marker.pose.position.y,
                'z':
                marker.pose.position.z,
                'c': (marker.color.r, marker.color.g, marker.color.b,
                      marker.color.a),
                's':
                marker.scale.x
            }
        print(map_objects)
        return map_objects

    def _set_envlims(self, ax_lims):
        """Set limits for all map axes. Set axis limit to that in ax_lims if
        provided, else bound the axis so that all objects in the map can be
        seen."""
        env_lims = {}

        for axis, lim in ax_lims.items():
            if lim is not None:
                env_lims[axis] = lim
            else:
                obj_values = [obj[axis] for obj in self.map_objects.values()]
                lim = (min(obj_values) - self.boundary[axis],
                       max(obj_values) + self.boundary[axis])
                env_lims[axis] = lim

        return env_lims

    def init_plot(self):
        """Initialize a 2D plot containing all the static map_objects.

        Args:
            - num_particles: number of particles in the particle filter to be plotted

        Return: Figure handler and the line object to be updated
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(self.env_lims['x'])
        ax.set_ylim(self.env_lims['y'])

        self._static_env_plot(ax)

        plt.legend(bbox_to_anchor=(1.01, 1),
                   loc='upper left',
                   borderaxespad=0.)
        plt.title('Environment')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        return fig, ax

    def _static_env_plot(self, ax):
        """
        Plot the map consisting of self.map_objects in 2D plot onto the given
        Axes object.
        """

        for obj_id, obj_val in self.map_objects.items():
            # Plot as continuous line or individual point depending on the info given
            if isinstance(obj_val['x'], float):
                ax.scatter(obj_val['x'],
                           obj_val['y'],
                           c=obj_val['c'],
                           s=obj_val['s'] * 10,
                           label=obj_id)
            elif isinstance(obj_val['x'], tuple):
                x = np.linspace(max(obj_val['x'][0], self.env_lims['x'][0]),
                                min(obj_val['x'][1], self.env_lims['x'][1]),
                                num=100)
                y = np.linspace(max(obj_val['y'][0], self.env_lims['y'][0]),
                                min(obj_val['y'][1], self.env_lims['y'][1]),
                                num=100)
                ax.plot(x,
                        y,
                        c=obj_val['c'],
                        linewidth=obj_val['s'],
                        label=obj_id)


def main():
    rospy.init_node('particle_filter_localization_visualizer', anonymous=True)
    rospy.Rate(5)

    viz = Vizualizer()

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
