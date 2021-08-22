#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
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
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.target_frame = 'utm'
        self.robot_name = self._get_robot_name()
        self.boundary = boundary
        self.map_objects = self._init_map_objects()
        self.env_lims = self._set_envlims(ax_lims)

        self.fig, self.ax = plt.subplots()
        self.init_plot()

        #self.particles_topic = '/{}/localization/particles'.format(
        #    self.robot_name)
        #self.particles = None
        #self.particle_color = 'b'
        #self._setup_particle_subscriber()

        self.groundtruth_pose = None

    def _get_robot_name(self):
        """Get robot name if exist, else use default = sam"""
        if rospy.has_param('~robot_name'):
            return rospy.get_param('~robot_name')
        return 'sam'

    def update_groundtruth_pose(self, msg):
        pose = msg.pose
        print('pose updated')
        try:
            trans = self.tf_buffer.lookup_transform(self.target_frame,
                                                    msg.header.frame_id,
                                                    rospy.Time())
            self.groundtruth_pose = tf2_geometry_msgs.do_transform_pose(
                pose, trans)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            print('Failed to transform. Error: {}'.format(error))

    def init_groundtruth_pose_plot(self):
        print('I initialized the groundtruth plot')
        self.groundtruth_line, = self.ax.plot([], [],
                                              label='groundtruth_pose',
                                              c='g',
                                              marker='o',
                                              ls='')
        return self.groundtruth_line

    def animate_groundtruth_pose(self, i):
        """
        This method is periodically called by FuncAnimation to update the
        environment plot with the new groundtruth pose.

        Args:
            - i: frame id
            - line: line plot object to be updated
        """
        print('Im in animate_groundtruth_pose!')
        self.groundtruth_line.set_xdata(self.groundtruth_pose.pose.position.x)
        self.groundtruth_line.set_ydata(self.groundtruth_pose.pose.position.y)
        return self.groundtruth_line

    def _setup_particle_subscriber(self):
        self.particle_sub = rospy.Subscriber(self.particles_topic,
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


#       animation.FuncAnimation(self.fig,
#                               self._animate_particles,
#                               fargs=(line, ),
#                               interval=50,
#                               blit=True)

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
        """Initialize a 2D plot containing all the static map_objects."""
        self.ax.set_xlim(self.env_lims['x'])
        self.ax.set_ylim(self.env_lims['y'])

        self._static_env_plot()

        plt.legend(bbox_to_anchor=(1.01, 1),
                   loc='upper left',
                   borderaxespad=0.)
        plt.title('Environment')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

    def _static_env_plot(self):
        """
        Plot the map consisting of self.map_objects in 2D plot onto the
        Visualizer's Axes object.
        """

        for obj_id, obj_val in self.map_objects.items():
            # Plot as continuous line or individual point depending on the info given
            if isinstance(obj_val['x'], float):
                self.ax.scatter(obj_val['x'],
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
                self.ax.plot(x,
                             y,
                             c=obj_val['c'],
                             linewidth=obj_val['s'],
                             label=obj_id)


def main():
    rospy.init_node('particle_filter_localization_visualizer', anonymous=True)
    rospy.Rate(5)

    viz = Vizualizer()

    groundtruth_topic = '/{}/sim/odom'.format(viz.robot_name)
    groundtruth_pose_sub = rospy.Subscriber(groundtruth_topic, Odometry,
                                            viz.update_groundtruth_pose)
    # It's important to assign this the FuncAnimation to a variable
    # Otherwise it will be ignored and not run
    ani = animation.FuncAnimation(viz.fig,
                                  viz.animate_groundtruth_pose,
                                  interval=50,
                                  init_func=viz.init_groundtruth_pose_plot)
    plt.show(block=True)


if __name__ == '__main__':
    main()
