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
        self.environment_handles = self.init_plot()

        self.particles = None
        self.groundtruth_pose = None
        self.groundtruth_line = self.init_groundtruth_pose_plot()
        self.particles_line = self.init_particles_plot()

    def _get_robot_name(self):
        """Get robot name if exist, else use default = sam"""
        if rospy.has_param('~robot_name'):
            return rospy.get_param('~robot_name')
        return 'sam'

    def update_groundtruth_pose(self, msg):
        pose = msg.pose
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
        """
        self.groundtruth_line.set_xdata(self.groundtruth_pose.pose.position.x)
        self.groundtruth_line.set_ydata(self.groundtruth_pose.pose.position.y)
        return self.groundtruth_line

    def update_particles(self, msg):
        num_particles = msg.layout.dim[0].size
        num_states = msg.layout.dim[1].size
        offset = msg.layout.data_offset

        self.particles = np.array(msg.data[offset:]).reshape(
            num_particles, num_states)

    def init_particles_plot(self):
        self.particles_line, = self.ax.plot([], [],
                                            label='particles',
                                            c='b',
                                            marker='o',
                                            ls='',
                                            markersize=1)
        return self.particles_line

    def animate_particles(self, i):
        """
        This method is periodically called by FuncAnimation to update the
        environment plot with the new particle states.

        Args:
            - i: frame id
        """
        if self.particles is not None:
            self.particles_line.set_xdata(self.particles[:, 0])
            self.particles_line.set_ydata(self.particles[:, 1])
            print('update particle line')
            print(self.particles)
        return self.particles_line

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

        environment_handles = self._static_env_plot()

        plt.title('Environment')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        return environment_handles

    def _static_env_plot(self):
        """
        Plot the map consisting of self.map_objects in 2D plot onto the
        Visualizer's Axes object.
        """
        environment_handles = []

        for obj_id, obj_val in self.map_objects.items():
            # Plot as continuous line or individual point depending on the info given
            if isinstance(obj_val['x'], float):
                handle, = self.ax.plot(obj_val['x'],
                                       obj_val['y'],
                                       c=obj_val['c'],
                                       marker='o',
                                       ls='',
                                       markersize=obj_val['s'] * 10,
                                       label=obj_id)
            elif isinstance(obj_val['x'], tuple):
                x = np.linspace(max(obj_val['x'][0], self.env_lims['x'][0]),
                                min(obj_val['x'][1], self.env_lims['x'][1]),
                                num=100)
                y = np.linspace(max(obj_val['y'][0], self.env_lims['y'][0]),
                                min(obj_val['y'][1], self.env_lims['y'][1]),
                                num=100)
                handle, = self.ax.plot(x,
                                       y,
                                       c=obj_val['c'],
                                       linewidth=obj_val['s'],
                                       label=obj_id)
            environment_handles.append(handle)

        return environment_handles


def main():
    rospy.init_node('particle_filter_localization_visualizer', anonymous=True)
    rospy.Rate(5)

    viz = Vizualizer()

    print(viz.env_lims)

    #TODO: refactor - merge init and animate functions for both groundtruth and particles
    groundtruth_topic = '/{}/sim/odom'.format(viz.robot_name)
    groundtruth_pose_sub = rospy.Subscriber(groundtruth_topic, Odometry,
                                            viz.update_groundtruth_pose)
    # It's important to store the FuncAnimation to a variable
    # Otherwise the animation object will be collected by the garbage collector
    animate_groundtruth = animation.FuncAnimation(viz.fig,
                                                  viz.animate_groundtruth_pose,
                                                  interval=50)

    particles_topic = '{}/localization/particles'.format(viz.robot_name)
    particles_sub = rospy.Subscriber(particles_topic, Float64MultiArray,
                                     viz.update_particles)
    animate_particles = animation.FuncAnimation(viz.fig,
                                                viz.animate_particles,
                                                interval=50)

    legend_handles = viz.environment_handles
    legend_handles.extend([viz.groundtruth_line, viz.particles_line])
    plt.legend(handles=legend_handles,
               bbox_to_anchor=(1.01, 1),
               loc='upper left',
               borderaxespad=0.)
    plt.show(block=True)


if __name__ == '__main__':
    main()
