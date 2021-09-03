#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from vision_msgs.msg import Detection2DArray
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

        self.robot_name = self._get_robot_name()
        self.target_frame = 'utm'
        self.groundtruth_frame = 'gt/{}/base_link'.format(self.robot_name)
        self.boundary = boundary
        self.map_objects = self._init_map_objects()
        self.env_lims = self._set_envlims(ax_lims)

        self.fig, self.ax = plt.subplots()
        self.environment_handles = self.init_plot()

        self.particles = None
        self.groundtruth_pose = None
        self.measurements = None
        self.groundtruth_line = self.init_empty_line_plot(
            label='groundtruth_pose', color='g', markersize=3)
        self.particles_line = self.init_empty_line_plot(label='particles',
                                                        color='b',
                                                        markersize=1)
        self.measurement_line = self.init_empty_line_plot(label='measurements',
                                                          color='k',
                                                          markersize=3)

    def _get_robot_name(self):
        """Get robot name if exist, else use default = sam"""
        if rospy.has_param('~robot_name'):
            return rospy.get_param('~robot_name')
        return 'sam'

    def _wait_for_transform(self, from_frame):
        """Wait for transform from from_frame to self.target_frame"""
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.target_frame, from_frame, rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))
        return trans

    def update_measurements(self, msg):
        pose = msg.detections[0].results[0].pose
        trans = self._wait_for_transform(from_frame=msg.header.frame_id)
        self.measurements = tf2_geometry_msgs.do_transform_pose(pose, trans)

    def _tf_to_pose(self, trans_stamped):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time()
        pose_stamped.header.frame_id = self.target_frame

        trans = trans_stamped.transform
        pose_stamped.pose.position = trans.translation
        pose_stamped.pose.orientation = trans.rotation
        return pose_stamped

    def update_groundtruth_pose(self, msg):
        trans = self._wait_for_transform(from_frame=self.groundtruth_frame)
        self.groundtruth_pose = self._tf_to_pose(trans)

    def init_empty_line_plot(self, label, color, markersize):
        line, = self.ax.plot([], [],
                             label=label,
                             c=color,
                             marker='o',
                             ls='',
                             markersize=markersize)
        return line

    def animate_groundtruth_pose(self, i):
        """
        This method is periodically called by FuncAnimation to update the
        environment plot with the new groundtruth pose.

        Args:
            - i: frame id
            - line: the line object
        """
        x, y = [], []
        if self.groundtruth_pose is not None:
            x = self.groundtruth_pose.pose.position.x
            y = self.groundtruth_pose.pose.position.y
        self.groundtruth_line.set_xdata(x)
        self.groundtruth_line.set_ydata(y)

    def animate_measurement_pose(self, i):
        """
        This method is periodically called by FuncAnimation to update the
        environment plot with the new measurement pose.

        Args:
            - i: frame id
            - line: the line object
        """
        x, y = [], []
        if self.measurements is not None:
            x = self.measurements.pose.position.x
            y = self.measurements.pose.position.y
        self.measurement_line.set_xdata(x)
        self.measurement_line.set_ydata(y)

    def update_particles(self, msg):
        num_particles = msg.layout.dim[0].size
        num_states = msg.layout.dim[1].size
        offset = msg.layout.data_offset

        self.particles = np.array(msg.data[offset:]).reshape(
            num_particles, num_states)

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
    groundtruth_pose_sub = rospy.Subscriber('/tf', TFMessage,
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

    measurement_topic = '/{}/sim/sidescan/detection_hypothesis'.format(
        viz.robot_name)
    measurement_sub = rospy.Subscriber(measurement_topic, Detection2DArray,
                                       viz.update_measurements)
    animate_measurements = animation.FuncAnimation(
        viz.fig, viz.animate_measurement_pose, interval=50)

    legend_handles = viz.environment_handles
    legend_handles.extend(
        [viz.groundtruth_line, viz.particles_line, viz.measurement_line])
    plt.legend(handles=legend_handles,
               bbox_to_anchor=(1.01, 1),
               loc='upper left',
               borderaxespad=0.)
    plt.show(block=True)


if __name__ == '__main__':
    main()
