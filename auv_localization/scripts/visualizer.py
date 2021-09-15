#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np


class Vizualizer:
    """Class for visualization of the environment, ground truth robot position
    and particle filter state estimates."""
    def __init__(self, robot_name, boundary={'x': 10, 'y': 10}):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.robot_name = robot_name
        self.target_frame = 'utm'
        self.groundtruth_frame = 'gt/{}/base_link'.format(self.robot_name)
        self.boundary = boundary
        self.map_objects = self._init_map_objects()
        self.env_lims = self._set_envlims()

        self.fig, self.ax = plt.subplots()
        self.environment_handles = self.init_plot()

        self.particles = None
        self.groundtruth_pose = None
        self.dr_pose = None
        self.measurements = None
        self.groundtruth_color = 'g'
        self.dr_color = 'c'
        self.particle_color = 'b'
        self.measurement_color = 'k'
        self.groundtruth_line = self.init_empty_line_plot(
            label='groundtruth_pose',
            color=self.groundtruth_color,
            markersize=3)
        self.dr_line = self.init_empty_line_plot(label='dr_pose',
                                                 color=self.dr_color,
                                                 markersize=3)
        self.particles_line = self.init_empty_line_plot(
            label='particles', color=self.particle_color, markersize=1)
        self.measurement_line = self.init_empty_line_plot(
            label='measurements', color=self.measurement_color, markersize=3)

        # summary stats
        self.dr_poses = []
        self.particle_mean = []
        self.particle_variance = []
        self.gt_poses = []
        self.dt = .5
        self.timer = rospy.Timer(rospy.Duration(self.dt),
                                 self.update_summary_stats)

    def _wait_for_transform(self, from_frame):
        """Wait for transform from from_frame to self.target_frame"""
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.target_frame, from_frame, rospy.Time(0),
                    rospy.Duration(3.0))
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

    def update_dr_pose(self, msg):
        trans = self._wait_for_transform(from_frame=msg.header.frame_id)
        self.dr_pose = tf2_geometry_msgs.do_transform_pose(msg.pose, trans)

    def update_particles(self, msg):
        num_particles = msg.layout.dim[0].size
        num_states = msg.layout.dim[1].size
        offset = msg.layout.data_offset

        self.particles = np.array(msg.data[offset:]).reshape(
            num_particles, num_states)

    def update_summary_stats(self, timer):
        if not all([(self.particles is not None), self.dr_pose,
                    self.groundtruth_pose]):
            return

        mean = self.particles.mean(axis=0)
        var = self.particles.var(axis=0)

        self.particle_mean.append(mean)
        self.particle_variance.append(var)

        (dr_roll, dr_pitch, dr_yaw) = euler_from_quaternion([
            self.dr_pose.pose.orientation.x, self.dr_pose.pose.orientation.y,
            self.dr_pose.pose.orientation.z, self.dr_pose.pose.orientation.w
        ])
        self.dr_poses.append([
            self.dr_pose.pose.position.x, self.dr_pose.pose.position.y,
            self.dr_pose.pose.position.z, dr_roll, dr_pitch, dr_yaw
        ])

        (groundtruth_roll, groundtruth_pitch,
         groundtruth_yaw) = euler_from_quaternion([
             self.groundtruth_pose.pose.orientation.x,
             self.groundtruth_pose.pose.orientation.y,
             self.groundtruth_pose.pose.orientation.z,
             self.groundtruth_pose.pose.orientation.w
         ])
        self.gt_poses.append([
            self.groundtruth_pose.pose.position.x,
            self.groundtruth_pose.pose.position.y,
            self.groundtruth_pose.pose.position.z, groundtruth_roll,
            groundtruth_pitch, groundtruth_yaw
        ])

    def init_empty_line_plot(self, label, color, markersize):
        line, = self.ax.plot([], [],
                             label=label,
                             c=color,
                             marker='o',
                             ls='',
                             markersize=markersize)
        return line

    def animate_dr_pose(self, i):
        x, y = [], []
        if self.dr_pose is not None:
            x = self.dr_pose.pose.position.x
            y = self.dr_pose.pose.position.y
        self.dr_line.set_xdata(x)
        self.dr_line.set_ydata(y)

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

    def _set_envlims(self):
        """Set limits for all map axes. Bound the axis so that all objects
        in the map can be seen and use self.boundary to determine the size
        of the canvas that should extend from the edge objects."""
        env_lims = {}

        for axis, axis_boundary in self.boundary.items():
            obj_values = [obj[axis] for obj in self.map_objects.values()]
            lim = (min(obj_values) - axis_boundary,
                   max(obj_values) + axis_boundary)
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

    def plot_summary_stats(self):
        fig, axes = plt.subplots(nrows=3,
                                 ncols=2,
                                 sharex=True,
                                 figsize=(9, 12))
        x_axis = list(range(len(self.gt_poses)))
        titles = ['x values', 'y values', 'z values', 'roll', 'pitch', 'yaw']

        for i in range(3):
            for j in range(2):
                axes[i][j].set_title(titles[i + j * 3])
                axes[i][j].plot(x_axis,
                                [pose[i + j * 3] for pose in self.gt_poses],
                                color=self.groundtruth_color,
                                label='groundtruth')
                axes[i][j].plot(x_axis,
                                [pose[i + j * 3] for pose in self.dr_poses],
                                color=self.dr_color,
                                label='dr_pose')
                axes[i][j].errorbar(
                    x_axis, [pose[i + j * 3] for pose in self.particle_mean],
                    yerr=[pose[i + j * 3] for pose in self.particle_variance],
                    errorevery=(0, 10),
                    ecolor='lightgray',
                    color=self.particle_color,
                    label='particles')
        plt.legend(bbox_to_anchor=(1, .5), loc='lower left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

        for i in range(6):
            gt_dim = np.array([pose[i] for pose in self.gt_poses])
            dr_dim = np.array([pose[i] for pose in self.dr_poses])
            particle_mean_dim = np.array(
                [pose[i] for pose in self.particle_mean])

            particle_diff = np.linalg.norm(particle_mean_dim - gt_dim)
            dr_diff = np.linalg.norm(dr_dim - gt_dim)
            print('dim{} pf error mean: {}, var: {}'.format(
                i, particle_diff.mean(), particle_diff.var()))
            print('dim{} dr error mean: {}, var: {}\n'.format(
                i, dr_diff.mean(), dr_diff.var()))


def main():
    rospy.init_node('particle_filter_localization_visualizer',
                    anonymous=True,
                    disable_signals=True)
    rospy.Rate(5)

    robot_name = rospy.get_param('~robot_name')
    boundary = rospy.get_param('~boundary')
    viz = Vizualizer(robot_name, boundary)

    print(viz.env_lims)

    #TODO: refactor - merge animate functions for both groundtruth and particles
    groundtruth_pose_sub = rospy.Subscriber('/tf', TFMessage,
                                            viz.update_groundtruth_pose)
    # It's important to store the FuncAnimation to a variable
    # Otherwise the animation object will be collected by the garbage collector
    animate_groundtruth = animation.FuncAnimation(viz.fig,
                                                  viz.animate_groundtruth_pose,
                                                  interval=50)

    dr_odom_topic = '/{}/dr/odom'.format(viz.robot_name)
    dr_odom_sub = rospy.Subscriber(dr_odom_topic, Odometry, viz.update_dr_pose)
    animate_dr = animation.FuncAnimation(viz.fig,
                                         viz.animate_dr_pose,
                                         interval=50)

    particles_topic = '/{}/localization/particles'.format(viz.robot_name)
    particles_sub = rospy.Subscriber(particles_topic, Float64MultiArray,
                                     viz.update_particles)
    animate_particles = animation.FuncAnimation(viz.fig,
                                                viz.animate_particles,
                                                interval=50)

    measurement_topic = '/{}/payload/sidescan/detection_hypothesis'.format(
        viz.robot_name)
    measurement_sub = rospy.Subscriber(measurement_topic, Detection2DArray,
                                       viz.update_measurements)
    animate_measurements = animation.FuncAnimation(
        viz.fig, viz.animate_measurement_pose, interval=50)

    legend_handles = viz.environment_handles
    legend_handles.extend(
        [viz.groundtruth_line, viz.particles_line, viz.measurement_line])
    plt.legend(handles=legend_handles,
               bbox_to_anchor=(1, .5),
               loc='center left',
               borderaxespad=0.)
    plt.tight_layout()
    plt.show(block=True)

    viz.plot_summary_stats()


if __name__ == '__main__':
    main()
