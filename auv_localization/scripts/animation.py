#!/usr/bin/env python

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


def static_env_plot(ax, xlim, ylim, objects_dict):
    """
    Plot the static environment consisting of objects in the scene as a 2D plot.
    """

    for obj_id, obj_val in objects_dict.items():
        # Plot as continuous line or individual point depending on the info given
        if isinstance(obj_val['x'], int):
            x = obj_val['x']
            y = obj_val['y']
            ax.scatter(obj_val['x'], obj_val['y'], c=obj_val['c'],
                    s=obj_val['s'], label=obj_id)
        elif isinstance(obj_val['x'], tuple):
            x = np.linspace(max(obj_val['x'][0], xlim[0]), min(obj_val['x'][1],
                xlim[1]), num=100)
            y = np.linspace(max(obj_val['y'][0], ylim[0]), min(obj_val['y'][1],
                ylim[1]), num=100)
            ax.plot(x, y, c=obj_val['c'], linewidth=obj_val['s'], label=obj_id)


def init_particles_plot(ax, num_particles, color='g'):
    """Initialize a dummy line plot with num_particles, render like scatter plot
    by using marker 'o' and not showing the line"""
    line, = ax.plot(np.zeros(num_particles), np.zeros(num_particles),
            label='particles', c=color, marker='o', ls='')
    return line


def init_plot(xlim, ylim, objects_dict, num_particles):
    """
    Args:
        - xlim: (min_x, max_x)
        - ylim: (min_y, max_y)
        - objects_dict: {point_obj: {x: 0, y: 0, z: 0, c: 'color', s: 1},
                         line_obj: {x: (0, 0), y: (0, 0), z: (0, 0), c: 'color', s: 1}}
        - num_particles: number of particles in the particle filter to be plotted

    Return: Figure handler and the line object to be updated
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    static_env_plot(ax, xlim, ylim, objects_dict)
    line = init_particles_plot(ax, num_particles)

    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.title('Environment')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    return fig, line


def animate(i, line):
    """
    This function is periodically called by FuncAnimation to update the
    environment plot with the new particle states.

    Args:
        - i: frame id
        - line: line plot object to be updated
        - TODO: particles: (num_state, num_particles) numpy array with particle
          states. Assumes num_particles to be static.
    """
    #TODO: take in particles by reading from ROS
    particles = np.random.randint(0, 100, (6, 1000))
    line.set_xdata(particles[0, :])
    line.set_ydata(particles[1, :])

    return line,

def main(xlim, ylim, objects_dict, num_particles):
    fig, line = init_plot(xlim, ylim, objects_dict, num_particles)
    animation.FuncAnimation(fig, animate, fargs=(line, ), interval=50, blit=True)
    plt.show()

if __name__ == '__main__':
    xlim = (-1, 110)
    ylim = (-1, 110)
    objects_dict = {'p1': {'x': 0, 'y': 2, 'c': 'r', 's': 1},
                    'p2': {'x': 5, 'y': 3, 'c': 'b', 's': 3},
                    'l1': {'x': (-.5, 2), 'y': (1, 5), 'c': 'c', 's': 1},
            }
    num_particles = 1000
    main(xlim, ylim, objects_dict, num_particles)
