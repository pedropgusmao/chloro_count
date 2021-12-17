# From https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import linalg
from random import random
from pathlib import Path
from typing import List


def cell2points(segments: List[np.ndarray], x_res, y_res, z_res):
    """Translates a list of 2D pixels into a a list of 3D points in nm.

    Args:
        segments (List[np.ndarray]): [description]
        x_res (float): x nm/pixel
        y_res (float): y nm/pixel
        z_res (float): Distance between planes in pixel.

    Returns:
        [type]: [description]
    """


class Cylinder:
    """Creates a Minimum Volume Cylider from a collection of points"""

    def _init_(self, P: np.ndarray):
        self.P = P

    def get_params(self):

        pass

    def get_volume(self):
        pass

    def plot(self):
        pass


class Ellipsoid:
    """Some stuff for playing with ellipsoids"""

    def __init__(self):
        pass

    def get_params(self, P=None, tolerance=0.01):
        """Find the minimum volume ellipsoid which holds all the points

        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!

        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z,...], <-- one point per line
             [x,y,z,...],
             [x,y,z,...]]

        Returns:
        (center, radii, rotation)

        """
        (N, dim) = np.shape(P)
        d = float(dim)

        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)])
        QT = Q.T

        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(
                np.dot(QT, np.dot(linalg.inv(V), Q))
            )  # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse
        center = np.dot(P.T, u)

        # the A matrix for the ellipse
        A = (
            linalg.inv(
                np.dot(P.T, np.dot(np.diag(u), P))
                - np.array([[a * b for b in center] for a in center])
            )
            / d
        )

        # Get the values we'd like to return
        U, s, rotation = linalg.svd(A)
        radii = 1.0 / np.sqrt(s)

        return (center, radii, rotation)

    def get_volume(self, radii):
        """Calculate the volume of the blob"""
        return 4.0 / 3.0 * np.pi * radii[0] * radii[1] * radii[2]

    def plot(
        self,
        center,
        radii,
        rotation,
        ax=None,
        plotAxes=False,
        cageColor="b",
        cageAlpha=0.2,
        P=None,
        inner_factor=None,
    ):
        """Plot an ellipsoid"""
        make_ax = ax == None
        if make_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)

        # cartesian coordinates that correspond to the spherical angles:
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # Inner Ellipsoid
        x_i = radii[0] * np.outer(np.cos(u), np.sin(v)) * inner_factor
        y_i = radii[1] * np.outer(np.sin(u), np.sin(v)) * inner_factor
        z_i = radii[2] * np.outer(np.ones_like(u), np.cos(v)) * inner_factor

        # Middle Ellipsoid
        x_m = radii[0] * np.outer(np.cos(u), np.sin(v)) * (1.0 + inner_factor) / 2.0
        y_m = radii[1] * np.outer(np.sin(u), np.sin(v)) * (1.0 + inner_factor) / 2.0
        z_m = (
            radii[2] * np.outer(np.ones_like(u), np.cos(v)) * (1.0 + inner_factor) / 2.0
        )

        # rotate accordingly
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = (
                    np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
                )
                [x_i[i, j], y_i[i, j], z_i[i, j]] = (
                    np.dot([x_i[i, j], y_i[i, j], z_i[i, j]], rotation) + center
                )
                [x_m[i, j], y_m[i, j], z_m[i, j]] = (
                    np.dot([x_m[i, j], y_m[i, j], z_m[i, j]], rotation) + center
                )

        if plotAxes:
            # make some purdy axes
            axes = np.array(
                [[radii[0], 0.0, 0.0], [0.0, radii[1], 0.0], [0.0, 0.0, radii[2]]]
            )
            # rotate accordingly
            for i in range(len(axes)):
                axes[i] = np.dot(axes[i], rotation)

            # plot axes
            for p in axes:
                X3 = np.linspace(-p[0], p[0], 100) + center[0]
                Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                ax.plot(X3, Y3, Z3, color=cageColor)

        # plot ellipsoid
        ax.plot_wireframe(
            x, y, z, rstride=4, cstride=4, color=cageColor, alpha=cageAlpha
        )

        ax.plot_wireframe(
            x_i, y_i, z_i, rstride=4, cstride=4, color="r", alpha=cageAlpha
        )
        if True:
            ax.plot_wireframe(
                x_m,
                y_m,
                z_m,
                rstride=4,
                cstride=4,
                color="black",
                alpha=0.9,
                linewidths=1,
            )

        if not (P is None):
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], color="g", marker="*", s=10)

        if make_ax:
            plt.show()
            plt.close(fig)
            del fig


def demo_ellipsoid():
    # make 100 random points
    P = np.reshape([random() * 100 for i in range(300)], (100, 3))

    # find the ellipsoid
    ET = Ellipsoid()
    (center, radii, rotation) = ET.get_params(P, 0.01)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # plot points
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color="g", marker="*", s=10)

    # plot ellipsoid
    ET.plot(center, radii, rotation, ax=ax, plotAxes=True)

    plt.show()
    plt.close(fig)
    del fig


if __name__ == "__main__":
    demo_ellipsoid()
