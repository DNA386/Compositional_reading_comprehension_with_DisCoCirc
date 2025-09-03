"""Utils for visualising (single qubit per noun) models."""

from typing import List, Tuple, Dict
from lambeq.backend.quantum import Diagram as Circuit

import qutip
import math
import torch
import matplotlib.pyplot as plt


# Type aliases
CartesianPoint = List[float]  # Len = 3
State1q = torch.Tensor
DensityMatrix1q = torch.Tensor
DensityMatrix2q = torch.Tensor

RGBCol = Tuple[float, float, float]
RGBACol = Tuple[float, float, float, float]
HexCol = str
Color = HexCol | RGBCol | RGBACol


def get_hex_from_rgb(*rgb: float) -> HexCol:
    if len(rgb) == 3:
        return '#%02x%02x%02x' % tuple(round(c * 255) for c in rgb)
    else:
        return '#%02x%02x%02x%02x' % tuple(round(c * 255) for c in rgb)


def eigenterpolate(U0, U1, s):
    """Interpolates between two matrices."""
    return U0 * eigenpow(U0.H * U1, s)


def eigenteroterpolate(U, s):
    """Compute an incremental version of the rotation U."""
    U_s = eigenpow(U, s)
    return U_s


def eigenpow(M, t):
    """Raises a matrix to a power."""
    return eigenlift(lambda b: b**t, M)


def eigenlift(f, M):
    """Lifts a numeric function to apply it to a matrix."""
    w, v = torch.linalg.eig(M)
    T = torch.zeros_like(M)
    for i in range(len(w)):
        eigen_val = w[i]
        eigen_vec = v[:, i].detach().clone()
        eigen_mat = torch.outer(eigen_vec, eigen_vec.conj())
        T += f(eigen_val) * eigen_mat
    return T


def get_bloch_point_from_dm(rho: DensityMatrix1q) -> CartesianPoint | None:
    """get cartesian bloch sphere coords from a density matrix"""
    if torch.allclose(rho, torch.zeros_like(rho)):
        return None
    point = [
        2.0 * rho[1][0].real,
        2.0 * rho[1][0].imag,
        2.0 * rho[0][0].real - 1.0,
    ]
    return point


def get_bloch_point_from_state(state: State1q) -> CartesianPoint | None:
    """get cartesian bloch sphere coords from a pure state"""
    alpha, beta = state
    # If this is zero, don't draw it.
    if torch.allclose(state, torch.zeros_like(state)):
        return None
    # Get polar form first
    r = 1  # because this must be a pure state.
    theta = 2 * math.acos(
        # Cap to be within the bounds...
        min(max(math.sqrt(alpha.imag ** 2 + alpha.real ** 2), -1), 1)
    )
    phi_alpha, phi_beta = tuple(
        math.atan(
            x.imag / (x.real if abs(x.real) > 1e-8 else math.copysign(1e-8, x.real))
        ) + (torch.pi if x.real < 0 else 0.)
        for x in state
    )
    phi = phi_beta - phi_alpha

    # now convert to bloch sphere xyz coords
    point = [
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        r * math.cos(theta),
    ]
    return point


def get_bloch_points(points: List[State1q | DensityMatrix1q]) -> List[CartesianPoint | None]:
    return [
        get_bloch_point_from_dm(point) if len(point.shape) > 1 else get_bloch_point_from_state(point)
        for point in points
    ]


def plot_points(
        complex_points: List[State1q | DensityMatrix1q],
        arcs=None, fig=None, axes=None, show=False, colors=None,
        markers=None, annotations=None, vectors=None, vector_colors=None,
        axis_labels=False,
):
    """Plot a set of states/density matrices onto the bloch sphere."""
    # convert into 3d coords
    points = get_bloch_points(complex_points)
    vector_points = get_bloch_points(vectors) if vectors else None
    cols = [col for i, col in enumerate(colors) if i >= len(points) or points[i] is not None] if colors is not None else None
    vector_cols = [col for i, col in enumerate(vector_colors) if i >= len(vector_points) or vector_points[i] is not None] if vector_colors is not None else None
    marks = [mark for i, mark in enumerate(markers) if i >= len(points) or points[i] is not None] if markers is not None else None

    b = qutip.Bloch(fig=fig, axes=axes)

    # set up axis labels
    if axis_labels:
        b.zlpos = [1.3, -1.4]
        b.zlabel = [r'$|\,0\,\rangle$', r'$|\,1\,\rangle$']
        b.xlpos = [1.6, -1.9]
        b.xlabel = [r'$|+\rangle$', r'$|-\rangle$']
        b.ylpos = [1.2, -1.4]
        b.ylabel = [r'$|\,i\,\rangle$', r'$|-i\,\rangle$']
    else:
        b.zlabel = ['', '']
        b.xlabel = ['', '']
        b.ylabel = ['', '']

    if colors:
        b.point_color = cols
    if vector_cols:
        b.vector_color = vector_cols
    if markers:
        b.point_marker = marks
    else:
        b.point_marker = ['o']
    b.point_size = [10]
    b.vector_width = 2
    b.font_size = 12
    b.sphere_alpha = 0.01
    b.view = [-30, 60]

    n_skipped = len([p for p in points if p is None])
    if n_skipped > 0:
        print("skipping", n_skipped)

    for i, point in enumerate(points):
        if point is not None:
            b.add_points(point, colors=[colors[i]])

    if annotations is not None:
        for annotation in annotations:
            if annotation is not None:
                p = annotation[0]
                b.add_annotation(
                    [
                        item + (0.03 if item > 0 else -0.03)
                        for item in p
                    ],
                    **{
                        **{
                            "horizontalalignment": "left" if p[1] > 0 else "right",
                            "verticalalignment": "bottom" if p[2] > 0 else "top",
                        },
                        **annotation[1],
                    }
                )

    if arcs is not None:
        for arc in arcs:
            bloch_arc = get_bloch_points(arc[:2])
            if not torch.allclose(torch.as_tensor(bloch_arc[0]), torch.as_tensor(bloch_arc[1])):
                b.add_arc(*bloch_arc, **arc[2])

    if vector_points is not None:
        b.add_vectors(
            [v for v in vector_points if v is not None],
            colors=vector_cols
        )

    if show:
        b.show()
    else:
        b.render()


def get_remote_prep_marker_states() -> List[State1q]:
    zero = torch.tensor([1. + 0j, 0.], dtype=torch.cdouble)
    one = torch.tensor([0. + 0j, 1.], dtype=torch.cdouble)
    return [
        zero,
        one,
        (zero + one) / math.sqrt(2.),  # -
        (zero - one) / math.sqrt(2.),  # +
        (zero + 1j * one) / math.sqrt(2.),  # i
        (zero - 1j * one) / math.sqrt(2.),  # -i
    ]


def get_remote_prep_states(n: int = 30) -> List[State1q]:
    """Generate a list of pure states that cover the bloch sphere. Density is controlled by n."""
    remote_preparation_states = [
        torch.tensor([
            torch.cos(alpha / 2),
            torch.sin(alpha / 2) * torch.exp(beta * 1j),
        ], dtype=torch.cdouble)
        for alpha in torch.linspace(0, 2 * torch.pi, 2 * n)
        for beta in torch.linspace(0, 2 * torch.pi, n)
    ]
    return remote_preparation_states


def get_sphere_colours(points: List[CartesianPoint]) -> List[Color]:
    """Colour states according to where they are on the bloch sphere."""
    return[
        (
            round((x + 1) / 2, 2),
            round((y + 1) / 2, 2),
            round((z + 1) / 2, 2),
        )
        for x, y, z in points
    ]


def remote_prepare(state: DensityMatrix2q, qubit_index: int, remote_prep: State1q) -> DensityMatrix1q:
    """
    Prepare qubit according to remote prep.
    Expect:
        1: remote_prep is pure
        2: state is mixed
        3: return a mixed single-qubit state
    """
    full_remote_prep = torch.kron(remote_prep, remote_prep.conj()).reshape([2, 2])
    marginal = torch.einsum(
        full_remote_prep, [0 + qubit_index, 2 + qubit_index], state, [0, 1, 2, 3], [1 - qubit_index, 3 - qubit_index]
    )
    # Re-normalise
    norm_factor = torch.trace(marginal)
    if torch.allclose(norm_factor, torch.zeros_like(norm_factor)):
        # Avoid division by 0. This state has probability approx 0 of occurring, so we won't draw it anyway.
        return marginal
    return marginal / norm_factor


def get_2qubit_sphere_points(
        state: DensityMatrix2q, n: int = 30, remote_preparation_states: List[State1q] = None
) -> Tuple[Dict[int, List[DensityMatrix1q]], List[Color]]:
    """Get the data for visualising a 2qubit state on a pair of bloch spheres."""
    if remote_preparation_states is None:
        remote_preparation_states = get_remote_prep_states(n)
    colour_order = get_sphere_colours(get_bloch_points(remote_preparation_states))

    projected_states = {}
    for qubit_index in [0, 1]:
        projected_states[qubit_index] = [
            # Prepare the other qubit in the specified state, and look at where this one ends up.
            remote_prepare(state, 1 - qubit_index, s) for s in remote_preparation_states
        ]
    return projected_states, colour_order


def display_1q_rotation(
        op, lambdeval: callable, figure: plt.figure, ax,
        title="", title_kwargs={}, as_surface=False, steps: int = 10, axis_labels=False,
):
    op_v = lambdeval(op)
    if as_surface:
        states = get_remote_prep_states(10)
        cols = get_sphere_colours(get_bloch_points(states))
        points = [
            torch.einsum(op_v, [0, 1], p, [0], [1])
            for p in states
        ]
        plot_points(
            points,
            fig=figure,
            axes=ax,
            colors=cols,
            axis_labels=axis_labels,
        )
    else:
        states_v = get_remote_prep_marker_states()
        cols_v = get_sphere_colours(get_bloch_points(states_v))
        points_v = [
            torch.einsum(op_v, [0, 1], p, [0], [1])
            for p in states_v
        ]
        point_arcs = [
            [
                torch.einsum(eigenteroterpolate(op_v, step / steps), [0, 1], state, [0], [1])
                for step in range(steps + 1)
            ]
            for state in states_v
        ]

        plot_points(
            points_v,
            fig=figure,
            axes=ax,
            colors=cols_v,
            arcs=[
                (start, end, {"fmt": get_hex_from_rgb(r, g, b, (j + 2) / (steps + 2))})
                for i, (r, g, b) in enumerate(cols_v)
                for j, (start, end) in enumerate(zip(point_arcs[i][:-1], point_arcs[i][1:]))
            ],
            vectors=points_v,
            vector_colors=cols_v,
            axis_labels=axis_labels,
        )
    ax.axis('off')
    ax.set_title(title, fontsize="x-large", **title_kwargs)


def display_2q_state(
        diag: Circuit, lambdeval: callable, figure: plt.figure, n_samples: int = 30, title: str = "",
        axis_labels: bool = False, annotations: list = [], annotation_points: List[State1q | DensityMatrix1q] = []
):
    figure.suptitle(title, fontsize="x-large")
    fig = figure.subfigures(1, 1)
    fig.patch.set_linewidth(0.5)
    fig.patch.set_edgecolor('#ccc')

    joint_state = torch.as_tensor(lambdeval(diag), dtype=torch.cdouble)
    projected_states, cols = get_2qubit_sphere_points(joint_state, n=n_samples)

    marker_states = get_remote_prep_marker_states()
    marker_shapes = ["o" for _ in range(len(projected_states[0]))] + ["d" for _ in range(len(marker_states))]
    vectors, marker_cols = get_2qubit_sphere_points(joint_state, remote_preparation_states=marker_states)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_points(
        projected_states[0] + marker_states + annotation_points,
        fig=fig,
        axes=ax1,
        colors=cols + marker_cols,
        markers=marker_shapes,
        vectors=vectors[0],
        vector_colors=marker_cols,
        axis_labels=axis_labels,
        annotations=[None for _ in range(len(projected_states[0]) + len(marker_states))] + annotations
    )
    ax1.axis('off')
    ax1.set_title(r"$\rho_{1, \psi}$", fontsize="x-large", y=-0.01)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_points(
        projected_states[1] + marker_states + annotation_points,
        fig=fig,
        axes=ax2,
        colors=cols + marker_cols,
        markers=marker_shapes,
        vectors=vectors[1],
        vector_colors=marker_cols,
        axis_labels=axis_labels,
        annotations=[None for _ in range(len(projected_states[0]) + len(marker_states))] + annotations
    )
    ax2.axis('off')
    ax2.set_title(r"$\rho_{\psi, 2}$", fontsize="x-large", y=-0.01)
