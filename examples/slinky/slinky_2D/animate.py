import numpy as np
import plotly.graph_objects as go


def animate(qs, n_nodes=None, frame_duration=50):
    """
    Animate rod configurations from qs.

    Parameters
    ----------
    qs : array-like
        Shape (T, dof)
    n_nodes : int or None
        Number of nodes. If None, inferred from dof via dof = 4*n_nodes - 1
    frame_duration : int
        Duration per frame in ms
    """
    qs = np.asarray(qs)

    if qs.ndim != 2:
        raise ValueError("qs must have shape (T, dof)")

    T, dof = qs.shape

    if n_nodes is None:
        if (dof + 1) % 4 != 0:
            raise ValueError(
                f"Cannot infer n_nodes from dof={dof}. Expected dof = 4*n_nodes - 1."
            )
        n_nodes = (dof + 1) // 4

    expected_dof = 4 * n_nodes - 1
    if dof != expected_dof:
        raise ValueError(
            f"Inconsistent n_nodes={n_nodes} for dof={dof}. Expected dof={expected_dof}."
        )

    # Position block starts for nodes only; skips edge DOFs naturally
    pos_starts = [4 * i for i in range(n_nodes)]

    # Collect all node coordinates across all timesteps for fixed axis limits
    all_coords = np.vstack([qs[:, s:s+3] for s in pos_starts])

    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    center = (mins + maxs) / 2.0

    max_range = np.max(maxs - mins)
    buffer = max(0.1 * max_range, 1e-6)

    plot_limit = (max_range / 2.0) + buffer
    x_range = [center[0] - plot_limit, center[0] + plot_limit]
    y_range = [center[1] - plot_limit, center[1] + plot_limit]
    z_range = [center[2] - plot_limit, center[2] + plot_limit]

    frames = []
    for t in range(T):
        row = qs[t]
        q_points = [row[s:s+3] for s in pos_starts]

        x = [p[0] for p in q_points]
        y = [p[1] for p in q_points]
        z = [p[2] for p in q_points]

        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="lines+markers",
                        line=dict(color="black", width=7),
                        marker=dict(size=5),
                    )
                ],
                name=str(t),
            )
        )

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=x_range, autorange=False),
                yaxis=dict(range=y_range, autorange=False),
                zaxis=dict(range=z_range, autorange=False),
                aspectmode="cube",
            ),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": frame_duration, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "type": "buttons",
                    "showactive": False,
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [f.name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": f.name,
                            "method": "animate",
                        }
                        for f in frames
                    ]
                }
            ],
        ),
        frames=frames,
    )

    return fig