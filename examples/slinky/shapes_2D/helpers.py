import numpy as np


def convert_linear_bc_to_direct_bc(
    input_path,
    output_path,
    lambdas=None,
):
    """
    Convert dataset saved with (qs, idx_b, xb_m, xb_c) into direct-BC format
    expected by the simplified Dataset loader:
        qs, xb, idx_b, lambdas

    Parameters
    ----------
    input_path : str
        Path to input .npz file containing:
            qs, idx_b, xb_m, xb_c
        and optionally lambdas.
    output_path : str
        Path to output .npz file to save:
            qs, xb, idx_b, lambdas
    lambdas : array-like or None
        If provided, use these lambdas.
        If None, tries to load 'lambdas' from input file.
        If not present, uses np.linspace(0, 1, T).

    Notes
    -----
    Supported input shapes:

    1) Multi-trajectory case:
        qs    : (n_traj, T, dof)
        xb_m  : (n_traj, n_b) or (n_traj, 1, n_b) or (n_traj, T, n_b)
        xb_c  : (n_traj, n_b) or (n_traj, 1, n_b) or (n_traj, T, n_b)

    2) Single-trajectory case:
        qs    : (T, dof)
        xb_m  : (n_b,) or (1, n_b) or (T, n_b)
        xb_c  : (n_b,) or (1, n_b) or (T, n_b)

    Output is always saved in multi-trajectory direct-BC form:
        qs       : (n_traj, T, dof)
        xb       : (n_traj, T, n_b)
        idx_b    : unchanged
        lambdas  : (T,)
    """
    data = np.load(input_path)

    qs = np.asarray(data["qs"])
    idx_b = np.asarray(data["idx_b"])
    xb_m = np.asarray(data["xb_m"])
    xb_c = np.asarray(data["xb_c"])

    # -----------------------------------------------------
    # Make qs always (n_traj, T, dof)
    # -----------------------------------------------------
    if qs.ndim == 2:
        qs = qs[None, ...]   # single trajectory -> multi-trajectory form
    elif qs.ndim != 3:
        raise ValueError(f"`qs` must have shape (T,dof) or (n_traj,T,dof), got {qs.shape}")

    n_traj, T, _ = qs.shape

    # -----------------------------------------------------
    # Get lambdas
    # -----------------------------------------------------
    if lambdas is None:
        if "lambdas" in data:
            lambdas = np.asarray(data["lambdas"])
        else:
            lambdas = np.linspace(0.0, 1.0, T)

    lambdas = np.asarray(lambdas)

    if lambdas.shape != (T,):
        raise ValueError(f"`lambdas` must have shape ({T},), got {lambdas.shape}")

    # -----------------------------------------------------
    # Helper: broadcast xb_m / xb_c to (n_traj, T, n_b)
    # -----------------------------------------------------
    def expand_bc_array(arr, name):
        arr = np.asarray(arr)

        # Case: single trajectory input
        if arr.ndim == 1:
            # (n_b,) -> (1, 1, n_b) -> broadcast
            arr = arr[None, None, :]
        elif arr.ndim == 2:
            # Could be (n_traj, n_b) OR (T, n_b) OR (1, n_b)
            if arr.shape[0] == n_traj:
                arr = arr[:, None, :]      # (n_traj, 1, n_b)
            elif arr.shape[0] == T:
                arr = arr[None, :, :]      # (1, T, n_b)
            elif arr.shape[0] == 1:
                arr = arr[None, :, :] if n_traj != 1 else arr[None, :, :]
            else:
                raise ValueError(
                    f"Cannot interpret shape of `{name}` = {arr.shape} "
                    f"for n_traj={n_traj}, T={T}"
                )
        elif arr.ndim == 3:
            pass
        else:
            raise ValueError(f"`{name}` must have ndim 1, 2, or 3, got shape {arr.shape}")

        # Now force broadcast to (n_traj, T, n_b)
        try:
            arr = np.broadcast_to(arr, (n_traj, T, arr.shape[-1]))
        except ValueError as e:
            raise ValueError(
                f"Could not broadcast `{name}` with shape {arr.shape} "
                f"to ({n_traj}, {T}, n_b)"
            ) from e

        return arr

    xb_m = expand_bc_array(xb_m, "xb_m")
    xb_c = expand_bc_array(xb_c, "xb_c")

    # -----------------------------------------------------
    # Build direct BC trajectory
    # xb(t) = xb_m * lambda(t) + xb_c
    # -----------------------------------------------------
    lambdas_reshaped = lambdas[None, :, None]   # (1, T, 1)
    xb = xb_m * lambdas_reshaped + xb_c         # (n_traj, T, n_b)

    # -----------------------------------------------------
    # Save converted file
    # -----------------------------------------------------
    np.savez(
        output_path,
        qs=qs,
        xb=xb,
        idx_b=idx_b,
        lambdas=lambdas,
    )

    print(f"Saved converted direct-BC dataset to: {output_path}")
    print(f"qs.shape      = {qs.shape}")
    print(f"xb.shape      = {xb.shape}")
    print(f"idx_b.shape   = {idx_b.shape}")
    print(f"lambdas.shape = {lambdas.shape}")