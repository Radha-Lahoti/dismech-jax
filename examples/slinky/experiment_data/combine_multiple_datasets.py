import numpy as np

def combine_multiple_datasets(file_list, out_file, verbose=True):
    """
    Combine multiple direct-BC datasets with variable trajectory lengths into one dataset.

    Expected saved fields in each input file:
        qs      : (n_traj, T, dof)
        xb      : (n_traj, T, n_b)
        idx_b   : (n_b,) or (n_traj, n_b)
        lambdas : (T,) or (n_traj, T)

    Output saved fields:
        qs      : (N_total, T_max, dof)
        xb      : (N_total, T_max, n_b)
        idx_b   : (n_b,)   if all files share same idx_b
                 or (N_total, n_b) if per-trajectory idx_b is needed
        lambdas : (N_total, T_max)
        valid   : (N_total, T_max)

    Padding strategy:
        - prepend padding in time dimension
        - qs, xb padded with edge values
        - lambdas padded with 0
        - valid is False on padded entries, True on original entries
    """
    if len(file_list) == 0:
        raise ValueError("file_list must contain at least one file.")

    datasets = [np.load(f) for f in file_list]

    # ----------------------------
    # Helpers
    # ----------------------------
    def normalize_lambdas(data_dict):
        """Return lambdas as shape (n_traj, T)."""
        qs = data_dict["qs"]
        lam = data_dict["lambdas"]
        n_traj = qs.shape[0]

        if lam.ndim == 1:
            # shared lambda vector -> repeat for each trajectory in this file
            lam = np.broadcast_to(lam[None, :], (n_traj, lam.shape[0]))
        elif lam.ndim == 2:
            if lam.shape[0] != n_traj:
                raise ValueError(
                    f"lambdas has shape {lam.shape}, but qs has {n_traj} trajectories."
                )
        else:
            raise ValueError(
                f"lambdas must have ndim 1 or 2, got shape {lam.shape}."
            )

        return lam

    def normalize_idx_b(data_dict):
        """Return idx_b as shape (n_traj, n_b)."""
        qs = data_dict["qs"]
        idx_b = data_dict["idx_b"]
        n_traj = qs.shape[0]

        if idx_b.ndim == 1:
            idx_b = np.broadcast_to(idx_b[None, :], (n_traj, idx_b.shape[0]))
        elif idx_b.ndim == 2:
            if idx_b.shape[0] != n_traj:
                raise ValueError(
                    f"idx_b has shape {idx_b.shape}, but qs has {n_traj} trajectories."
                )
        else:
            raise ValueError(
                f"idx_b must have ndim 1 or 2, got shape {idx_b.shape}."
            )

        return idx_b

    def process_data(data_dict, target_l):
        """
        Normalize and pad one dataset to target_l.
        Returns:
            qs_pad      : (n_traj, target_l, dof)
            xb_pad      : (n_traj, target_l, n_b)
            lam_pad     : (n_traj, target_l)
            valid       : (n_traj, target_l)
            idx_b_all   : (n_traj, n_b)
        """
        qs = data_dict["qs"]
        xb = data_dict["xb"]
        lam = normalize_lambdas(data_dict)
        idx_b_all = normalize_idx_b(data_dict)

        n_traj, current_l, dof = qs.shape
        _, xb_l, n_b = xb.shape

        if xb_l != current_l:
            raise ValueError(
                f"Inconsistent time dimension: qs has T={current_l}, xb has T={xb_l}."
            )
        if lam.shape[1] != current_l:
            raise ValueError(
                f"Inconsistent time dimension: qs has T={current_l}, lambdas has T={lam.shape[1]}."
            )

        diff = target_l - current_l
        if diff < 0:
            raise ValueError(
                f"target_l={target_l} is smaller than current_l={current_l}."
            )

        qs_pad = np.pad(qs, ((0, 0), (diff, 0), (0, 0)), mode="edge")
        xb_pad = np.pad(xb, ((0, 0), (diff, 0), (0, 0)), mode="edge")
        lam_pad = np.pad(lam, ((0, 0), (diff, 0)), mode="constant", constant_values=0)

        valid = np.concatenate(
            [
                np.zeros((n_traj, diff), dtype=bool),
                np.ones((n_traj, current_l), dtype=bool),
            ],
            axis=1,
        )

        return qs_pad, xb_pad, lam_pad, valid, idx_b_all

    # ----------------------------
    # Global checks
    # ----------------------------
    lengths = []
    dof_ref = None
    nb_ref = None

    for k, d in enumerate(datasets):
        qs = d["qs"]
        xb = d["xb"]

        if qs.ndim != 3:
            raise ValueError(f"{file_list[k]}: qs must have shape (n_traj, T, dof), got {qs.shape}")
        if xb.ndim != 3:
            raise ValueError(f"{file_list[k]}: xb must have shape (n_traj, T, n_b), got {xb.shape}")

        dof = qs.shape[2]
        n_b = xb.shape[2]
        T = qs.shape[1]

        if dof_ref is None:
            dof_ref = dof
        elif dof != dof_ref:
            raise ValueError(
                f"{file_list[k]}: dof mismatch. Expected {dof_ref}, got {dof}."
            )

        if nb_ref is None:
            nb_ref = n_b
        elif n_b != nb_ref:
            raise ValueError(
                f"{file_list[k]}: boundary size mismatch. Expected {nb_ref}, got {n_b}."
            )

        lengths.append(T)

    max_l = max(lengths)

    # ----------------------------
    # Process and collect
    # ----------------------------
    qs_blocks = []
    xb_blocks = []
    lam_blocks = []
    valid_blocks = []
    idx_blocks = []

    for file_name, d in zip(file_list, datasets):
        qs_i, xb_i, lam_i, valid_i, idx_i = process_data(d, max_l)

        qs_blocks.append(qs_i)
        xb_blocks.append(xb_i)
        lam_blocks.append(lam_i)
        valid_blocks.append(valid_i)
        idx_blocks.append(idx_i)

        if verbose:
            print(f"\nProcessed {file_name}")
            print("  qs      ", qs_i.shape)
            print("  xb      ", xb_i.shape)
            print("  lambdas ", lam_i.shape)
            print("  valid   ", valid_i.shape)
            print("  idx_b   ", idx_i.shape)

    # ----------------------------
    # Concatenate across trajectories
    # ----------------------------
    qs = np.concatenate(qs_blocks, axis=0)
    xb = np.concatenate(xb_blocks, axis=0)
    lambdas = np.concatenate(lam_blocks, axis=0)
    valid = np.concatenate(valid_blocks, axis=0)
    idx_b_all = np.concatenate(idx_blocks, axis=0)

    # If all trajectories share same idx_b, save compact 1D version.
    if np.all(idx_b_all == idx_b_all[0]):
        idx_b_out = idx_b_all[0]
    else:
        idx_b_out = idx_b_all

    if verbose:
        print("\nFINAL")
        print("  qs      ", qs.shape)
        print("  xb      ", xb.shape)
        print("  lambdas ", lambdas.shape)
        print("  valid   ", valid.shape)
        print("  idx_b   ", idx_b_out.shape)

    np.savez(
        out_file,
        qs=qs,
        xb=xb,
        idx_b=idx_b_out,
        lambdas=lambdas,
        valid=valid,
    )

    if verbose:
        print(f"\nCombined {len(file_list)} files into {out_file}.")


def combine_multiple_datasets_with_forces(file_list, out_file, force_key="F", verbose=True):
    """
    Combine multiple direct-BC datasets with force data and equal trajectory lengths.

    Expected saved fields in each input file:
        qs        : (n_traj, T, dof)
        xb        : (n_traj, T, n_b)
        idx_b     : (n_b,) or (n_traj, n_b)
        lambdas   : (T,) or (n_traj, T)
        force_key : (n_traj, T, ...) or (n_traj, T)
        valid     : (n_traj, T), optional

    Output saved fields:
        qs        : (N_total, T, dof)
        xb        : (N_total, T, n_b)
        idx_b     : (n_b,) if all files share same idx_b,
                    otherwise (N_total, n_b)
        lambdas   : (N_total, T)
        valid     : (N_total, T)
        force_key : (N_total, T, ...) or (N_total, T)

    Unlike combine_multiple_datasets, this function does not pad. All datasets
    must have exactly the same trajectory length T.
    """
    if len(file_list) == 0:
        raise ValueError("file_list must contain at least one file.")

    datasets = [np.load(f) for f in file_list]

    def normalize_lambdas(data_dict, file_name):
        qs = data_dict["qs"]
        lam = data_dict["lambdas"]
        n_traj = qs.shape[0]
        T = qs.shape[1]

        if lam.ndim == 1:
            if lam.shape[0] != T:
                raise ValueError(
                    f"{file_name}: lambdas has length {lam.shape[0]}, but qs has T={T}."
                )
            lam = np.broadcast_to(lam[None, :], (n_traj, T))
        elif lam.ndim == 2:
            if lam.shape != (n_traj, T):
                raise ValueError(
                    f"{file_name}: lambdas must have shape {(n_traj, T)}, got {lam.shape}."
                )
        else:
            raise ValueError(
                f"{file_name}: lambdas must have ndim 1 or 2, got shape {lam.shape}."
            )

        return lam

    def normalize_idx_b(data_dict, file_name):
        qs = data_dict["qs"]
        idx_b = data_dict["idx_b"]
        n_traj = qs.shape[0]

        if idx_b.ndim == 1:
            idx_b = np.broadcast_to(idx_b[None, :], (n_traj, idx_b.shape[0]))
        elif idx_b.ndim == 2:
            if idx_b.shape[0] != n_traj:
                raise ValueError(
                    f"{file_name}: idx_b has shape {idx_b.shape}, but qs has {n_traj} trajectories."
                )
        else:
            raise ValueError(
                f"{file_name}: idx_b must have ndim 1 or 2, got shape {idx_b.shape}."
            )

        return idx_b

    def get_valid(data_dict, file_name):
        qs = data_dict["qs"]
        expected_shape = qs.shape[:2]

        if "valid" not in data_dict.files:
            return np.ones(expected_shape, dtype=bool)

        valid = data_dict["valid"]
        if valid.shape != expected_shape:
            raise ValueError(
                f"{file_name}: valid must have shape {expected_shape}, got {valid.shape}."
            )

        return valid

    qs_blocks = []
    xb_blocks = []
    force_blocks = []
    lam_blocks = []
    valid_blocks = []
    idx_blocks = []

    T_ref = None
    dof_ref = None
    nb_ref = None
    force_trailing_ref = None

    for file_name, d in zip(file_list, datasets):
        missing = [key for key in ("qs", "xb", "idx_b", "lambdas", force_key) if key not in d.files]
        if missing:
            raise ValueError(f"{file_name}: missing required field(s): {missing}.")

        qs = d["qs"]
        xb = d["xb"]
        forces = d[force_key]

        if qs.ndim != 3:
            raise ValueError(f"{file_name}: qs must have shape (n_traj, T, dof), got {qs.shape}.")
        if xb.ndim != 3:
            raise ValueError(f"{file_name}: xb must have shape (n_traj, T, n_b), got {xb.shape}.")
        if forces.ndim < 2:
            raise ValueError(
                f"{file_name}: {force_key} must have shape (n_traj, T, ...) or (n_traj, T), got {forces.shape}."
            )

        n_traj, T, dof = qs.shape
        xb_shape = xb.shape[:2]
        force_shape = forces.shape[:2]

        if xb_shape != (n_traj, T):
            raise ValueError(
                f"{file_name}: xb leading shape {xb_shape} must match qs leading shape {(n_traj, T)}."
            )
        if force_shape != (n_traj, T):
            raise ValueError(
                f"{file_name}: {force_key} leading shape {force_shape} must match qs leading shape {(n_traj, T)}."
            )

        if T_ref is None:
            T_ref = T
            dof_ref = dof
            nb_ref = xb.shape[2]
            force_trailing_ref = forces.shape[2:]
        else:
            if T != T_ref:
                raise ValueError(
                    f"{file_name}: trajectory length T={T} does not match expected T={T_ref}. "
                    "This function only combines equal-length trajectories."
                )
            if dof != dof_ref:
                raise ValueError(f"{file_name}: dof mismatch. Expected {dof_ref}, got {dof}.")
            if xb.shape[2] != nb_ref:
                raise ValueError(
                    f"{file_name}: boundary size mismatch. Expected {nb_ref}, got {xb.shape[2]}."
                )
            if forces.shape[2:] != force_trailing_ref:
                raise ValueError(
                    f"{file_name}: {force_key} trailing shape mismatch. "
                    f"Expected {force_trailing_ref}, got {forces.shape[2:]}."
                )

        lambdas = normalize_lambdas(d, file_name)
        idx_b = normalize_idx_b(d, file_name)
        valid = get_valid(d, file_name)

        qs_blocks.append(qs)
        xb_blocks.append(xb)
        force_blocks.append(forces)
        lam_blocks.append(lambdas)
        valid_blocks.append(valid)
        idx_blocks.append(idx_b)

        if verbose:
            print(f"\nProcessed {file_name}")
            print("  qs      ", qs.shape)
            print("  xb      ", xb.shape)
            print(f"  {force_key:<8}", forces.shape)
            print("  lambdas ", lambdas.shape)
            print("  valid   ", valid.shape)
            print("  idx_b   ", idx_b.shape)

    qs = np.concatenate(qs_blocks, axis=0)
    xb = np.concatenate(xb_blocks, axis=0)
    forces = np.concatenate(force_blocks, axis=0)
    lambdas = np.concatenate(lam_blocks, axis=0)
    valid = np.concatenate(valid_blocks, axis=0)
    idx_b_all = np.concatenate(idx_blocks, axis=0)

    if np.all(idx_b_all == idx_b_all[0]):
        idx_b_out = idx_b_all[0]
    else:
        idx_b_out = idx_b_all

    if verbose:
        print("\nFINAL")
        print("  qs      ", qs.shape)
        print("  xb      ", xb.shape)
        print(f"  {force_key:<8}", forces.shape)
        print("  lambdas ", lambdas.shape)
        print("  valid   ", valid.shape)
        print("  idx_b   ", idx_b_out.shape)

    np.savez(
        out_file,
        qs=qs,
        xb=xb,
        idx_b=idx_b_out,
        lambdas=lambdas,
        valid=valid,
        **{force_key: forces},
    )

    if verbose:
        print(f"\nCombined {len(file_list)} files into {out_file}.")
