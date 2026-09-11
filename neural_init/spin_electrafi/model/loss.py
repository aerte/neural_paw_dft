import torch
from ase import Atoms


class DensityLoss:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.use_wandb = config["wandb"]
        self.density_loss_fn = torch.nn.L1Loss(reduction="sum")
        self.r2loss = torch.nn.MSELoss()
        self.loss_type = config["loss_type"]

    def compute_integrated_error(
        self, pred_cd, true_cd_tens, dv, n_valence_electrons, sampled_points=None
    ):
        # pred_cd is either full grid (same shape as true) OR a 1D slice (len = |sampled_points|)
        pred_cd = pred_cd.to(self.device)
        true_cd_tens = true_cd_tens.to(self.device)

        if sampled_points is None:
            # full-grid path
            delta = torch.abs(pred_cd - true_cd_tens).sum()
            denom = true_cd_tens.sum()
        else:
            # subsampled path: compare on the same indices, and normalize by the true sum on that slice
            t_flat = true_cd_tens.flatten()
            t_sel = t_flat.index_select(0, sampled_points)
            p_sel = pred_cd.flatten()
            delta = torch.abs(p_sel - t_sel).sum()
            denom = t_sel.sum()

        denom = denom.clamp_min(1e-12)
        NMAE = delta / denom
        return NMAE

    def compute_R2_loss(
        self,
        pred_cd: torch.Tensor,
        true_cd_tens: torch.Tensor,
        sampled_points: torch.Tensor | None = None,
    ):
        pred_cd = pred_cd.to(self.device).flatten()
        if sampled_points is None:
            true_vec = true_cd_tens.to(self.device).flatten()
        else:
            true_vec = (
                true_cd_tens.to(self.device).flatten().index_select(0, sampled_points)
            )
        return self.r2loss(pred_cd, true_vec)

    def compute_mae_loss(
        self,
        pred_cd: torch.Tensor,
        true_cd_tens: torch.Tensor,
        sampled_points: torch.Tensor | None = None,
    ):
        pred_cd = pred_cd.to(self.device).flatten()
        if sampled_points is None:
            true_vec = true_cd_tens.to(self.device).flatten()
        else:
            true_vec = (
                true_cd_tens.to(self.device).flatten().index_select(0, sampled_points)
            )
        loss = torch.abs(pred_cd - true_vec).mean()
        return loss

    def compute_total_loss(
        self,
        sys: Atoms,
        pred_cd: torch.tensor,
        true_cd_tens: torch.tensor,
        grid_dict: dict,
        n_valence_electrons: int,
        sampled_points: torch.tensor = None,
        training: bool = True,
        volume: float = None,
    ):
        pred_cd = pred_cd.to(self.device)
        true_cd_tens = true_cd_tens.to(self.device)
        if sampled_points is not None:
            sampled_points = sampled_points.to(self.device)

        grid_points = grid_dict["nx"] * grid_dict["ny"] * grid_dict["nz"]
        dv = (volume if volume is not None else sys.get_volume()) / grid_points

        integrated_error = self.compute_integrated_error(
            pred_cd, true_cd_tens, dv, n_valence_electrons, sampled_points
        )
        integrated_error = integrated_error.to(self.device)

        if self.loss_type == "R2":
            loss = self.compute_R2_loss(
                pred_cd, true_cd_tens, sampled_points
            )  # <- slice-aware
        elif self.loss_type == "mae":
            loss = self.compute_mae_loss(pred_cd, true_cd_tens, sampled_points)
        else:
            loss = integrated_error

        return loss, integrated_error




class SpinDifferenceLoss:
    """Loss for the signed spin / charge-difference density m(r) = rho_a - rho_b.

    Two regimes, both scale-carrying (unlike a cosine similarity, which leaves the
    amplitude of the prediction completely undetermined):

      - magnetic  (int|m| dv >= mag_min):  NMAE normalised by sum|m_true|.
        NOT by sum(m_true) -- for a signed field that is the *net* moment, which is
        zero for an antiferromagnet and can be negative, making NMAE unbounded and
        sign-flipped.
      - non-magnetic:                      absolute L1 against ~zero, so the head
        actually learns to output nothing where there is no spin density.
    """

    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.mag_min = float(config.get("spin_mag_min", 0.1))  # electrons

    @staticmethod
    def _align(pred_cd, true_cd_tens, sampled_points, device):
        pred = pred_cd.to(device).flatten()
        true = true_cd_tens.to(device).flatten()
        if sampled_points is not None:
            true = true.index_select(0, sampled_points.to(device))
        return pred, true

    def compute_spin_loss(
        self,
        pred_cd: torch.Tensor,
        true_cd_tens: torch.Tensor,
        dv: float,
        sampled_points: torch.Tensor | None = None,
    ):
        """Returns (loss, nmae, abs_err_electrons).

        nmae is None for the non-magnetic branch, where it is not meaningful.

        With sampled_points set, the magnetic/non-magnetic decision is made on the
        sampled slice rather than the full grid, so it is only exact for full-grid
        training (*_sample_frac: 0, which is what the spin config uses).
        """
        pred, true = self._align(pred_cd, true_cd_tens, sampled_points, self.device)

        delta = torch.abs(pred - true).sum()
        abs_err = delta * dv  # electrons, directly interpretable
        m_abs = torch.abs(true).sum()

        if float(m_abs.detach()) * dv >= self.mag_min:
            nmae = delta / m_abs.clamp_min(1e-12)
            return nmae, nmae, abs_err

        return torch.abs(pred - true).mean(), None, abs_err
