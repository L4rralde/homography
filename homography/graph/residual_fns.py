import torch

from .core import Residual, Edge
from ..sl4 import SL4


class ResidualSL4(Residual):
    ndof = 15
    def residual(
        self,
        parent_est: SL4, 
        child_est: SL4,
        meas: SL4
    ) -> torch.Tensor:
        """
        Computes residual in Lie algebra.
        r = Log(H_{i,j}^{-1} @ (H_i^{-1} H_j))
        parameters:
            - parent_est: SL4 matrix
            - child_est: SL4 matrix
            - meas: SL4 matrix
        """
        prediction = parent_est.inv() @ child_est
        delta_sl4 = meas.inv() @ prediction
        return (delta_sl4).Log()

    def edge_residual(self, edge: Edge) -> torch.Tensor:
        parent_est = edge.parent.estimate.as_matrix()
        child_est = edge.child.estimate.as_matrix()
        meas = edge.transform.as_matrix()
        return self.residual(
            SL4(torch.from_numpy(parent_est)),
            SL4(torch.from_numpy(child_est)),
            SL4(torch.from_numpy(meas))
        )

    def edge_jacobian(self, edge: Edge) -> torch.Tensor:
        parent_pose = torch.from_numpy(
            edge.parent.estimate.as_matrix()
        )
        parent_est_sl4 = SL4(parent_pose)
        child_pose = torch.from_numpy(
            edge.child.estimate.as_matrix()
        )
        child_est_sl4 = SL4(child_pose)
        meas = torch.from_numpy(edge.transform.as_matrix())
        meas_sl4 = SL4(meas)
    
        zero_perturbation = torch.zeros(self.ndof)
        parent_j = torch.autograd.functional.jacobian(
            lambda d: self.residual(
                SL4.Exp(d) @ parent_est_sl4,
                child_est_sl4,
                meas_sl4
            ),
            zero_perturbation
        )
        child_j = torch.autograd.functional.jacobian(
            lambda d: self.residual(
                parent_est_sl4,
                SL4.Exp(d) @ child_est_sl4,
                meas_sl4
            ),
            zero_perturbation
        )
        return parent_j, child_j
