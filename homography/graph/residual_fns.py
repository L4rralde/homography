import torch

from core import Residual, Edge
import sys, os
sys.path.append(os.path.basename(os.path.basename(__file__)))
from sl4 import SL4


class ResidualSL4(Residual):
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
        return (meas.inv() @ prediction).Log()

    def edge_jacobian(self, edge: Edge) -> torch.Tensor:
        parent_pose = torch.from_numpy(
            edge.parent.estimate.as_matrix()
        ).requires_grad_(True)
        parent_est_sl4 = SL4(parent_pose)
        child_pose = torch.from_numpy(
            edge.child.estimate.as_matrix()
        ).requires_grad_(True)
        child_est_sl4 = SL4(child_pose)
        meas = torch.from_numpy(edge.transform.as_matrix())
        meas_sl4 = SL4(meas)
    
        zero_perturbation = torch.zeros(self.transform_type.ndof)
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
