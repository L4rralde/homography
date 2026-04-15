from typing import Type

import torch

from homography.graph.core import Vertex
from homography import transforms
from .core import Edge
from ..sl4 import SL4, SL4Affine


class EdgeSL4(Edge):
    def __init__(
        self,
        parent: Vertex,
        child: Vertex,
        transform: transforms.Homography,
        information: torch.Tensor = None
    ):
        assert isinstance(transform, transforms.Homography)
        super().__init__(parent, child, transform, information)
        self.lie_group: Type[SL4] = SL4

    @property
    def ndof(self) -> int:
        return self.lie_group.ndof

    def residual_fn(self, parent_est: SL4, child_est: SL4, meas: SL4) -> torch.Tensor:
        """
        Computes residual in Lie algebra.
        r = Log(H_{i,j}^{-1} @ (H_i^{-1} H_j))
        parameters:
            - parent_est: SL4 matrix
            - child_est: SL4 matrix
            - meas: SL4 matrix
        """
        prediction = parent_est.inv() @ child_est
        delta_transform = meas.inv() @ prediction
        return (delta_transform).Log()

    def edge_residual(self) -> torch.Tensor:
        parent_est = self.parent.estimate.as_matrix()
        child_est = self.child.estimate.as_matrix()
        meas = self.transform.as_matrix()
        return self.residual(
            self.lie_group(torch.from_numpy(parent_est)),
            self.lie_group(torch.from_numpy(child_est)),
            self.lie_group(torch.from_numpy(meas))
        )

    def edge_jacobian(self) -> torch.Tensor:
        parent_pose = torch.from_numpy(
            self.parent.estimate.as_matrix()
        )
        parent_est_in_lie = self.lie_group(parent_pose)

        child_pose = torch.from_numpy(
            self.child.estimate.as_matrix()
        )
        child_est_in_lie = self.lie_group(child_pose)

        meas = torch.from_numpy(self.transform.as_matrix())
        meas_in_lie = self.lie_group(meas)

        zero_perturbation = torch.zeros(self.ndof)
        parent_j = torch.autograd.functional.jacobian(
            lambda d: self.residual(
                self.lie_group.Exp(d) @ parent_est_in_lie,
                child_est_in_lie,
                meas_in_lie
            ),
            zero_perturbation
        )
        child_j = torch.autograd.functional.jacobian(
            lambda d: self.residual(
                parent_est_in_lie,
                SL4.Exp(d) @ child_est_in_lie,
                meas_in_lie
            ),
            zero_perturbation
        )

        return parent_j, child_j


class EdgeSL4Affine(EdgeSL4):
    def __init__(
        self,
        parent: Vertex,
        child: Vertex,
        transform: transforms.Affine,
        information: torch.Tensor = None
    ):
        assert isinstance(transform, transforms.Affine)
        super().__init__(parent, child, transform, information)
        self.lie_group: Type[SL4Affine] = SL4Affine
