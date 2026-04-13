from typing import List, Tuple, Type
from abc import ABC, abstractmethod

import torch

import sys, os
sys.path.append(os.path.abspath(os.path.abspath(__file__)))
import transforms


class Vertex:
    def __init__(self, idx: int, estimate: transforms.Transform) -> None:
        self.idx: int = idx
        self.estimate: transforms.Transform = estimate.copy()

    @classmethod
    def Homography(cls, idx: int, estimate: transforms.Homography) -> "Vertex":
        assert type(estimate) == transforms.Homography
        cls(idx, estimate)

    @classmethod
    def Affine(cls, idx: int, estimate: transforms.Affine) -> "Vertex":
        assert type(estimate) == transforms.Affine
        cls(idx, estimate)

    @classmethod
    def Sim3(cls, idx: int, estimate: transforms.Sim3) -> "Vertex":
        assert type(estimate) == transforms.Sim3
        cls(idx, estimate)

    def update(self, delta: torch.Tensor) -> None:
        assert delta.shape == (self.estimate.ndof,)
        delta_transform = type(self.estimate).from_tangent(delta)
        self.estimate = delta_transform @ self.estimate

    def copy(self):
        return self.__class__(
            self.idx,
            self.estimate.copy()
        )


def relative_transform_matrix(parent_est: transforms.Transform, child_est: transforms.Transform):
    return parent_est.inv().as_matrix @ child_est.as_matrix()


class Edge:
    def __init__(
        self,
        parent: Vertex,
        child: Vertex,
        transform: transforms.Transform,
        information: torch.Tensor|None = None
    ):
        self.parent: Vertex = parent
        self.child: Vertex = child
        self.transform: transforms.Transform = transform.copy()
        ndof = self.transform.ndof
        self.information: torch.Tensor = (
            information if information is not None
            else torch.eye(ndof)
        )
        
        assert self.information.shape == (ndof, ndof)

    @classmethod
    def Sim3(
        cls,
        parent: Vertex,
        child: Vertex,
        transform: transforms.Sim3,
        information: torch.Tensor|None = None
    ) -> "Edge":
        assert type(transform) == transforms.Sim3
        return cls(parent, child, transform, information)
    
    @classmethod
    def Affine(
        cls,
        parent: Vertex,
        child: Vertex,
        transform: transforms.Affine,
        information: torch.Tensor|None = None
    ) -> "Edge":
        assert type(transform) == transforms.Affine
        return cls(parent, child, transform, information)

    @classmethod
    def Homography(
        cls,
        parent: Vertex,
        child: Vertex,
        transform: transforms.Homography,
        information: torch.Tensor|None = None
    ) -> "Edge":
        assert type(transform) == transforms.Homography
        return cls(parent, child, transform, information)

    def copy(self):
        return self.__class__(
            self.parent.copy(),
            self.child.copy(),
            self.transform.copy(),
            self.information.clone()
        )

    def update(self) -> None:
        transfom_mat = relative_transform_matrix(
            self.parent.estimate,
            self.child.estimate
        )
        self.transform = type(self.transform).from_matrix(transfom_mat)


#FUTURE. Include to Edge class
class Residual(ABC):
    @abstractmethod
    def residual(
        self,
        parent_est: object, 
        child_est: object,
        meas: object
    ) -> torch.Tensor:
        """
        Computes residual in Lie algebra.
        r = Log(H_{i,j}^{-1} @ (H_i^{-1} H_j))
        parameters:
            - parent_est: SL4 matrix
            - child_est: SL4 matrix
            - meas: SL4 matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def edge_jacobian(self, edge: Edge) -> torch.Tensor:
        raise NotImplementedError()

    def edge_residual(self, edge: Edge) -> torch.Tensor:
        return self.residual(
            edge.parent.estimate,
            edge.child.estimate,
            edge.transform
        )

    def loss(self, edges: List[Edge]) -> None:
        acc = 0.0
        for edge in edges:
            residual = self.edge_residual(edge).unsqueeze(1) #m x 1
            edge_loss = residual.T @ edge.information @ residual
            acc = acc + edge_loss
        return acc


class Algorithm(ABC):
    def __init__(
        self,
        edges: List[Edge],
        vertices: List[Vertex],
        residual: Residual,
        eps: float = 1e-6
    ):
        self.residual: Residual = residual
        self.__edges: List[Edge] = edges
        self.__vertices: List[Vertex] = vertices
        self.eps: float = eps
        self.__pose_type = self.get_pose_type()

    def get_pose_type(self) -> Type[transforms.Transform]:
        if not self.__vertices:
            return None
        vertex_types = set(v.estimate for v in self.__vertices)
        assert len(vertex_types) == 1
        return vertex_types[0]

    def get_edges_type_list(self) -> List[Type[transforms.Transform]]:
        if not self.__edges:
            return []
        return list(set(e.transform for e in self.__edges))

    @property
    def edges(self) -> List[Edge]:
        return self.__edges

    @property
    def vertices(self) -> List[Vertex]:
        return self.__vertices
    
    @property
    def n(self) -> int:
        return len(self.__vertices)

    @property
    def m(self) -> int:
        #FUTURE. Allow different types of edges:
        return self.__edge_types_list[0].ndof

    #This assumes all edges are of the same type.
    def compute_h_and_b(self) -> Tuple[torch.Tensor, torch.Tensor]:
        #1. Compute the errors
        n, m = self.n, self.m #FUTURE. Allow different types of edges:

        H = torch.zeros(n, n, m, m) #FUTURE. Allow different types of edges:
        b = torch.zeros(n, m) #FUTURE. Allow different types of edges:

        for edge in self.__edges:
            residual = self.residual.edge_residual(edge) #FUTURE. Allow different types of edges. residual function will depend on edge
            jacob_i, jacob_j = self.residual.edge_jacobian(edge)
            parent_idx = edge.parent.idx
            child_idx = edge.child.idx
            omega = edge.information
            H[parent_idx, parent_idx] += jacob_i.T @ omega @ jacob_i
            H[parent_idx, child_idx] += jacob_i.T @ omega @ jacob_j
            H[child_idx, parent_idx] += jacob_j.T @ omega @ jacob_i
            H[child_idx, child_idx] += jacob_j.T @ omega @ jacob_j

            b[parent_idx] += jacob_i.T @ omega @ residual
            b[child_idx] += jacob_j.T @ omega @ residual

        #Fix the first node
        H[0, :] = 0
        H[:, 0] = 0
        H[0, 0] = torch.eye(m)
        b[0] = 0

        H = H.permute(0, 2, 1, 3).reshape(n*m, n*m)
        b = b.view(n * m)
        return H, b

    def step_size(self, delta: torch.Tensor) -> float:
        #FUTURE. Use backtracking or something like that
        return 1.0

    def update(self, delta: torch.Tensor) -> None:
        for i, vertex in enumerate(self.__vertices):
            vertex.update(delta[i])
        for edge in self.__edges:
            edge.update()

    @abstractmethod
    def optimize(self, n_iter: int) -> None:
        raise NotImplementedError

    def append_edge(self, edge: Edge) -> None:
        if len(self.__edges) > 0:
            assert type(edge.transform) == self.__pose_type
        self.edges.append(edge)

    def append_vertex(self, vertex: Vertex) -> None:
        assert vertex.idx == len(self.__vertices)
        self.__vertices.append(vertex)


class Optimizer:
    def __init__(
        self,
        residual_class: Type[Residual],
        algorithm_class: Type[Algorithm],
        vertices: List[Vertex] = [],
        edges: List[Edge] = [],
    ) -> None:
        self.algorithm: Algorithm = algorithm_class(
            edges,
            vertices,
            residual_class(),
        )

    @property
    def edges(self) -> List[Edge]:
        return self.algorithm.edges

    @property
    def vertices(self) -> List[Vertex]:
        return self.algorithm.vertices

    @property
    def residual(self) -> Residual:
        return self.algorithm.residual

    def append_vertex(self, vertex: Vertex) -> None:
        self.algorithm.append_vertex(vertex)

    def append_edge(self, edge: Edge) -> None:
        self.algorithm.append_edge(edge)

    def optimize(self, n_iter) -> None:
        self.algorithm.optimize(n_iter)
