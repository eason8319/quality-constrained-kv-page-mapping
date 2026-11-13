"""HiGHS multiple-choice, aligned-byte-budget allocation."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import eye, kron, vstack, csr_matrix


def solve(scores,costs,budget):
    scores=np.asarray(scores,dtype=float)
    costs=np.broadcast_to(costs,scores.shape)
    pages,choices=scores.shape
    one=kron(eye(pages),np.ones((1,choices)))
    constraints=LinearConstraint(vstack([one,csr_matrix(costs.reshape(1,-1))]),
                                 [1]*pages+[-np.inf],[1]*pages+[budget])
    result=milp(scores.ravel(),integrality=np.ones(scores.size),bounds=Bounds(0,1),
                constraints=constraints,options={"mip_rel_gap":0.0})
    if not result.success:
        raise ValueError(result.message)
    selected=result.x.reshape(scores.shape).argmax(axis=1)
    rows=np.arange(pages)
    return dict(choices=selected.tolist(),objective=float(scores[rows,selected].sum()),
                bytes=int(costs[rows,selected].sum()),gap=float(result.mip_gap))
