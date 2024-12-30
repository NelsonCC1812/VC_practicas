from torch.nn import PairwiseDistance

_dst = PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

dst = lambda t1,t2: _dst(t1,t2)