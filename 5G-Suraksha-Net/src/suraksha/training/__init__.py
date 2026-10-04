"""Training-side tooling for the temporal fight classifier.

Kept separate from `suraksha.fight` (which is the inference path) but sharing the
two things that MUST NOT drift: the network definition (`training.model`) and the
per-frame feature vector (`fight.candidate.FightCandidateDetector._pair_features`).
"""
