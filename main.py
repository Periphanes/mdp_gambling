import mdptoolbox, mdptoolbox.example
import numpy as np

testMDP = mdptoolbox.mdp.MDP(1,1,1,1,1)

# MDP Class Docs

"""

A Markov Decision Problem.

Let ``S`` = the number of states, and ``A`` = the number of acions.

Parameters
----------
transitions : array
    Transition probability matrices. These can be defined in a variety of
    ways. The simplest is a numpy array that has the shape ``(A, S, S)``,
    though there are other possibilities. It can be a tuple or list or
    numpy object array of length ``A``, where each element contains a numpy
    array or matrix that has the shape ``(S, S)``. This "list of matrices"
    form is useful when the transition matrices are sparse as
    ``scipy.sparse.csr_matrix`` matrices can be used. In summary, each
    action's transition matrix must be indexable like ``transitions[a]``
    where ``a`` ∈ {0, 1...A-1}, and ``transitions[a]`` returns an ``S`` ×
    ``S`` array-like object.
reward : array
    Reward matrices or vectors. Like the transition matrices, these can
    also be defined in a variety of ways. Again the simplest is a numpy
    array that has the shape ``(S, A)``, ``(S,)`` or ``(A, S, S)``. A list
    of lists can be used, where each inner list has length ``S`` and the
    outer list has length ``A``. A list of numpy arrays is possible where
    each inner array can be of the shape ``(S,)``, ``(S, 1)``, ``(1, S)``
    or ``(S, S)``. Also ``scipy.sparse.csr_matrix`` can be used instead of
    numpy arrays. In addition, the outer list can be replaced by any object
    that can be indexed like ``reward[a]`` such as a tuple or numpy object
    array of length ``A``.
discount : float
    Discount factor. The per time-step discount factor on future rewards.
    Valid values are greater than 0 upto and including 1. If the discount
    factor is 1, then convergence is cannot be assumed and a warning will
    be displayed. Subclasses of ``MDP`` may pass ``None`` in the case where
    the algorithm does not use a discount factor.
epsilon : float
    Stopping criterion. The maximum change in the value function at each
    iteration is compared against ``epsilon``. Once the change falls below
    this value, then the value function is considered to have converged to
    the optimal value function. Subclasses of ``MDP`` may pass ``None`` in
    the case where the algorithm does not use an epsilon-optimal stopping
    criterion.
max_iter : int
    Maximum number of iterations. The algorithm will be terminated once
    this many iterations have elapsed. This must be greater than 0 if
    specified. Subclasses of ``MDP`` may pass ``None`` in the case where
    the algorithm does not use a maximum number of iterations.

Attributes
----------
P : array
    Transition probability matrices.
R : array
    Reward vectors.
V : tuple
    The optimal value function. Each element is a float corresponding to
    the expected value of being in that state assuming the optimal policy
    is followed.
discount : float
    The discount rate on future rewards.
max_iter : int
    The maximum number of iterations.
policy : tuple
    The optimal policy.
time : float
    The time used to converge to the optimal policy.
verbose : boolean
    Whether verbose output should be displayed or not.

Methods
-------
run
    Implemented in child classes as the main algorithm loop. Raises an
    exception if it has not been overridden.
setSilent
    Turn the verbosity off
setVerbose
    Turn the verbosity on

"""