import numpy as np
from itertools import combinations

# Generates all bitmasks of length 'n' with 'k' bits set to 1.
def _n_k_bitmasks(n, k):
    all_combinations = list(combinations(range(n), k))
    k_bitmasks = np.zeros((len(all_combinations), n), dtype=int)
    for i, positions in enumerate(all_combinations):
        k_bitmasks[i, list(positions)] = 1
    return k_bitmasks

# Calculates the binomial coefficient (n choose k) using factorials.
def _n_k_coefficient(n, k):
    return _factorial(k) * _factorial(n - k - 1) / _factorial(n)

# Generates all possible bitmasks of a given length.
def _generate_bitmasks(length):
    if length == 0:
      return [[]]
    num_bitmasks = 2 ** length
    bitmasks = []
    for i in range(num_bitmasks)[::-1]:
        bitmask = bin(i)[2:].zfill(length)
        bitmask = [int(i) for i in bitmask]
        bitmasks.append(bitmask)
    return bitmasks

# Inserts an element 'x' at position 'idx' in list 'l'.
def _insert(l, idx, x):
    assert idx <= len(l) and idx >= 0
    _l = l.copy()
    _l.insert(idx, x)
    return _l

# Inserts an element 'x' at position 'idx' in a NumPy array 'arr'.
def _insert_np(arr, idx, x):
    assert 0 <= idx <= arr.size  # Modified to use the size attribute of NumPy arrays
    return np.concatenate((arr[:idx], np.array([x]), arr[idx:]))

# Masks elements in 'sequence' according to 'bitmask', optionally collapsing the result.
def _mask_input(sequence, bitmask, mask_token='[PAD]', collapse=True):
    assert len(sequence) == len(bitmask)
    if collapse:
        res = [sequence[i] for i in range(len(sequence)) if bitmask[i] == 1]
        padding = [mask_token for i in range(len(sequence) - len(res))]
        return padding + res
    else:
        return [sequence[i] if bitmask[i] == 1 else mask_token for i in range(len(sequence))]

# Recursively calculates the factorial of 'n'.
def _factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * _factorial(n - 1)

# Calculates the number of k-combinations from a set of n elements.
def _combinations(n, k):
    return _factorial(n) // (_factorial(k) * _factorial(n - k))
