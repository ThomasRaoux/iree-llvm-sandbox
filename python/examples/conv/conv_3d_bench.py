# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

import typing as tp

fun_name = 'conv_3d_ndhwc_dhwcf_main'
op_name = 'linalg.conv_3d_ndhwc_dhwcf'

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    e.print_ir(after_all=False) for e in [
        SingleTilingExpert(
            fun_name=fun_name,
            op_name=op_name,
            #           N  D, H  W  C   KH KW F
            tile_sizes=[1, 2, 1, 8, 32, 1, 1, 8])
    ]
]

################################################################################
### Problem instantiation
################################################################################

keys = ['N', 'D', 'H', 'W', 'C', 'KD', 'KH', 'KW', 'F', 'strides', 'dilations']


def make_size_list(keys: tp.Sequence[str], sizes: tp.Sequence):
  return {k: v for k, v in zip(keys, sizes)}


# CHECK-NOT: FAILURE
def main():
  n_iters = 1

  # Specify default configuration and parse command line.
  args = test_argparser(
    "conv 3d benchmark",
    #  N  D   H   W   C  KD  KH  KW   F     st      dil
    default_problem_sizes_list = [
      [8, 4, 16, 16, 32,  3,  3,  3, 64, [1, 1, 1], [1, 1, 1]],
      [8, 4, 16, 16, 32,  3,  3,  3, 64, [1, 1, 2], [1, 1, 2]],
      [8, 4, 16, 16, 32,  3,  3,  3, 64, [1, 2, 1], [1, 1, 2]],
      [8, 4, 16, 16, 32,  3,  3,  3, 64, [1, 2, 2], [1, 2, 2]],
      [8, 4, 16, 16, 32,  3,  3,  3, 64, [1, 2, 3], [3, 2, 2]],
      [8, 4, 16, 16, 32,  3,  3,  3, 64, [3, 2, 2], [1, 3, 2]],
    ],
    default_expert_list = [
      idx for idx, _ in enumerate(all_experts)
    ],
    default_runtime_only_list = [],
    default_spec_list = [])

  test_harness(lambda sizes, types: ConvolutionProblem(
      'NDHWC', 'DHWCF', strides=sizes['strides'], dilations=sizes['dilations']),
               [[np.float32] * 3],
               [make_size_list(keys, sizes) for sizes in args.problem_sizes_list],
               [all_experts[idx] for idx in args.expert_list if idx < len(all_experts)],
               n_iters=n_iters,
               function_name=fun_name)


if __name__ == '__main__':
  main()
