//===-- VectorMaskingUtils.h - Utilities for vector masking -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_VECTOREXT_VECTORWARPUTILS_H_
#define DIALECT_VECTOREXT_VECTORWARPUTILS_H_

#include "llvm/ADT/STLExtras.h"

namespace mlir {
class OpBuilder;
class RewritePatternSet;

namespace vector_ext {
// TODO: Move this as op attribute
static const unsigned distributionRatio = 32;

class WarpSingleLaneOp;

// TODO: Add options on how to distribute.
void distributeTransferWrite(OpBuilder &builder, WarpSingleLaneOp op);


void populatePropagateVectorDistributionPatterns(RewritePatternSet &pattern);

} // namespace vector_ext
} // namespace mlir

#endif // DIALECT_VECTOREXT_VECTORWARPUTILS_H_
