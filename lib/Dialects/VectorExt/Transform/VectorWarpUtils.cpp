//===- VectorWarpUtils.cpp - Utilities vector warp ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/VectorExt/VectorExtOps.h"
#include "Dialects/VectorExt/VectorExtWarpUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::vector_ext;

// Clones `op` into a new operations that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(OpBuilder &builder, Location loc,
                                              Operation *op,
                                              ArrayRef<Value> operands,
                                              ArrayRef<Type> resultTypes) {
  OperationState res(loc, op->getName().getStringRef(), operands, resultTypes,
                     op->getAttrs());
  return builder.createOperation(res);
}

static WarpSingleLaneOp moveRegionToNewWarp(OpBuilder &b,
                                           WarpSingleLaneOp warpOp,
                                           ValueRange newYieldedValues,
                                           TypeRange newReturnTypes,
                                           bool replaceLoopResults) {
  // Create a new op before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(warpOp);
  SmallVector<Type> types(warpOp.getResultTypes().begin(),
                          warpOp.getResultTypes().end());
  types.append(newReturnTypes.begin(), newReturnTypes.end());
  // auto operands = llvm::to_vector<4>(op.args());
  auto newWarpOp =
      b.create<WarpSingleLaneOp>(warpOp.getLoc(), types, warpOp.laneid());

  Region &opBody = warpOp.getBodyRegion();
  Region& newOpBody = newWarpOp.getBodyRegion();
  newOpBody.takeBody(opBody);
  auto yield =
      cast<vector_ext::YieldOp>(newOpBody.getBlocks().begin()->getTerminator());
  yield.operandsMutable().append(newYieldedValues);

  // Replace results if requested.
  if (replaceLoopResults) {
    for (auto it :
         llvm::zip(warpOp.getResults(),
                   newWarpOp.getResults().take_front(warpOp.getNumResults())))
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }
  return newWarpOp;
}

void mlir::vector_ext::distributeTransferWrite(OpBuilder &builder,
                                               WarpSingleLaneOp op) {
  vector::TransferWriteOp writeOp;
  for (Operation &op : llvm::reverse(op.getBody()->getOperations())) {
    writeOp = dyn_cast<vector::TransferWriteOp>(op);
    if (writeOp)
      break;
    // If we find an op with side effect before finding a transfer_write we
    // cannot hoist out the transfer write.
    if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!memInterface.hasNoEffect())
        return;
      if (op.hasTrait<OpTrait::HasRecursiveSideEffects>())
        return;
    }
    if (op.hasTrait<OpTrait::HasRecursiveSideEffects>())
      return;
  }
  if(!writeOp)
    return;
  // TODO: Add a callback to determine the target shape. For now we just
  // distribute along the most inner dimension.
  SmallVector<int64_t> targetShape(writeOp.getVectorType().getShape().begin(),
                                   writeOp.getVectorType().getShape().end());
  targetShape.back() = targetShape.back() / distributionRatio;
  VectorType targeType =
      VectorType::get(targetShape, writeOp.getVectorType().getElementType());
  SmallVector<Value> yieldValues = { writeOp.vector() };
  SmallVector<Type> retTypes = { targeType };
  WarpSingleLaneOp newWarpOp =
      moveRegionToNewWarp(builder, op, yieldValues, retTypes, true);
  writeOp->moveAfter(newWarpOp);
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(writeOp);
  
  // Create a map (d0, d1) -> (d1). This map should come as a user option on how
  // to distirbute the transfer_write and can be propagated to the WarpOp.
  int64_t vecRank = writeOp.getVectorType().getRank();
  auto map = AffineMap::get(vecRank, 0, builder.getAffineDimExpr(vecRank - 1));

  SmallVector<Value, 4> indices(writeOp.indices().begin(),
                                writeOp.indices().end());
  AffineMap indexMap = map.compose(writeOp.permutation_map());
  unsigned idCount = 0;
  Location loc = writeOp.getLoc();
  for (auto it : llvm::zip(indexMap.getResults(), map.getResults())) {
    AffineExpr d0, d1;
    bindDims(op.getContext(), d0, d1);
    auto indexExpr = std::get<0>(it).dyn_cast<AffineDimExpr>();
    if (!indexExpr)
      continue;
    unsigned indexPos = indexExpr.getPosition();
    unsigned vectorPos = std::get<1>(it).cast<AffineDimExpr>().getPosition();
    auto scale =
        getAffineConstantExpr(targeType.getDimSize(vectorPos), op.getContext());
    assert(idCount == 0 && "multiple Ids not supported yet.");
    indices[indexPos] =
        makeComposedAffineApply(builder, loc, d0 + scale * d1,
                                {indices[indexPos], op.laneid()});
  }

  writeOp.vectorMutable().assign(newWarpOp.getResults().back());
  writeOp.indicesMutable().assign(indices);
  op->erase();
}

struct WarpOpElementwise : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<vector_ext::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    llvm::Optional<unsigned> index;
    for (OpOperand& yieldOperand : yield->getOpOperands()) {
      Value yieldValue = yieldOperand.get();
      Operation *definedOp = yieldValue.getDefiningOp();
      if (definedOp &&
          OpTrait::hasElementwiseMappableTraits(definedOp)) {
        // TODO: The value may have several use in the YieldOp. We should handle
        // it.
        if (definedOp->hasOneUse() &&
            !warpOp.getResult(yieldOperand.getOperandNumber()).use_empty()) {
          index = yieldOperand.getOperandNumber();
          break;
        }
      }
    }
    if(!index)
      return failure();

    Operation *elementWise = yield.getOperand(*index).getDefiningOp();
    Value distributedVal = warpOp.getResult(*index);
    SmallVector<Value> yieldValues;
    SmallVector<Type> retTypes;
    SmallVector<unsigned> operandForwarded;
    for (OpOperand &operand : elementWise->getOpOperands()) {
      if(!warpOp.getBodyRegion().isAncestor(operand.get().getParentRegion()))
        continue;
      auto targetType = VectorType::get(
          distributedVal.getType().cast<VectorType>().getShape(),
          operand.get().getType().cast<VectorType>().getElementType());
      operandForwarded.push_back(operand.getOperandNumber());
      yieldValues.push_back(operand.get());
      retTypes.push_back(targetType);
    }
    WarpSingleLaneOp newWarpOp =
        moveRegionToNewWarp(rewriter, warpOp, yieldValues, retTypes, false);
    SmallVector<Value> newOperands(elementWise->getOperands().begin(),
                                   elementWise->getOperands().end());
    for (unsigned i : operandForwarded) {
      newOperands[i] = newWarpOp.getResult(i + warpOp.getNumResults());
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newWarpOp);
    Operation *newOp = cloneOpWithOperandsAndTypes(
        rewriter, warpOp.getLoc(), elementWise, newOperands,
        {warpOp.getResult(*index).getType()});
    SmallVector<Value> results(newWarpOp.getResults().begin(),
                               newWarpOp.getResults().end());
    results[*index] = newOp->getResult(0);
    results.resize(warpOp.getNumResults());
    rewriter.replaceOp(warpOp, results);
    return success();
  }
};

void mlir::vector_ext::populatePropagateVectorDistributionPatterns(
    RewritePatternSet &pattern) {
  pattern.add<WarpOpElementwise>(pattern.getContext());
}