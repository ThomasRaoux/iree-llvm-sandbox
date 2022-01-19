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

/// Helper to create a new WarpSingleLaneOp regions with different signature.
static WarpSingleLaneOp
moveRegionToNewWarpSignature(OpBuilder &b, WarpSingleLaneOp warpOp,
                             ValueRange newYieldedValues,
                             TypeRange newReturnTypes) {
  // Create a new op before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(warpOp);
  auto newWarpOp = b.create<WarpSingleLaneOp>(warpOp.getLoc(), newReturnTypes,
                                              warpOp.laneid());

  Region &opBody = warpOp.getBodyRegion();
  Region &newOpBody = newWarpOp.getBodyRegion();
  newOpBody.takeBody(opBody);
  auto yield =
      cast<vector_ext::YieldOp>(newOpBody.getBlocks().begin()->getTerminator());
  yield.operandsMutable().assign(newYieldedValues);
  return newWarpOp;
}

/// Helper to create a new WarpSingleLaneOp region with extra outputs.
static WarpSingleLaneOp
moveRegionToNewWarpAddReturn(OpBuilder &b, WarpSingleLaneOp warpOp,
                             ValueRange newYieldedValues,
                             TypeRange newReturnTypes) {

  SmallVector<Type> types(warpOp.getResultTypes().begin(),
                          warpOp.getResultTypes().end());
  types.append(newReturnTypes.begin(), newReturnTypes.end());
  auto yield = cast<vector_ext::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  SmallVector<Value> yieldValue(yield.getOperands().begin(),
                                yield.getOperands().end());
  yieldValue.append(newYieldedValues.begin(), newYieldedValues.end());
  WarpSingleLaneOp newWarpOp =
      moveRegionToNewWarpSignature(b, warpOp, yieldValue, types);
  for (auto it :
       llvm::zip(warpOp.getResults(),
                 newWarpOp.getResults().take_front(warpOp.getNumResults())))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  return newWarpOp;
}

llvm::Optional<std::pair<Operation *, unsigned>>
getWarpResult(WarpSingleLaneOp warpOp, std::function<bool(Operation *)> fn) {
  auto yield = cast<vector_ext::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValue = yieldOperand.get();
    Operation *definedOp = yieldValue.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty())
        return std::make_pair(definedOp, yieldOperand.getOperandNumber());
    }
  }
  return {};
}

// Currently the distribution map is implicit based on the vector shape. In the
// future it will be part of the op.
static AffineMap calculateImplicitMap(Value yield, Value ret) {
  auto srcType = yield.getType().cast<VectorType>();
  auto dstType = ret.getType().cast<VectorType>();
  SmallVector<AffineExpr, 4> perm;
  // Check which dimension have a multiplicity greater than 1 and associated
  // them to the IDs in order.
  for (unsigned i = 0, e = srcType.getRank(); i < e; i++) {
    if (srcType.getDimSize(i) != dstType.getDimSize(i))
      perm.push_back(getAffineDimExpr(i, yield.getContext()));
  }
  auto map = AffineMap::get(srcType.getRank(), 0, perm, yield.getContext());
  return map;
}

/// Sink out elementwise op feeding into a warp op yield.
/// ```
/// %0 = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
///   %3 = arith.addf %1, %2 : vector<32xf32>
///   vector_ext.yield %3 : vector<32xf32>
/// }
/// ```
/// To
/// ```
/// %r:3 = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>,
/// vector<1xf32>, vector<1xf32>) {
///   ...
///   %4 = arith.addf %2, %3 : vector<32xf32>
///   vector_ext.yield %4, %2, %3 : vector<32xf32>, vector<32xf32>,
///   vector<32xf32>
/// }
/// %0 = arith.addf %r#1, %r#2 : vector<1xf32>
struct WarpOpElementwise : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    auto yieldOperand = getWarpResult(warpOp, [](Operation *op) {
      return OpTrait::hasElementwiseMappableTraits(op);
    });
    if (!yieldOperand)
      return failure();
    Operation *elementWise = yieldOperand->first;
    Value distributedVal = warpOp.getResult(yieldOperand->second);
    SmallVector<Value> yieldValues;
    SmallVector<Type> retTypes;
    SmallVector<unsigned> operandForwarded;
    for (OpOperand &operand : elementWise->getOpOperands()) {
      auto targetType = VectorType::get(
          distributedVal.getType().cast<VectorType>().getShape(),
          operand.get().getType().cast<VectorType>().getElementType());
      operandForwarded.push_back(operand.getOperandNumber());
      retTypes.push_back(targetType);
      yieldValues.push_back(operand.get());
    }
    WarpSingleLaneOp newWarpOp =
        moveRegionToNewWarpAddReturn(rewriter, warpOp, yieldValues, retTypes);
    SmallVector<Value> newOperands(elementWise->getOperands().begin(),
                                   elementWise->getOperands().end());
    for (unsigned i : operandForwarded) {
      newOperands[i] = newWarpOp.getResult(i + warpOp.getNumResults());
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newWarpOp);
    Operation *newOp = cloneOpWithOperandsAndTypes(
        rewriter, warpOp.getLoc(), elementWise, newOperands,
        {warpOp.getResult(yieldOperand->second).getType()});
    newWarpOp.getResult(yieldOperand->second)
        .replaceAllUsesWith(newOp->getResult(0));
    rewriter.eraseOp(warpOp);
    return success();
  }
};

/// Sink out transfer_read op feeding into a warp op yield.
/// ```
/// %0 = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
//    %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>,
//    vector<32xf32>
///   vector_ext.yield %2 : vector<32xf32>
/// }
/// ```
/// To
/// ```
/// %dead = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>,
/// vector<1xf32>, vector<1xf32>) {
///   ...
///   %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>,
///   vector<32xf32> vector_ext.yield %2 : vector<32xf32>
/// }
/// %0 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>, vector<1xf32>
struct WarpOpTransferRead : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    auto operand = getWarpResult(
        warpOp, [](Operation *op) { return isa<vector::TransferReadOp>(op); });
    if (!operand)
      return failure();
    auto read = cast<vector::TransferReadOp>(operand->first);
    if (!llvm::all_of(read->getOperands(), [&](Value value) {
          return warpOp.isDefinedOutsideOfRegion(value);
        }))
      return failure();
    Value distributedVal = warpOp.getResult(operand->second);

    SmallVector<Value, 4> indices(read.indices().begin(), read.indices().end());
    AffineMap map = calculateImplicitMap(read.getResult(), distributedVal);
    AffineMap indexMap = map.compose(read.permutation_map());
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(warpOp);
    for (auto it : llvm::zip(indexMap.getResults(), map.getResults())) {
      AffineExpr d0, d1;
      bindDims(read.getContext(), d0, d1);
      auto indexExpr = std::get<0>(it).dyn_cast<AffineDimExpr>();
      if (!indexExpr)
        continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = std::get<1>(it).cast<AffineDimExpr>().getPosition();
      auto scale = getAffineConstantExpr(
          distributedVal.getType().cast<VectorType>().getDimSize(vectorPos),
          read.getContext());
      indices[indexPos] =
          makeComposedAffineApply(rewriter, read.getLoc(), d0 + scale * d1,
                                  {indices[indexPos], warpOp.laneid()});
    }
    Value newRead = rewriter.create<vector::TransferReadOp>(
        read.getLoc(), distributedVal.getType(), read.source(), indices,
        read.permutation_mapAttr(), read.padding(), read.mask(),
        read.in_boundsAttr());
    distributedVal.replaceAllUsesWith(newRead);
    return success();
  }
};

/// Remove any result that has no use along with the matching yieldOp operand.
// TODO: Move this in WarpSingleLaneOp canonicalization.
struct WarpOpDeadResult : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    SmallVector<Value> yieldValue;
    bool hasDeadResults = false;
    auto yield = cast<vector_ext::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    for (OpResult result : warpOp.getResults()) {
      if (result.use_empty()) {
        hasDeadResults = true;
        continue;
      }
      resultTypes.push_back(result.getType());
      yieldValue.push_back(yield.getOperand(result.getResultNumber()));
    }
    if (!hasDeadResults)
      return failure();
    WarpSingleLaneOp newWarpOp =
        moveRegionToNewWarpSignature(rewriter, warpOp, yieldValue, resultTypes);
    unsigned resultIndex = 0;
    for (OpResult result : warpOp.getResults()) {
      if (result.use_empty())
        continue;
      result.replaceAllUsesWith(newWarpOp.getResult(resultIndex++));
    }
    rewriter.eraseOp(warpOp);
    return success();
  }
};

/// Helper to figure out if an op has side effects or recursive side-effects.
static bool hasSideEffect(Operation &op) {
  // If we find an op with side effect before finding a transfer_write we
  // cannot hoist out the transfer write.
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (!memInterface.hasNoEffect())
      return true;
    if (op.hasTrait<OpTrait::HasRecursiveSideEffects>())
      return true;
  }
  if (op.hasTrait<OpTrait::HasRecursiveSideEffects>())
    return true;
  return false;
}

static AffineMap getDistributionMap(vector::TransferWriteOp writeOp) {
  // Create a map (d0, d1) -> (d1). This map should come as a user option on
  // how to distirbute the transfer_write and can be propagated to the WarpOp.
  int64_t vecRank = writeOp.getVectorType().getRank();
  OpBuilder builder(writeOp.getContext());
  auto map = AffineMap::get(vecRank, 0, builder.getAffineDimExpr(vecRank - 1));
  return map;
}

// TODO: Move to the op.
static unsigned distributionRatio = 32;

void mlir::vector_ext::distributeTransferWrite(OpBuilder &builder,
                                               WarpSingleLaneOp op) {

  vector::TransferWriteOp writeOp;
  while (1) {
    // Find the first transfer_write from the end of the block.
    for (Operation &elementOp : llvm::reverse(op.getBody()->getOperations())) {
      writeOp = dyn_cast<vector::TransferWriteOp>(elementOp);
      if (writeOp)
        break;
      if (hasSideEffect(elementOp))
        return;
    }
    if (!writeOp)
      return;
    if (!llvm::all_of(writeOp->getOperands(), [&](Value value) {
          return writeOp.vector() == value ||
                 op.isDefinedOutsideOfRegion(value);
        }))
      return;
    AffineMap map = getDistributionMap(writeOp);
    SmallVector<int64_t> targetShape(writeOp.getVectorType().getShape().begin(),
                                     writeOp.getVectorType().getShape().end());
    assert(map.getNumResults() == 1 && "implement multi-dim distribution");
    for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
      unsigned position = map.getDimPosition(i);
      targetShape[position] = targetShape[position] / distributionRatio;
    }
    VectorType targeType =
        VectorType::get(targetShape, writeOp.getVectorType().getElementType());
    SmallVector<Value> yieldValues = {writeOp.vector()};
    SmallVector<Type> retTypes = {targeType};
    WarpSingleLaneOp newWarpOp =
        moveRegionToNewWarpAddReturn(builder, op, yieldValues, retTypes);
    writeOp->moveAfter(newWarpOp);
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(writeOp);

    AffineMap indexMap = map.compose(writeOp.permutation_map());
    Location loc = writeOp.getLoc();
    SmallVector<Value> indices(writeOp.indices().begin(),
                               writeOp.indices().end());
    for (auto it : llvm::zip(indexMap.getResults(), map.getResults())) {
      AffineExpr d0, d1;
      bindDims(op.getContext(), d0, d1);
      auto indexExpr = std::get<0>(it).dyn_cast<AffineDimExpr>();
      if (!indexExpr)
        continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = std::get<1>(it).cast<AffineDimExpr>().getPosition();
      auto scale =
          getAffineConstantExpr(targetShape[vectorPos], op.getContext());
      indices[indexPos] = makeComposedAffineApply(
          builder, loc, d0 + scale * d1, {indices[indexPos], op.laneid()});
    }
    writeOp.vectorMutable().assign(newWarpOp.getResults().back());
    writeOp.indicesMutable().assign(indices);
    op->erase();
    op = newWarpOp;
  }
}

void mlir::vector_ext::populatePropagateVectorDistributionPatterns(
    RewritePatternSet &pattern) {
  pattern.add<WarpOpElementwise, WarpOpTransferRead, WarpOpDeadResult>(
      pattern.getContext());
}

static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  if (!llvm::all_of(op->getOperands(), definedOutside))
    return false;
  if(hasSideEffect(*op))
    return false;
  if(op->getNumRegions() != 0)
    return false;
  return true;
}

void mlir::vector_ext::moveScalarUniformCode(WarpSingleLaneOp warpOp) {
  Block* body = warpOp.getBody();

  // Keep track of the ops we want to hoist.
  llvm::SmallSetVector<Operation*, 8> opsToMove;

  // Helper to check if a value is or will be defined outside of the region.
  auto isDefinedOutsideOfBody = [&](Value value) {
    auto *definingOp = value.getDefiningOp();
    return (definingOp && opsToMove.count(definingOp)) ||
           warpOp.isDefinedOutsideOfRegion(value);
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there.
  for (auto &op : body->without_terminator()) {
    bool hasVectorResult = llvm::any_of(op.getResults(), [](Value result) {
      return result.getType().isa<VectorType>();
    });
    if (!hasVectorResult && canBeHoisted(&op, isDefinedOutsideOfBody))
      opsToMove.insert(&op);
  }

  // Move all the ops marked as uniform outside of the region.
  for(Operation* op : opsToMove)
    op->moveBefore(warpOp);
}

