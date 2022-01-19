//===- TestVectorWarp.cpp - Test vector warp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/VectorExt/VectorExtOps.h"
#include "Dialects/VectorExt/VectorExtWarpUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::vector_ext;

namespace {

struct TestVectorWarp : public PassWrapper<TestVectorWarp, FunctionPass> {

  TestVectorWarp() = default;
  TestVectorWarp(const TestVectorWarp &pass) {}

  StringRef getArgument() const final { return "test-vector-warp-distribute"; }
  StringRef getDescription() const final {
    return "Test vector warp transformations";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect, VectorExtDialect>();
  }

  Option<bool> distributeTransferWriteOps{
      *this, "distribute-transfer-write",
      llvm::cl::desc("Test distribution of transfer write"),
      llvm::cl::init(false)};

  Option<bool> hoistUniform{
      *this, "hoist-uniform",
      llvm::cl::desc("Test hoist uniform"),
      llvm::cl::init(false)};

  Option<bool> propagateDistribution{
      *this, "propagate-distribution",
      llvm::cl::desc("Test distribution propgation"), llvm::cl::init(false)};

  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    funcOp.walk([&](Operation * op) {
      if (auto warpOp = dyn_cast<WarpSingleLaneOp>(op)) {
        if (hoistUniform) {
          moveScalarUniformCode(warpOp);
        }
        if (distributeTransferWriteOps) {
          OpBuilder builder(op->getContext());
          distributeTransferWrite(builder, warpOp);
        }
        WalkResult::interrupt();
      }
    });
    MLIRContext *ctx = &getContext();
    if (propagateDistribution) {
      RewritePatternSet patterns(ctx);
      vector_ext::populatePropagateVectorDistributionPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }
  }
};

} // namespace

namespace mlir {
namespace test_ext {
void registerTestVectorWarps() { PassRegistration<TestVectorWarp>(); }
} // namespace test_ext
} // namespace mlir
