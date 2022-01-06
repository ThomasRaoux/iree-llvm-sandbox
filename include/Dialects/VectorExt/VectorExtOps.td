//===-- VectorExtOps.td - Vector Extension dialect ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_VECTOREXT_VECTOREXTOPS
#define DIALECTS_VECTOREXT_VECTOREXTOPS

include "Dialects/VectorExt/VectorExtBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VectorExt_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<VectorExt_Dialect, mnemonic, traits> {
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def VectorExt_PredicateOp : VectorExt_Op<"predicate",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"vector_ext::YieldOp">, RecursiveSideEffects,
       NoRegionArguments]> {
  let summary = "vector.predicate operation";
  let description = [{ TODO }];

  let arguments = (ins VectorOf<[I1]>:$predicate);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$truePredicateRegion);

// TODO(diegocaballero): Add falsePredicateRegion.

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$predicate)>,
    //OpBuilder<(ins "TypeRange":$resultTypes, "Value":$predicate)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$predicate,
     CArg<"function_ref<void(OpBuilder &, Location)>",
          "buildTerminatedBody">:$truePredicateBuilder)>
  ];
}

def VectorExt_YieldOp : VectorExt_Op<"yield", [
    NoSideEffect, ReturnLike, Terminator]> {
  let summary = "VectorExt yield op";
  let description = [{
    `vector_ext.yield` is a special terminator operation for blocks inside
    regions in `vector_ext` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def VectorExt_WarpSingleLaneOp : VectorExt_Op<"warp_single_lane",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"vector_ext::YieldOp">,
       RecursiveSideEffects]> {
  let summary = "vector.warp_single_lane operation";
  let description = [{
    Operation that encapsulate code that should be ran on a single lane within a
    warp.

    This may be lowered to an scf.if region as below:
    ```
      %cnd = arith.cmpi eq, %threadid, %c0 : index
      scf.if %cnd {

      }
    ```
    Operands are vector values distributed on all threads of the warp. The
    matching region argument is a vector of all the values of those threads
    available to the single active lane.

    Return values are distributed on all threads using laneId as index. The
    vector is distributed based on the shape ratio between the vector type of
    the yield and the result type.
    If the shapes are the same this means the value is broadcasted to all lanes.
    In the future the distribution can be made more explicit using affine_maps.
  }];

  let arguments = (ins Index:$laneid, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$warpRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$laneid)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$laneid)>
  ];
}

#endif // DIALECTS_VECTOREXT_VECTOREXTOPS
