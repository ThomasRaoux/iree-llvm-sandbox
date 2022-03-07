// RUN: mlir-proto-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute="hoist-uniform" | FileCheck --check-prefixes=CHECK-HOIST %s
// RUN: mlir-proto-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute="hoist-uniform distribute-transfer-write" | FileCheck --check-prefixes=CHECK-D %s
// RUN: mlir-proto-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute="hoist-uniform distribute-transfer-write propagate-distribution" | FileCheck --check-prefixes=CHECK-ALL %s


func @warp_reduction(%in: memref<1024xf32>,%out: memref<1xf32>) {
  %id = gpu.thread_id x
  vector_ext.warp_execute_on_lane_0(%id) {
    %cst = arith.constant dense<3.840000e+02> : vector<1xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %init = vector.transfer_read %out[%c0], %cst_0 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
    %13 = scf.for %arg0 = %c0 to %c1024 step %c32 iter_args(%arg1 = %init) -> (vector<1xf32>) {
      %20 = vector.transfer_read %in[%arg0], %cst_0 {in_bounds = [true]} : memref<1024xf32>, vector<32xf32>
      %21 = vector.reduction <add>, %20 : vector<32xf32> into f32
      %22 = vector.broadcast %21 : f32 to vector<1xf32>
      %23 = arith.addf %22, %arg1 : vector<1xf32>
      scf.yield %23 : vector<1xf32>
    }
    %14 = arith.divf %13, %cst : vector<1xf32>
    vector.transfer_write %14, %out[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32>
  }
  return
}
