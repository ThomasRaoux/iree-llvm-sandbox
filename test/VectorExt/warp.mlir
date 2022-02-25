// RUN: mlir-proto-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute=propagate-distribution -canonicalize | FileCheck %s
// RUN: mlir-proto-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute=rewrite-warp-ops-to-scf-if -canonicalize | FileCheck %s --check-prefix=CHECK-SCF-IF

#map = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (s0 * 128 + 128)>
#map2 = affine_map<()[s0] -> (s0 * 4 + 128)>

module {
  func @warp_propagate_read(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0:3 = vector_ext.warp_execute_on_lane_0(%arg0) -> 
    (vector<4xf32>, vector<4xf32>, vector<1xf32>) {
      %def = "some_def"() : () -> (vector<32xf32>)
      %r1 = vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
      %r2 = vector.transfer_read %arg2[%c128], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
      %3:2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %r1, %arg5 = %r2) 
      -> (vector<128xf32>, vector<128xf32>) {
        %o1 = affine.apply #map1()[%arg3]
        %o2 = affine.apply #map2()[%arg3]
        %4 = vector.transfer_read %arg1[%o1], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
        %5 = vector.transfer_read %arg1[%o2], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
        %6 = arith.addf %4, %arg4 : vector<128xf32>
        %7 = arith.addf %5, %arg5 : vector<128xf32>
        scf.yield %6, %7 : vector<128xf32>, vector<128xf32>
      }
      vector_ext.yield %3#0, %3#1, %def : vector<128xf32>, vector<128xf32>, vector<32xf32>
    }
    %1 = affine.apply #map()[%arg0]
    vector.transfer_write %0#0, %arg2[%1] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
    %2 = affine.apply #map2()[%arg0]
    vector.transfer_write %0#1, %arg2[%2] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
    "some_use"(%0#2) : (vector<1xf32>) -> ()
    return
  }
}