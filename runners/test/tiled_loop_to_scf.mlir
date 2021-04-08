// RUN: mlir-proto-opt --linalg-tiled-loop-to-scf %s --split-input-file | FileCheck %s

#map0 = affine_map<(d0) -> (24, -d0 + 192)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 192 + s0 + d1)>
#map2 = affine_map<(d0) -> (16, -d0 + 192)>

func @tiled_loop_to_parallel(%arg0: memref<192x192xf32>,
                             %arg1: memref<192x192xf32>,
                             %arg2: memref<192x192xf32>) {
   %cst = constant 0.000000e+00 : f32
   %c24 = constant 24 : index
   %c16 = constant 16 : index
   %c0 = constant 0 : index
   %c192 = constant 192 : index

  linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%c192, %c192) step (%c24, %c16)
      ins (%arg0, %arg1: memref<192x192xf32>, memref<192x192xf32>)
      outs (%arg2: memref<192x192xf32>) {
    %0 = affine.min #map0(%i)
    %1 = memref.subview %arg0[%i, 0] [%0, 192] [1, 1]
      : memref<192x192xf32> to memref<?x192xf32, #map1>
    %2 = affine.min #map2(%j)
    %3 = memref.subview %arg1[0, %j] [192, %2] [1, 1]
      : memref<192x192xf32> to memref<192x?xf32, #map1>
    %4 = memref.subview %arg2[%i, %j] [%0, %2] [1, 1]
      : memref<192x192xf32> to memref<?x?xf32, #map1>
    linalg.fill(%4, %cst) : memref<?x?xf32, #map1>, f32
    linalg.matmul ins(%1, %3 : memref<?x192xf32, #map1>,
                               memref<192x?xf32, #map1>)
                  outs(%4 : memref<?x?xf32, #map1>)
    linalg.yield
  }
  return
}
// CHECK-LABEL: func @tiled_loop_to_parallel(
// CHECK-SAME: %[[A:.*]]: memref<192x192xf32>, %[[B:.*]]: memref<192x192xf32>,
// CHECK-SAME: %[[C:.*]]: memref<192x192xf32>) {
// CHECK:  %[[C0_F32:.*]] = constant 0.000000e+00 : f32
// CHECK:  %[[C24:.*]] = constant 24 : index
// CHECK:  %[[C16:.*]] = constant 16 : index
// CHECK:  %[[C0:.*]] = constant 0 : index
// CHECK:  %[[C192:.*]] = constant 192 : index
// CHECK:  scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME: to (%[[C192]], %[[C192]]) step (%[[C24]], %[[C16]]) {
// CHECK:    %[[A_sub:.*]] = memref.subview %[[A]]{{\[}}%[[I]]
// CHECK:    %[[B_sub:.*]] = memref.subview %[[B]][0, %[[J]]]
// CHECK:    %[[C_sub:.*]] = memref.subview %[[C]]{{\[}}%[[I]]
// CHECK:    linalg.fill(%[[C_sub]], %[[C0_F32]]) : memref<?x?xf32, #{{.*}}>
// CHECK:    linalg.matmul ins(%[[A_sub]], %[[B_sub]] : {{.*}}) outs(%[[C_sub]]
// CHECK:    scf.yield


// -----

func @tiled_loop_to_for(%arg0: memref<192x192xf32>,
                        %arg1: memref<192x192xf32>,
                        %arg2: memref<f32>) {
   %c24 = constant 24 : index
   %c16 = constant 16 : index
   %c0 = constant 0 : index
   %c192 = constant 192 : index

  linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%c192, %c192) step (%c24, %c16)
      ins (%arg0, %arg1: memref<192x192xf32>, memref<192x192xf32>)
      outs (%arg2: memref<f32>)
      iterators["reduction", "reduction"] {
    linalg.yield
  }
  return
}
// CHECK-LABEL: func @tiled_loop_to_for
// CHECK:  %[[C24:.*]] = constant 24 : index
// CHECK:  %[[C16:.*]] = constant 16 : index
// CHECK:  %[[C0:.*]] = constant 0 : index
// CHECK:  %[[C192:.*]] = constant 192 : index
// CHECK:  cf.for %{{.*}} = %[[C0]] to %[[C192]] step %[[C24]]
// CHECK:    cf.for %{{.*}} = %[[C0]] to %[[C192]] step %[[C16]]