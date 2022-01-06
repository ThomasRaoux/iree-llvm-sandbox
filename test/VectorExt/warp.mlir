func @warp_result(%laneid: index, %A: memref<32xf32>, %B: memref<32xf32>, 
                  %C: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  vector_ext.warp_single_lane (%laneid) {
    %0 = vector.transfer_read %A[%c0], %f0 : memref<32xf32>, vector<32xf32>
    %1 = vector.transfer_read %B[%c0], %f0 : memref<32xf32>, vector<32xf32>
    %2 = arith.addf %0, %1 : vector<32xf32>
    vector.transfer_write %2, %C[%c0]: vector<32xf32>, memref<32xf32>
  }
  return
}