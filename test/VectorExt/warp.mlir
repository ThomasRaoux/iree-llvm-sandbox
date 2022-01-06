func @warp_result(%laneid: index, %A: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  vector_ext.warp_single_lane (%laneid) {
    %0 = arith.constant dense<2.0>: vector<32xf32>
    vector.transfer_write %0, %A[%c0]: vector<32xf32>, memref<32xf32>
  }
  return
}