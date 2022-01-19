
#map0 =  affine_map<(d0)[s0] -> (d0 + s0)>
func @warp_result(%laneid: index, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>,
                  %arg3: memref<1024xf32>, %gid : index) {
  vector_ext.warp_execute_on_lane_0(%laneid) {
    %sa = memref.subview %arg1[%gid] [128] [1] : memref<1024xf32> to memref<128xf32, #map0>
    %sb = memref.subview %arg2[%gid] [128] [1] : memref<1024xf32> to memref<128xf32, #map0>
    %sc = memref.subview %arg3[%gid] [128] [1] : memref<1024xf32> to memref<128xf32, #map0>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %2 = vector.transfer_read %sa[%c0], %cst : memref<128xf32, #map0>, vector<32xf32>
    %3 = vector.transfer_read %sa[%c32], %cst : memref<128xf32, #map0>, vector<32xf32>
    %4 = vector.transfer_read %sb[%c0], %cst : memref<128xf32, #map0>, vector<64xf32>
    %5 = vector.transfer_read %sb[%c32], %cst : memref<128xf32, #map0>, vector<64xf32>
    %6 = arith.addf %2, %3 : vector<32xf32>
    %7 = arith.addf %4, %5 : vector<64xf32>
    vector.transfer_write %6, %sc[%c0] : vector<32xf32>, memref<128xf32, #map0>
    vector.transfer_write %7, %sc[%c32] : vector<64xf32>, memref<128xf32, #map0>
  }
  return
}