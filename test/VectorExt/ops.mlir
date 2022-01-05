// RUN: mlir-proto-opt %s -split-input-file | FileCheck %s

func @predicate_noresults(%pred: vector<8xi1>) {
  vector_ext.predicate (%pred: vector<8xi1>) {
  }
  return
}

// CHECK-LABEL:   func @predicate_noresults(
// CHECK-NEXT:      vector_ext.predicate(%{{.*}}: vector<8xi1>) {
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// -----

func @predicate_results(%pred: vector<32xi1>) {
  vector_ext.predicate (%pred: vector<32xi1>) -> i32 {
    %c0 = arith.constant 0 : i32
    vector_ext.yield %c0 : i32
  }
  return
}

// CHECK-LABEL:   func @predicate_results(
// CHECK-NEXT:      %{{.*}} = vector_ext.predicate(%{{.*}}: vector<32xi1>) -> (i32) {
// CHECK-NEXT:        %[[CONST:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        vector_ext.yield %[[CONST]] : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// -----

func @warp(%laneid: index) {
  vector_ext.warp_single_lane (%laneid) {
  }
  return
}

// -----

func @warp_result(%laneid: index, %v0 : vector<4xi32>) -> (vector<4xi32>) {
  %2 = vector_ext.warp_single_lane (%laneid) args(%v0 : vector<4xi32>) -> (vector<4xi32>) {
   ^bb0(%arg0 : vector<128xi32>) :
    %0 = arith.constant dense<2>: vector<128xi32>
    %1 = arith.addi %arg0, %0 : vector<128xi32>
    vector_ext.yield %1 : vector<128xi32>
  }
  return %2 : vector<4xi32>
}