// REQUIRES: flagtree-common-ir
// RUN: triton-opt %s --convert-common-ir-to-ttgir | FileCheck %s --implicit-check-not=tile. --implicit-check-not=builtin.unrealized_conversion_cast

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module {
  // CHECK-LABEL: tt.func private @read
  // CHECK-SAME: %[[ARG:.*]]: !ttg.memdesc<16xf32, #shared, #smem, mutable>
  // CHECK: ttg.local_load %[[ARG]]
  tt.func private @read(%buf: !tile.buf<[16], f32, shared>) -> tensor<16xf32> {
    %desc = builtin.unrealized_conversion_cast %buf : !tile.buf<[16], f32, shared> to !ttg.memdesc<16xf32, #shared, #smem, mutable>
    %value = ttg.local_load %desc : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32>
    tt.return %value : tensor<16xf32>
  }

  // CHECK-LABEL: tt.func public @call_and_capture
  // CHECK: %[[BUF:.*]] = ttg.local_alloc
  // CHECK: tt.call @read(%[[BUF]]) : (!ttg.memdesc<16xf32, #shared, #smem, mutable>)
  // CHECK: ttg.warp_specialize(%[[BUF]])
  // CHECK: partition0(%[[PART:.*]]: !ttg.memdesc<16xf32, #shared, #smem, mutable>)
  // CHECK: ttg.local_load %[[PART]]
  tt.func public @call_and_capture() {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    %value = tt.call @read(%buf) : (!tile.buf<[16], f32, shared>) -> tensor<16xf32>
    ttg.warp_specialize(%buf)
    default {
      ttg.warp_yield
    }
    partition0(%arg: !tile.buf<[16], f32, shared>) num_warps(4) {
      %tensor = tile.to_tensor %arg : <[16], f32, shared> -> tensor<16xf32>
      ttg.warp_return
    } : (!tile.buf<[16], f32, shared>) -> ()
    tt.return
  }

  // CHECK-LABEL: tt.func public @alias_view
  // CHECK: %[[BASE:.*]] = ttg.local_alloc
  // CHECK-NOT: ttg.local_alloc
  // CHECK: %[[VIEW:.*]] = tle.memdesc_alias %[[BASE]] {offset_bytes = 16 : i64}
  // CHECK-NOT: ttg.local_alloc
  // CHECK: ttg.local_load %[[VIEW]]
  tt.func public @alias_view() -> tensor<8xi16> {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[32], i8, shared>
    %base = builtin.unrealized_conversion_cast %buf : !tile.buf<[32], i8, shared> to !ttg.memdesc<32xi8, #shared, #smem, mutable>
    %view = tle.memdesc_alias %base {offset_bytes = 16 : i64} : !ttg.memdesc<32xi8, #shared, #smem, mutable> -> !ttg.memdesc<8xi16, #shared, #smem, mutable>
    %alias = builtin.unrealized_conversion_cast %view : !ttg.memdesc<8xi16, #shared, #smem, mutable> to !tile.buf<[8], i16, shared>
    %value = tile.to_tensor %alias : <[8], i16, shared> -> tensor<8xi16>
    tt.return %value : tensor<8xi16>
  }
}
