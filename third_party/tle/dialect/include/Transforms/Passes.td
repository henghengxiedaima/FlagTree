/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// flagtree tle
#ifndef TRITON_TLE_PASSES
#define TRITON_TLE_PASSES

include "mlir/Pass/PassBase.td"

def TritonTleFuseNodeRemoteTransfers
    : Pass<"triton-tle-fuse-node-remote-transfers", "mlir::ModuleOp"> {
  let summary = "fuse node remote pointer copies into TLE transfers";

  let description = [{
    At the TTIR-to-TTGIR boundary, matches direct one-dimensional contiguous
    copies between an entry-function global pointer argument and a
    pointer-valued node ``tle.remote_pointers`` marker. Tensor accesses must
    reuse the same ``tt.make_range(0, N)`` and may be unmasked or use the same
    contiguous prefix mask on both sides. Prefix lengths must satisfy
    ``1 <= valid_n <= N``; constant violations are rejected during compilation
    and dynamic violations trigger a device assertion. Without a range, the copy
    transfers one element; scalar masks are rejected. The
    load/store/address chain
    is replaced with the existing result-less node transfer form, and local
    pointer argument ordinals are recorded as module metadata for host-side
    registered-buffer validation.
    After fusion and dead-address cleanup, all surviving ranked tensors,
    including those in functions without node transfers, are limited to 2^20
    elements. Larger node transfer descriptions (up to the IR construction
    limit of 2^25) are legal only if they are eliminated before this check.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleEarlyAssignMemorySpace
    : Pass<"triton-tle-early-assign-memory-space", "mlir::ModuleOp"> {
  let summary = "early assign memory space";

  let description = [{
    This pass assigns memory spaces to tensor pointers according to `tt.memory_space` attribute on tensors.
    This is done early to enable subsequent passes to make better optimization decisions based on memory spaces.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleSelectEncodings
    : Pass<"triton-tle-select-encodings", "mlir::ModuleOp"> {
  let summary = "select shared encodings for local pointer and related users";

  let description = [{
    This pass performs shared-memory pointer canonicalization and encoding
    selection for `tle.local_pointers` and dependent users so downstream
    conversion sees stable, cost-aware layouts.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleInsertLocalPointerBarriers
    : Pass<"triton-tle-insert-local-pointer-barriers", "mlir::ModuleOp"> {
  let summary =
      "insert CTA barriers between tle.local_pointers stores and loads";

  let description = [{
    This pass ensures data staged through ``tle.local_pointers`` backed shared
    memory buffers is globally visible before the program re-reads the same
    buffer. It looks for sequences of ``tl.store`` operations targeting
    ``tle.local_pointers`` tensors followed by ``tl.load`` reads and inserts a
    ``gpu.barrier`` between the two when needed.
  }];

  let dependentDialects = ["mlir::gpu::GPUDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleOptimizeLocalPointerLoads
    : Pass<"triton-tle-optimize-local-pointer-loads", "mlir::ModuleOp"> {
  let summary = "rewrite full-view tle.local_pointers loads into ttg.local_load";

  let description = [{
    This pass matches ``tl.load`` operations whose pointer operand is produced
    by ``tle.local_pointers`` with full-view indexing (including zero-indices
    form) and rewrites them into ``ttg.local_load`` directly from the backing
    memdesc. The transform is conservative and only fires when the load has no
    mask/other operands.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleOptimizeLocalPointerStores
    : Pass<"triton-tle-optimize-local-pointer-stores", "mlir::ModuleOp"> {
  let summary = "rewrite tle.local_pointers stores into ttg.local_store";

  let description = [{
    This pass rewrites ``tl.store`` operations that target pointers produced by
    ``tle.local_pointers`` into ``ttg.local_store`` to avoid pointer-based
    shared-memory stores in the backend.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleOptimizeLocalPointerAsyncStores
    : Pass<"triton-tle-optimize-local-pointer-async-stores", "mlir::ModuleOp"> {
  let summary =
      "rewrite gmem load -> tle.local_pointers subview store into async_copy";

  let description = [{
    This pass matches conservative staging patterns of the form:

      %v = tt.load %gptr ...
      tt.store %local_ptr_subview, %v ...

    where ``%local_ptr_subview`` is produced by ``tle.local_pointers`` and
    denotes a static full subview of a backing shared memdesc. The pass
    rewrites the pair into:

      %sub = ttg.memdesc_subslice %base[...]
      %tok = ttg.async_copy_global_to_local %gptr, %sub ...
      %grp = ttg.async_commit_group %tok
      ttg.async_wait %grp {num = 0}

    The transform is conservative and only fires when the store value is
    produced by a single-use global load and the subview offsets are static.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTlePromoteLocalStoreStaging
    : Pass<"triton-tle-promote-local-store-staging", "mlir::ModuleOp"> {
  let summary =
      "promote transient TLE local_store staging to pipelineable local_alloc";

  let description = [{
    This pass rewrites safe loop-local staging patterns of the form:

      %v = tt.load ...
      ttg.local_store %v, %smem
      ... users of %smem ...

    into:

      %stage = ttg.local_alloc %v
      ... users of %stage ...

    for buffers that are overwritten every loop iteration and only read after
    the store. The rewrite exposes the standard TritonGPU load -> local_alloc
    pattern so the existing software pipeliner can expand it into double
    buffered async copies when the loop has ``tt.num_stages > 1``.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleTileStylePipelineSchedule
    : Pass<"triton-tle-tile-style-pipeline-schedule", "mlir::ModuleOp"> {
  let summary =
      "rewrite eligible Triton schedules into a TileLang-style preload-2 pipeline";

  let description = [{
    This pass runs after Triton loop scheduling and before software pipeline
    expansion. For carefully selected loops, it rewrites the scheduled stages
    from a standard ``stage0(load) + stage1(consume)`` layout into a
    TileLang-style layout with an empty lead stage:

      stage0(load) + stage2(consume)

    together with ``tt.scheduled_max_stage = 2`` and a TLE marker attribute.
    The resulting pipeline shape creates a two-iteration prologue while keeping
    the original Triton software pipeliner as the fallback path when the loop
    is not eligible.
  }];

  let dependentDialects = ["mlir::scf::SCFDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleDowngradeInvalidAsyncCopy
    : Pass<"triton-tle-downgrade-invalid-async-copy", "mlir::ModuleOp"> {
  let summary =
      "downgrade TLE async copies that cannot be lowered to cp.async";

  let description = [{
    This pass is TLE-owned because it repairs async copies created or exposed
    by TLE staging/pipeline transforms. For ``ttg.async_copy_global_to_local``
    operations whose final vector width is not a legal cp.async transaction
    size, it replaces the async copy and its simple token chain with a
    synchronous ``tt.load`` plus ``ttg.local_store`` before software
    pipelining.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleOptimizeExclusiveCumsumLayouts
    : Pass<"triton-tle-optimize-exclusive-cumsum-layouts", "mlir::ModuleOp"> {
  let summary = "fold convert_layout around tle.exclusive_cumsum";

  let description = [{
    This pass folds patterns of the form:
      ttg.convert_layout -> tle.exclusive_cumsum -> ttg.convert_layout
    when the source and sink layouts can be made identical, so the cumsum
    runs directly on the canonical layout and redundant converts are removed.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleLowerExclusiveCumsum
    : Pass<"triton-tle-lower-exclusive-cumsum", "mlir::ModuleOp"> {
  let summary = "lower tle.exclusive_cumsum with a specialized Triton IR pattern";

  let description = [{
    This pass lowers ``tle.exclusive_cumsum`` into Triton IR with a
    cumsum-specialized pattern: one ``tt.scan`` for prefix accumulation and an
    index-select ``tt.reduce`` to extract the tail value used as total sum.
    This avoids a second add-reduction over the original input.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleLowerExtractTile
    : Pass<"triton-tle-lower-extract-tile", "mlir::ModuleOp"> {
  let summary = "Lower TLE extract_tile operations";

  let description = [{
    This pass lowers tle.extract_tile operations to target-specific GPU
    operations or equivalent Triton/TritonGPU IR patterns.
  }];

  let dependentDialects = ["mlir::triton::tle::TleDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonTleLowerInsertTile
    : Pass<"triton-tle-lower-insert-tile", "mlir::ModuleOp"> {
  let summary = "Lower TLE insert_tile operations";

  let description = [{
    This pass lowers tle.insert_tile operations to target-specific GPU
    operations or equivalent Triton/TritonGPU IR patterns.
  }];

  let dependentDialects = ["mlir::triton::tle::TleDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}
def TleConvertArgToMemDesc
    : Pass<"tle-convert-arg-to-memdesc", "mlir::ModuleOp"> {}

def TleRemoveRedundantCopy
    : Pass<"tle-remove-redundant-copy", "mlir::ModuleOp"> {}

def TleDSLRegionInline : Pass<"tle-dslregion-inline", "mlir::ModuleOp"> {}

def TritonTleAddDistributedParams
    : Pass<"triton-tle-add-distributed-params", "mlir::ModuleOp"> {

  let summary = "Add distributed runtime parameters to Triton kernel functions";

  let description = [{
    This pass updates Triton kernel function signatures by inserting two
    runtime parameters at the beginning of each kernel function:

      * device_comm_ptr
      * device_mem_ptr

    These parameters provide device-side communication and runtime state
    required by distributed TLE operations.

    All call sites and function argument uses are updated accordingly.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}


#endif // TRITON_TLE_PASSES
