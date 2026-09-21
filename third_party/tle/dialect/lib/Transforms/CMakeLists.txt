# flagtree tle
add_triton_library(TritonTLETransforms
  TleFuseNodeRemoteTransfers.cpp
  TleEarlyAssignMemorySpace.cpp
  TleSelectEncodings.cpp
  TleInsertLocalPointerBarriers.cpp
  TleOptimizeLocalPointerLoads.cpp
  TleOptimizeLocalPointerStores.cpp
  TleOptimizeLocalPointerAsyncStores.cpp
  TlePromoteLocalStoreStaging.cpp
  TleTileStylePipelineSchedule.cpp
  TleDowngradeInvalidAsyncCopy.cpp
  TleOptimizeExclusiveCumsumLayouts.cpp
  TleLowerExclusiveCumsum.cpp
  TleTileToLLVMUtils.cpp
  ExtractTileToLLVM.cpp
  InsertTileToLLVM.cpp
  TleLowerExtractTile.cpp
  TleLowerInsertTile.cpp
  ConvertArgToMemDesc.cpp
  RemoveRedundantCopy.cpp
  DSLRegionInline.cpp
  TleUtility.cpp
  TleAddDistributedParams.cpp

  DEPENDS
  TleTransformsIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  ${TLE_TRITON_GPU_ATTR_DEPENDS}
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonTLEAnalysis
  TritonGPUIR
  TritonAnalysis
  TritonIR
)

target_include_directories(TritonTLETransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/include
  ${CMAKE_SOURCE_DIR}/include
)
