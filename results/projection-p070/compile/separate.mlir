!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32>, layout_copy_tv = <"((4,64),((2,2,2),1)):((128,1),((512,64,1024),0))">, tiler_mn = <"[(8,8,2):(1,16,8);(4,2,2):(2,1,8)]">>
!memref_gmem_bf16 = !cute.memref<bf16, gmem, align<16>, "(?,?,?):(?{i64 div=8},?{i64 div=8},1)">
!memref_gmem_bf16_1 = !cute.memref<bf16, gmem, align<16>, "(?,?,?):(?{i64 div=8},1,?{i64 div=8})">
!memref_gmem_f32 = !cute.memref<f32, gmem, align<16>, "(?,?,?):(?{i64 div=4},?{i64 div=4},1)">
!memref_gmem_f32_1 = !cute.memref<f32, gmem, align<16>, "(?,?,?):(?{i64 div=4},1,?{i64 div=4})">
!memref_rmem_f32 = !cute.memref<f32, rmem, align<32>, "((2,2,16),1,1):((1,2,4),0,0)">
!memref_rmem_f32_1 = !cute.memref<f32, rmem, align<32>, "((1,(2,2,2)),1,2):((0,(1,2,4)),0,8)">
!memref_rmem_f32_2 = !cute.memref<f32, rmem, align<32>, "((2,2,2),1,(8,1)):((1,2,4),0,(8,0))">
!memref_rmem_f32_3 = !cute.memref<f32, rmem, align<32>, "((2,2,2),1,2,(1,1,1),1,4):((1,2,4),0,8,(0,0,0),0,16)">
!memref_rmem_f32_4 = !cute.memref<f32, rmem, align<32>, "((2,2,2),1,2,1,4):((1,2,4),0,8,0,16)">
!memref_rmem_f32_5 = !cute.memref<f32, rmem, align<32>, "((2,2,2),1,2,(1,4)):((1,2,4),0,8,(0,16))">
!memref_rmem_f32_6 = !cute.memref<f32, rmem, align<32>, "((1,8),1,2,(1,4)):((0,1),0,8,(0,16))">
!memref_rmem_f32_7 = !cute.memref<f32, rmem, align<32>, "((1,8),1,2):((0,1),0,8)">
!memref_rmem_f32_8 = !cute.memref<f32, rmem, align<32>, "(16,1):(1,0)">
!memref_rmem_f32_9 = !cute.memref<f32, rmem, align<32>, "(8,2):(1,8)">
!memref_rmem_f32_10 = !cute.memref<f32, rmem, align<32>, "(1,2,2,2,1,2):(0,1,2,4,0,8)">
!memref_rmem_f32_11 = !cute.memref<f32, rmem, align<32>, "(1,2,2,1,2):(0,2,4,0,8)">
!memref_rmem_f32_12 = !cute.memref<f32, rmem, "(1,2,2,1,2):(0,2,4,0,8)">
!memref_rmem_f32_13 = !cute.memref<f32, rmem, align<32>, "(1,2,2,1,2):(0,1,2,0,4)">
!memref_smem_bf16 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">
!memref_smem_bf16_1 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "(((8,16),(64,1)),(1,6)):(((64,512),(1,0)),(0,8192))">
!memref_smem_bf16_2 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "(((8,16),(64,1))):(((64,512),(1,0)))">
!memref_smem_bf16_3 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((8192,1)):((1,0))">
!memref_smem_bf16_4 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((8192,1),1):((1,0),0)">
!memref_smem_bf16_5 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((8192,1),(1)):((1,0),(0))">
!memref_smem_bf16_6 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((64,16),1,4,(1,6)):((64,1),0,16,(0,8192))">
!memref_smem_bf16_7 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((128,16),1,4,(1,6)):((64,1),0,16,(0,8192))">
!memref_smem_f32 = !cute.memref<f32, smem, align<1024>, S<3,4,3>, "((8,16),(32,1),(1,2)):((32,256),(1,0),(0,4096))">
!memref_smem_f32_1 = !cute.memref<f32, smem, align<1024>, S<3,4,3>, "(((8,16),(32,1)),(1,2)):(((32,256),(1,0)),(0,4096))">
!memref_smem_f32_2 = !cute.memref<f32, smem, align<1024>, S<3,4,3>, "((4096,1),(1,2)):((1,0),(0,4096))">
!memref_smem_f32_3 = !cute.memref<f32, smem, align<8>, S<3,4,3>, "((1,(2,2,2)),1,2,(1,2)):((0,(1,256,8)),0,16,(0,4096))">
!memref_smem_f32_4 = !cute.memref<f32, smem, align<8>, S<3,4,3>, "((1,(2,2,2)),1,2):((0,(1,256,8)),0,16)">
!memref_smem_f32_5 = !cute.memref<f32, smem, align<8>, S<3,4,3>, "(1,2,2,2,1,2):(0,1,256,8,0,16)">
!memref_smem_f32_6 = !cute.memref<f32, smem, align<8>, S<3,4,3>, "(1,2,2,1,2):(0,256,8,0,16)">
!memref_smem_f32_7 = !cute.memref<f32, smem, S<3,4,3>, "(1,2,2,1,2):(0,256,8,0,16)">
!memref_smem_f32_8 = !cute.memref<f32, smem, align<1024>, S<3,4,3>, "((4096,1)):((1,0))">
!memref_smem_f32_9 = !cute.memref<f32, smem, align<1024>, S<3,4,3>, "((4096,1),1):((1,0),0)">
!memref_smem_f32_10 = !cute.memref<f32, smem, align<1024>, S<3,4,3>, "((4096,1),(1)):((1,0),(0))">
!memref_smem_i32 = !cute.memref<i32, smem, align<16>, "(4,1):(1,0)">
!memref_smem_i8 = !cute.memref<i8, smem, align<1024>, "229376:1">
!memref_smem_i8_1 = !cute.memref<i8, smem, align<8>, "96:1">
!memref_smem_i8_2 = !cute.memref<i8, smem, align<8>, "16:1">
!memref_smem_i8_3 = !cute.memref<i8, smem, align<16>, "16:1">
!mma_bf16_bf16_f32_64x128x16 = !cute.tiled_mma<!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, atom_layout_MNK = <"(2,1,1):(1,0,0)">, permutation_MNK = <"[_;_;_]">>
#loop_unroll = #llvm.loop_unroll<disable = true, count = 1 : i32>
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll>
module attributes {gpu.container_module} {
  gpu.module @kernels {
    cuda.kernel @kernel_cutlass_kernel_quackgemm_default_epiGemmDefaultSm90_object_at__TiledMMA_ThrLayoutVMNK128211112800_PermutationMNK____MMAAtom_ThrID1281_ShapeMNK6412816_TVLayoutA12864160164_TVLayoutB_0(%arg0: !mma_bf16_bf16_f32_64x128x16, %arg1: !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN> {cute_nvgpu.grid_constant}, %arg2: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, %arg3: !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN> {cute_nvgpu.grid_constant}, %arg4: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, %arg5: !cute_nvgpu.atom.non_exec_tiled_tma_store<f32, copy_bits = 131072, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = F32_RN> {cute_nvgpu.grid_constant}, %arg6: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, %arg7: !cute.layout<"(1,1,1):(0,0,0)">, %arg8: !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">, %arg9: !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">, %arg10: !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(32,1),(1,2)):((32,256),(1,0),(0,4096))">, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, cute.kernel, cutlass_launch_facts = {cluster_launch = true, cooperative_launch = false, exact_block_dim = array<i64: 384, 1, 1>, exact_cluster_dim = array<i64: 1, 1, 1>, schema_version = 1 : i64}, gpu.kernel, nvvm.minctasm = 1 : i32, nvvm.reqntid = array<i32: 384, 1, 1>, smem.hint_smem_base_uniform, smem.partition_num = 2 : i32} {
      %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
      %c3_i32 = arith.constant 3 : i32
      %c256_i32 = arith.constant 256 : i32
      %c4_i32 = arith.constant 4 : i32
      %c128_i32 = arith.constant 128 : i32
      %c6_i32 = arith.constant 6 : i32
      %c32768_i32 = arith.constant 32768 : i32
      %c10000000_i32 = arith.constant 10000000 : i32
      %c2_i32 = arith.constant 2 : i32
      %c-1_i32 = arith.constant -1 : i32
      %c9_i32 = arith.constant 9 : i32
      %true = arith.constant true
      %false = arith.constant false
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c32_i32 = arith.constant 32 : i32
      %0 = nvvm.read.ptx.sreg.tid.x : i32
      %1 = nvvm.read.ptx.sreg.tid.y : i32
      %2 = nvvm.read.ptx.sreg.tid.z : i32
      %3 = nvvm.read.ptx.sreg.ntid.x : i32
      %4 = nvvm.read.ptx.sreg.ntid.y : i32
      %5 = arith.muli %1, %3 : i32
      %6 = arith.addi %0, %5 : i32
      %7 = arith.muli %2, %3 : i32
      %8 = arith.muli %7, %4 : i32
      %9 = arith.addi %6, %8 : i32
      %10 = arith.floordivsi %9, %c32_i32 : i32
      %11 = cute_nvgpu.arch.make_warp_uniform(%10) : i32
      %12 = cute_nvgpu.arch.make_warp_uniform(%11) : i32
      %13 = arith.cmpi eq, %12, %c8_i32 : i32
      scf.if %13 {
        cute_nvgpu.prefetch_tma_desc(%arg1) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> ()
        cute_nvgpu.prefetch_tma_desc(%arg3) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> ()
        cute_nvgpu.prefetch_tma_desc(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f32, copy_bits = 131072, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = F32_RN>) -> ()
      }
      %smem = cute.memref.alloca() {smem.partition_id = 1 : i32, smem.struct_fields = ["sD:32768:0", "sC:0:32768", "epi:0:32768", "sA:98304:32768", "sB:98304:131072", "sAuxA:0:229376", "sSFA:0:229376", "sSFB:0:229376"]} : !memref_smem_i8
      %iter = cute.get_iter(%smem) : !memref_smem_i8
      %int_tuple = cute.make_int_tuple() : () -> !cute.int_tuple<"32768">
      %ptr = cute.add_offset(%iter, %int_tuple) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"32768">) -> !cute.ptr<i8, smem, align<1024>>
      %int_tuple_0 = cute.make_int_tuple() : () -> !cute.int_tuple<"131072">
      %ptr_1 = cute.add_offset(%iter, %int_tuple_0) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"131072">) -> !cute.ptr<i8, smem, align<1024>>
      %smem_2 = cute.memref.alloca() {smem.partition_id = 0 : i32} : !memref_smem_i8_1
      %iter_3 = cute.get_iter(%smem_2) : !memref_smem_i8_1
      %iter_4 = cute.recast_iter(%iter_3) : !cute.ptr<i8, smem, align<8>> to !cute.ptr<i64, smem>
      %14 = arith.cmpi eq, %12, %c0_i32 : i32
      scf.if %14 {
        %23 = nvvm.elect.sync -> i1
        scf.if %23 {
          %24 = builtin.unrealized_conversion_cast %iter_4 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %24, %c1_i32 : !llvm.ptr<3>, i32
          %int_tuple_21 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
          %ptr_22 = cute.add_offset(%iter_4, %int_tuple_21) : (!cute.ptr<i64, smem>, !cute.int_tuple<"1">) -> !cute.ptr<i64, smem>
          %25 = builtin.unrealized_conversion_cast %ptr_22 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %25, %c1_i32 : !llvm.ptr<3>, i32
          %int_tuple_23 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
          %ptr_24 = cute.add_offset(%iter_4, %int_tuple_23) : (!cute.ptr<i64, smem>, !cute.int_tuple<"2">) -> !cute.ptr<i64, smem>
          %26 = builtin.unrealized_conversion_cast %ptr_24 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %26, %c1_i32 : !llvm.ptr<3>, i32
          %int_tuple_25 = cute.make_int_tuple() : () -> !cute.int_tuple<"3">
          %ptr_26 = cute.add_offset(%iter_4, %int_tuple_25) : (!cute.ptr<i64, smem>, !cute.int_tuple<"3">) -> !cute.ptr<i64, smem>
          %27 = builtin.unrealized_conversion_cast %ptr_26 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %27, %c1_i32 : !llvm.ptr<3>, i32
          %int_tuple_27 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
          %ptr_28 = cute.add_offset(%iter_4, %int_tuple_27) : (!cute.ptr<i64, smem>, !cute.int_tuple<"4">) -> !cute.ptr<i64, smem>
          %28 = builtin.unrealized_conversion_cast %ptr_28 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %28, %c1_i32 : !llvm.ptr<3>, i32
          %int_tuple_29 = cute.make_int_tuple() : () -> !cute.int_tuple<"5">
          %ptr_30 = cute.add_offset(%iter_4, %int_tuple_29) : (!cute.ptr<i64, smem>, !cute.int_tuple<"5">) -> !cute.ptr<i64, smem>
          %29 = builtin.unrealized_conversion_cast %ptr_30 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %29, %c1_i32 : !llvm.ptr<3>, i32
        }
      }
      %int_tuple_5 = cute.make_int_tuple() : () -> !cute.int_tuple<"6">
      %ptr_6 = cute.add_offset(%iter_4, %int_tuple_5) : (!cute.ptr<i64, smem>, !cute.int_tuple<"6">) -> !cute.ptr<i64, smem>
      scf.if %14 {
        %23 = nvvm.elect.sync -> i1
        scf.if %23 {
          %24 = builtin.unrealized_conversion_cast %ptr_6 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %24, %c8_i32 : !llvm.ptr<3>, i32
          %int_tuple_21 = cute.make_int_tuple() : () -> !cute.int_tuple<"7">
          %ptr_22 = cute.add_offset(%iter_4, %int_tuple_21) : (!cute.ptr<i64, smem>, !cute.int_tuple<"7">) -> !cute.ptr<i64, smem>
          %25 = builtin.unrealized_conversion_cast %ptr_22 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %25, %c8_i32 : !llvm.ptr<3>, i32
          %int_tuple_23 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
          %ptr_24 = cute.add_offset(%iter_4, %int_tuple_23) : (!cute.ptr<i64, smem>, !cute.int_tuple<"8">) -> !cute.ptr<i64, smem>
          %26 = builtin.unrealized_conversion_cast %ptr_24 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %26, %c8_i32 : !llvm.ptr<3>, i32
          %int_tuple_25 = cute.make_int_tuple() : () -> !cute.int_tuple<"9">
          %ptr_26 = cute.add_offset(%iter_4, %int_tuple_25) : (!cute.ptr<i64, smem>, !cute.int_tuple<"9">) -> !cute.ptr<i64, smem>
          %27 = builtin.unrealized_conversion_cast %ptr_26 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %27, %c8_i32 : !llvm.ptr<3>, i32
          %int_tuple_27 = cute.make_int_tuple() : () -> !cute.int_tuple<"10">
          %ptr_28 = cute.add_offset(%iter_4, %int_tuple_27) : (!cute.ptr<i64, smem>, !cute.int_tuple<"10">) -> !cute.ptr<i64, smem>
          %28 = builtin.unrealized_conversion_cast %ptr_28 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %28, %c8_i32 : !llvm.ptr<3>, i32
          %int_tuple_29 = cute.make_int_tuple() : () -> !cute.int_tuple<"11">
          %ptr_30 = cute.add_offset(%iter_4, %int_tuple_29) : (!cute.ptr<i64, smem>, !cute.int_tuple<"11">) -> !cute.ptr<i64, smem>
          %29 = builtin.unrealized_conversion_cast %ptr_30 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %29, %c8_i32 : !llvm.ptr<3>, i32
        }
      }
      %15 = arith.remsi %0, %c32_i32 : i32
      %16 = arith.cmpi slt, %15, %c1_i32 : i32
      %smem_7 = cute.memref.alloca() {smem.partition_id = 0 : i32} : !memref_smem_i8_2
      %iter_8 = cute.get_iter(%smem_7) : !memref_smem_i8_2
      %iter_9 = cute.recast_iter(%iter_8) : !cute.ptr<i8, smem, align<8>> to !cute.ptr<i64, smem>
      scf.if %14 {
        %23 = nvvm.elect.sync -> i1
        scf.if %23 {
          %24 = builtin.unrealized_conversion_cast %iter_9 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %24, %c1_i32 : !llvm.ptr<3>, i32
        }
      }
      %int_tuple_10 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
      %ptr_11 = cute.add_offset(%iter_9, %int_tuple_10) : (!cute.ptr<i64, smem>, !cute.int_tuple<"1">) -> !cute.ptr<i64, smem>
      scf.if %14 {
        %23 = nvvm.elect.sync -> i1
        scf.if %23 {
          %24 = builtin.unrealized_conversion_cast %ptr_11 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %24, %c9_i32 : !llvm.ptr<3>, i32
        }
      }
      %shape = cute.make_shape() : () -> !cute.shape<"(4,1)">
      %lay = cute.make_layout(%shape) : !cute.layout<"(4,1):(1,0)">
      %smem_12 = cute.memref.alloca() {smem.partition_id = 0 : i32} : !memref_smem_i8_3
      %iter_13 = cute.get_iter(%smem_12) : !memref_smem_i8_3
      %iter_14 = cute.recast_iter(%iter_13) : !cute.ptr<i8, smem, align<16>> to !cute.ptr<i32, smem, align<16>>
      %view = cute.make_view(%iter_14, %lay) : !memref_smem_i32
      nvvm.fence.mbarrier.init
      %17 = cute.composed_get_outer(%arg8) : (!cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">) -> !cute.layout<"((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">
      %iter_15 = cute.recast_iter(%ptr) : !cute.ptr<i8, smem, align<1024>> to !cute.ptr<bf16, smem, align<1024>, S<3,4,3>>
      %view_16 = cute.make_view(%iter_15, %17) : !memref_smem_bf16
      %18 = cute.composed_get_outer(%arg9) : (!cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">) -> !cute.layout<"((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">
      %iter_17 = cute.recast_iter(%ptr_1) : !cute.ptr<i8, smem, align<1024>> to !cute.ptr<bf16, smem, align<1024>, S<3,4,3>>
      %view_18 = cute.make_view(%iter_17, %18) : !memref_smem_bf16
      %19 = cute.composed_get_outer(%arg10) : (!cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(32,1),(1,2)):((32,256),(1,0),(0,4096))">) -> !cute.layout<"((8,16),(32,1),(1,2)):((32,256),(1,0),(0,4096))">
      %iter_19 = cute.recast_iter(%iter) : !cute.ptr<i8, smem, align<1024>> to !cute.ptr<f32, smem, align<1024>, S<3,4,3>>
      %view_20 = cute.make_view(%iter_19, %19) : !memref_smem_f32
      %sz = cute.size(%arg4) <{mode = [1]}> : (!cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">) -> !cute.int_tuple<"?">
      %e0 = cute.get_leaves(%sz) : !cute.int_tuple<"?">
      %20 = cute.get_scalars(%e0) : !cute.int_tuple<"?">
      nvvm.barrier
      %21 = arith.cmpi sge, %12, %c8_i32 : i32
      scf.if %21 {
        nvvm.setmaxregister decrease 40
        %23 = arith.cmpi slt, %12, %c9_i32 : i32
        %24 = arith.andi %21, %23 : i1
        scf.if %24 {
          nvvm.griddepcontrol wait
          %25 = nvvm.read.ptx.sreg.ctaid.z : i32
          %int_tuple_21 = cute.make_int_tuple(%arg11, %arg12, %arg13) : (i32, i32, i32) -> !cute.int_tuple<"(?,?,?)">
          %sz_22 = cute.size(%int_tuple_21) : (!cute.int_tuple<"(?,?,?)">) -> !cute.int_tuple<"?">
          %e0_23 = cute.get_leaves(%sz_22) : !cute.int_tuple<"?">
          %int_tuple_24 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
          %mul = cute.tuple_mul(%e0_23, %int_tuple_24) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?{div=2}">
          %26 = cute.get_scalars(%mul) : !cute.int_tuple<"?{div=2}">
          %27 = arith.cmpi sgt, %26, %25 : i32
          %28:4 = scf.if %27 -> (i32, i32, i32, i32) {
            %32 = arith.floordivsi %25, %c2_i32 : i32
            %33 = arith.muli %32, %c2_i32 : i32
            %34 = arith.subi %25, %33 : i32
            %35 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %32, %arg15 : (i32, i32) -> i32
            %36 = arith.shrui %35, %arg16 : i32
            %37 = arith.cmpi eq, %arg15, %c0_i32 : i32
            %38 = arith.select %37, %32, %36 : i32
            %39 = arith.muli %38, %arg17 : i32
            %40 = arith.subi %32, %39 : i32
            %41 = arith.muli %38, %c2_i32 : i32
            %42 = arith.addi %41, %34 : i32
            %43 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %40, %arg25 : (i32, i32) -> i32
            %44 = arith.shrui %43, %arg26 : i32
            %45 = arith.cmpi eq, %arg25, %c0_i32 : i32
            %46 = arith.select %45, %40, %44 : i32
            %47 = arith.muli %46, %arg27 : i32
            %48 = arith.subi %40, %47 : i32
            %49 = arith.cmpi slt, %46, %arg18 : i32
            %50:2 = scf.if %49 -> (i32, i32) {
              %62 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %48, %arg19 : (i32, i32) -> i32
              %63 = arith.shrui %62, %arg20 : i32
              %64 = arith.cmpi eq, %arg19, %c0_i32 : i32
              %65 = arith.select %64, %48, %63 : i32
              %66 = arith.muli %65, %arg21 : i32
              %67 = arith.subi %48, %66 : i32
              scf.yield %65, %67 : i32, i32
            } else {
              %62 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %48, %arg22 : (i32, i32) -> i32
              %63 = arith.shrui %62, %arg23 : i32
              %64 = arith.cmpi eq, %arg22, %c0_i32 : i32
              %65 = arith.select %64, %48, %63 : i32
              %66 = arith.muli %65, %arg24 : i32
              %67 = arith.subi %48, %66 : i32
              scf.yield %65, %67 : i32, i32
            }
            %51 = arith.remsi %46, %c2_i32 : i32
            %52 = arith.cmpi eq, %51, %c1_i32 : i32
            %53 = scf.if %52 -> (i32) {
              %62 = arith.cmpi eq, %arg14, %c0_i32 : i32
              %63 = arith.select %62, %arg12, %arg11 : i32
              %64 = arith.subi %63, %c1_i32 : i32
              %65 = arith.subi %64, %50#0 : i32
              scf.yield %65 : i32
            } else {
              scf.yield %50#0 : i32
            }
            %54 = arith.muli %46, %arg21 : i32
            %55 = arith.addi %54, %50#1 : i32
            %56 = arith.cmpi eq, %arg14, %c1_i32 : i32
            %57 = arith.select %56, %53, %55 : i32
            %58 = arith.select %56, %55, %53 : i32
            %59 = arith.floordivsi %42, %c2_i32 : i32
            %60 = arith.muli %59, %c2_i32 : i32
            %61 = arith.subi %42, %60 : i32
            scf.yield %61, %57, %58, %59 : i32, i32, i32, i32
          } else {
            scf.yield %c0_i32, %c0_i32, %c0_i32, %c0_i32 : i32, i32, i32, i32
          }
          %29:12 = scf.while (%arg28 = %c0_i32, %arg29 = %c0_i32, %arg30 = %c1_i32, %arg31 = %28#1, %arg32 = %28#2, %arg33 = %28#0, %arg34 = %28#3, %arg35 = %27, %arg36 = %25, %arg37 = %c0_i32, %arg38 = %c0_i32, %arg39 = %c0_i32, %arg40 = %c0_i32) : (i32, i32, i32, i32, i32, i32, i32, i1, i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
            scf.condition(%arg35) %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg36, %arg37, %arg38, %arg39, %arg40 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
          } do {
          ^bb0(%arg28: i32, %arg29: i32, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32):
            iket.range_push("tma_load")
            %coord = cute.make_coord(%arg34) : (i32) -> !cute.coord<"(_,_,?)">
            %slice = cute.slice(%arg2, %coord) : !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute.coord<"(_,_,?)">
            %tile = cute.make_tile() : () -> !cute.tile<"[128:1;64:1]">
            %coord_27 = cute.make_coord(%arg31) : (i32) -> !cute.coord<"(?,_)">
            %tiled_view = cute.local_tile(%slice, %tile, %coord_27) : (!cute.coord_tensor<"(0,0,?)", "(?,?):(1@1,1@0)">, !cute.tile<"[128:1;64:1]">, !cute.coord<"(?,_)">) -> !cute.coord_tensor<"(0,?{div=128},?)", "(128,64,?):(1@1,1@0,64@0)">
            %grouped = cute.group_modes(%view_16) <0, 2> : (!memref_smem_bf16) -> !memref_smem_bf16_1
            %grouped_28 = cute.group_modes(%tiled_view) <0, 2> : (!cute.coord_tensor<"(0,?{div=128},?)", "(128,64,?):(1@1,1@0,64@0)">) -> !cute.coord_tensor<"(0,?{div=128},?)", "((128,64),?):((1@1,1@0),64@0)">
            %slice_29 = cute.slice(%arg4, %coord) : !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute.coord<"(_,_,?)">
            %coord_30 = cute.make_coord(%arg32) : (i32) -> !cute.coord<"(?,_)">
            %tiled_view_31 = cute.local_tile(%slice_29, %tile, %coord_30) : (!cute.coord_tensor<"(0,0,?)", "(?,?):(1@1,1@0)">, !cute.tile<"[128:1;64:1]">, !cute.coord<"(?,_)">) -> !cute.coord_tensor<"(0,?{div=128},?)", "(128,64,?):(1@1,1@0,64@0)">
            %grouped_32 = cute.group_modes(%view_18) <0, 2> : (!memref_smem_bf16) -> !memref_smem_bf16_1
            %grouped_33 = cute.group_modes(%tiled_view_31) <0, 2> : (!cute.coord_tensor<"(0,?{div=128},?)", "(128,64,?):(1@1,1@0,64@0)">) -> !cute.coord_tensor<"(0,?{div=128},?)", "((128,64),?):((1@1,1@0),64@0)">
            %int_tuple_34 = cute.make_int_tuple(%20) : (i32) -> !cute.int_tuple<"?">
            %tile_35 = cute.make_tile() : () -> !cute.tile<"64:1">
            %shp = cute.ceil_div(%int_tuple_34, %tile_35) : !cute.int_tuple<"?">, !cute.tile<"64:1">
            %e0_36 = cute.get_leaves(%shp) : !cute.int_tuple<"?">
            %div = cute.tuple_div(%e0_36, %int_tuple_24) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?">
            %mul_37 = cute.tuple_mul(%div, %int_tuple_24) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?{div=2}">
            %sub = cute.tuple_sub(%e0_36, %mul_37) : (!cute.int_tuple<"?">, !cute.int_tuple<"?{div=2}">) -> !cute.int_tuple<"?">
            %32 = cute.get_scalars(%sub) : !cute.int_tuple<"?">
            %int_tuple_38 = cute.make_int_tuple(%arg33) : (i32) -> !cute.int_tuple<"?">
            %mul_39 = cute.tuple_mul(%int_tuple_38, %div) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
            %33 = arith.minsi %arg33, %32 : i32
            %int_tuple_40 = cute.make_int_tuple(%33) : (i32) -> !cute.int_tuple<"?">
            %add = cute.tuple_add(%mul_39, %int_tuple_40) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
            %34 = arith.cmpi sgt, %32, %arg33 : i32
            %35 = arith.extui %34 : i1 to i32
            %int_tuple_41 = cute.make_int_tuple(%35) : (i32) -> !cute.int_tuple<"?">
            %add_42 = cute.tuple_add(%div, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
            %36 = cute.get_scalars(%add_42) : !cute.int_tuple<"?">
            %37 = arith.cmpi sgt, %36, %c0_i32 : i32
            %38 = scf.if %37 -> (i1) {
              %int_tuple_50 = cute.make_int_tuple(%arg29) : (i32) -> !cute.int_tuple<"?">
              %ptr_51 = cute.add_offset(%ptr_6, %int_tuple_50) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %67 = builtin.unrealized_conversion_cast %ptr_51 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %68 = nvvm.mbarrier.wait.parity %67, %arg30 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
              scf.yield %68 : i1
            } else {
              scf.yield %true : i1
            }
            %39:4 = scf.for %arg40 = %c0_i32 to %36 step %c1_i32 iter_args(%arg41 = %38, %arg42 = %arg28, %arg43 = %arg29, %arg44 = %arg30) -> (i1, i32, i32, i32)  : i32 {
              %67 = arith.extui %arg41 : i1 to i32
              %68 = arith.cmpi eq, %67, %c0_i32 : i32
              scf.if %68 {
                %int_tuple_95 = cute.make_int_tuple(%arg43) : (i32) -> !cute.int_tuple<"?">
                %ptr_96 = cute.add_offset(%ptr_6, %int_tuple_95) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
                %82 = builtin.unrealized_conversion_cast %ptr_96 : !cute.ptr<i64, smem> to !llvm.ptr<3>
                nvvm.mbarrier.try_wait.parity %82, %arg44, %c10000000_i32 : !llvm.ptr<3>, i32, i32
              }
              %69 = nvvm.elect.sync -> i1
              scf.if %69 {
                %int_tuple_95 = cute.make_int_tuple(%arg43) : (i32) -> !cute.int_tuple<"?">
                %ptr_96 = cute.add_offset(%iter_4, %int_tuple_95) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
                %82 = builtin.unrealized_conversion_cast %ptr_96 : !cute.ptr<i64, smem> to !llvm.ptr<3>
                %83 = nvvm.mbarrier.arrive.expect_tx %82, %c32768_i32 : !llvm.ptr<3>, i32 -> i64
              }
              %int_tuple_50 = cute.make_int_tuple(%arg43) : (i32) -> !cute.int_tuple<"?">
              %ptr_51 = cute.add_offset(%iter_4, %int_tuple_50) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %int_tuple_52 = cute.make_int_tuple(%arg40) : (i32) -> !cute.int_tuple<"?">
              %add_53 = cute.tuple_add(%add, %int_tuple_52) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
              %coord_54 = cute.make_coord(%add_53) : (!cute.int_tuple<"?">) -> !cute.coord<"(_,?)">
              %slice_55 = cute.slice(%grouped_28, %coord_54) : !cute.coord_tensor<"(0,?{div=128},?)", "((128,64),?):((1@1,1@0),64@0)">, !cute.coord<"(_,?)">
              %coord_56 = cute.make_coord(%arg43) : (i32) -> !cute.coord<"(_,?)">
              %slice_57 = cute.slice(%grouped, %coord_56) : !memref_smem_bf16_1, !cute.coord<"(_,?)">
              %shape_58 = cute.make_shape() : () -> !cute.shape<"1">
              %lay_59 = cute.make_layout(%shape_58) : !cute.layout<"1:0">
              %coord_60 = cute.make_coord() : () -> !cute.coord<"0">
              %res_smem_tensor, %res_target_tensors = cute_nvgpu.atom.tma_partition(%arg1, %coord_60, %lay_59, %slice_57, %slice_55) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord<"0">, !cute.layout<"1:0">, !memref_smem_bf16_2, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "((128,64)):((1@1,1@0))">) -> (!memref_smem_bf16_3, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">)
              %filtered = cute.filter_zeros(%res_smem_tensor) : !memref_smem_bf16_3
              %iter_61 = cute.get_iter(%filtered) : !memref_smem_bf16_3
              %filtered_62 = cute.filter_zeros(%res_target_tensors) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %iter_63 = cute.get_iter(%filtered_62) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %tup = cute.deref_arith_tuple_iter(%iter_63) : !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
              %e0_64, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=64},?{div=128},?)">
              %lay_65 = cute.get_layout(%filtered_62) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %lay_66 = cute.get_layout(%filtered) : !memref_smem_bf16_3
              %append = cute.append_to_rank<2> (%lay_65, %lay_59) : !cute.layout<"(((64,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
              %int_tuple_67 = cute.make_int_tuple(%e0_64, %e1, %e2) : (!cute.int_tuple<"?{div=64}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=64},?{div=128},?)">
              %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_67) : (!cute.int_tuple<"(?{div=64},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
              %view_68 = cute.make_view(%int_tup_iter, %append) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),1):(((1@0,1@1),0),0)">
              %grouped_69 = cute.group_modes(%view_68) <1, 2> : (!cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),(1)):(((1@0,1@1),0),(0))">
              %append_70 = cute.append_to_rank<2> (%lay_66, %lay_59) : !cute.layout<"((8192,1)):((1,0))">, !cute.layout<"1:0">
              %view_71 = cute.make_view(%iter_61, %append_70) : !memref_smem_bf16_4
              %grouped_72 = cute.group_modes(%view_71) <1, 2> : (!memref_smem_bf16_4) -> !memref_smem_bf16_5
              %70 = cute_nvgpu.atom.make_exec_tma(%arg1) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> !cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>
              %71 = cute_nvgpu.atom.set_value<tma_bar>(%70, %ptr_51) : (!cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>, !cute.ptr<i64, smem>)
              cute.copy(%71, %grouped_69, %grouped_72) {from_block_api, multicast_layout = #cute.layout<"(1,1):(1,0)">} : (!cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),(1)):(((1@0,1@1),0),(0))">, !memref_smem_bf16_5)
              %slice_73 = cute.slice(%grouped_33, %coord_54) : !cute.coord_tensor<"(0,?{div=128},?)", "((128,64),?):((1@1,1@0),64@0)">, !cute.coord<"(_,?)">
              %slice_74 = cute.slice(%grouped_32, %coord_56) : !memref_smem_bf16_1, !cute.coord<"(_,?)">
              %res_smem_tensor_75, %res_target_tensors_76 = cute_nvgpu.atom.tma_partition(%arg3, %coord_60, %lay_59, %slice_74, %slice_73) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord<"0">, !cute.layout<"1:0">, !memref_smem_bf16_2, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "((128,64)):((1@1,1@0))">) -> (!memref_smem_bf16_3, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">)
              %filtered_77 = cute.filter_zeros(%res_smem_tensor_75) : !memref_smem_bf16_3
              %iter_78 = cute.get_iter(%filtered_77) : !memref_smem_bf16_3
              %filtered_79 = cute.filter_zeros(%res_target_tensors_76) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %iter_80 = cute.get_iter(%filtered_79) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %tup_81 = cute.deref_arith_tuple_iter(%iter_80) : !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
              %e0_82, %e1_83, %e2_84 = cute.get_leaves(%tup_81) : !cute.int_tuple<"(?{div=64},?{div=128},?)">
              %lay_85 = cute.get_layout(%filtered_79) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %lay_86 = cute.get_layout(%filtered_77) : !memref_smem_bf16_3
              %append_87 = cute.append_to_rank<2> (%lay_85, %lay_59) : !cute.layout<"(((64,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
              %int_tuple_88 = cute.make_int_tuple(%e0_82, %e1_83, %e2_84) : (!cute.int_tuple<"?{div=64}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=64},?{div=128},?)">
              %int_tup_iter_89 = cute.make_arith_tuple_iter(%int_tuple_88) : (!cute.int_tuple<"(?{div=64},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
              %view_90 = cute.make_view(%int_tup_iter_89, %append_87) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),1):(((1@0,1@1),0),0)">
              %grouped_91 = cute.group_modes(%view_90) <1, 2> : (!cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),(1)):(((1@0,1@1),0),(0))">
              %append_92 = cute.append_to_rank<2> (%lay_86, %lay_59) : !cute.layout<"((8192,1)):((1,0))">, !cute.layout<"1:0">
              %view_93 = cute.make_view(%iter_78, %append_92) : !memref_smem_bf16_4
              %grouped_94 = cute.group_modes(%view_93) <1, 2> : (!memref_smem_bf16_4) -> !memref_smem_bf16_5
              %72 = cute_nvgpu.atom.make_exec_tma(%arg3) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> !cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>
              %73 = cute_nvgpu.atom.set_value<tma_bar>(%72, %ptr_51) : (!cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>, !cute.ptr<i64, smem>)
              cute.copy(%73, %grouped_91, %grouped_94) {from_block_api, multicast_layout = #cute.layout<"(1,1):(0,1)">} : (!cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),(1)):(((1@0,1@1),0),(0))">, !memref_smem_bf16_5)
              %74 = arith.addi %arg43, %c1_i32 : i32
              %75 = arith.addi %arg42, %c1_i32 : i32
              %76 = arith.cmpi eq, %74, %c6_i32 : i32
              %77 = arith.select %76, %c0_i32, %74 : i32
              %78 = scf.if %76 -> (i32) {
                %82 = arith.xori %arg44, %c1_i32 : i32
                scf.yield %82 : i32
              } else {
                scf.yield %arg44 : i32
              }
              %79 = arith.addi %arg40, %c1_i32 : i32
              %80 = arith.cmpi sgt, %36, %79 : i32
              %81 = scf.if %80 -> (i1) {
                %int_tuple_95 = cute.make_int_tuple(%77) : (i32) -> !cute.int_tuple<"?">
                %ptr_96 = cute.add_offset(%ptr_6, %int_tuple_95) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
                %82 = builtin.unrealized_conversion_cast %ptr_96 : !cute.ptr<i64, smem> to !llvm.ptr<3>
                %83 = nvvm.mbarrier.wait.parity %82, %78 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
                scf.yield %83 : i1
              } else {
                scf.yield %true : i1
              }
              scf.yield %81, %75, %77, %78 : i1, i32, i32, i32
            } {loop_annotation = #loop_annotation}
            iket.range_pop
            %40 = arith.addi %arg36, %c1_i32 : i32
            %41 = nvvm.read.ptx.sreg.nctaid.z : i32
            %42 = arith.addi %arg35, %41 : i32
            %43 = arith.cmpi sgt, %26, %42 : i32
            %44:4 = scf.if %43 -> (i32, i32, i32, i32) {
              %67 = arith.floordivsi %42, %c2_i32 : i32
              %68 = arith.muli %67, %c2_i32 : i32
              %69 = arith.subi %42, %68 : i32
              %70 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %67, %arg15 : (i32, i32) -> i32
              %71 = arith.shrui %70, %arg16 : i32
              %72 = arith.cmpi eq, %arg15, %c0_i32 : i32
              %73 = arith.select %72, %67, %71 : i32
              %74 = arith.muli %73, %arg17 : i32
              %75 = arith.subi %67, %74 : i32
              %76 = arith.muli %73, %c2_i32 : i32
              %77 = arith.addi %76, %69 : i32
              %78 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %75, %arg25 : (i32, i32) -> i32
              %79 = arith.shrui %78, %arg26 : i32
              %80 = arith.cmpi eq, %arg25, %c0_i32 : i32
              %81 = arith.select %80, %75, %79 : i32
              %82 = arith.muli %81, %arg27 : i32
              %83 = arith.subi %75, %82 : i32
              %84 = arith.cmpi slt, %81, %arg18 : i32
              %85:2 = scf.if %84 -> (i32, i32) {
                %97 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %83, %arg19 : (i32, i32) -> i32
                %98 = arith.shrui %97, %arg20 : i32
                %99 = arith.cmpi eq, %arg19, %c0_i32 : i32
                %100 = arith.select %99, %83, %98 : i32
                %101 = arith.muli %100, %arg21 : i32
                %102 = arith.subi %83, %101 : i32
                scf.yield %100, %102 : i32, i32
              } else {
                %97 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %83, %arg22 : (i32, i32) -> i32
                %98 = arith.shrui %97, %arg23 : i32
                %99 = arith.cmpi eq, %arg22, %c0_i32 : i32
                %100 = arith.select %99, %83, %98 : i32
                %101 = arith.muli %100, %arg24 : i32
                %102 = arith.subi %83, %101 : i32
                scf.yield %100, %102 : i32, i32
              }
              %86 = arith.remsi %81, %c2_i32 : i32
              %87 = arith.cmpi eq, %86, %c1_i32 : i32
              %88 = scf.if %87 -> (i32) {
                %97 = arith.cmpi eq, %arg14, %c0_i32 : i32
                %98 = arith.select %97, %arg12, %arg11 : i32
                %99 = arith.subi %98, %c1_i32 : i32
                %100 = arith.subi %99, %85#0 : i32
                scf.yield %100 : i32
              } else {
                scf.yield %85#0 : i32
              }
              %89 = arith.muli %81, %arg21 : i32
              %90 = arith.addi %89, %85#1 : i32
              %91 = arith.cmpi eq, %arg14, %c1_i32 : i32
              %92 = arith.select %91, %88, %90 : i32
              %93 = arith.select %91, %90, %88 : i32
              %94 = arith.floordivsi %77, %c2_i32 : i32
              %95 = arith.muli %94, %c2_i32 : i32
              %96 = arith.subi %77, %95 : i32
              scf.yield %96, %92, %93, %94 : i32, i32, i32, i32
            } else {
              scf.yield %c0_i32, %c0_i32, %c0_i32, %c0_i32 : i32, i32, i32, i32
            }
            %45 = arith.xori %arg39, %c1_i32 : i32
            %int_tuple_43 = cute.make_int_tuple(%arg38) : (i32) -> !cute.int_tuple<"?">
            %ptr_44 = cute.add_offset(%ptr_11, %int_tuple_43) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %46 = builtin.unrealized_conversion_cast %ptr_44 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %46, %45, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            %47 = arith.muli %44#3, %c2_i32 : i32
            %48 = arith.addi %47, %44#0 : i32
            %49 = arith.extui %43 : i1 to i32
            %50 = nvvm.read.ptx.sreg.laneid : i32
            %51 = arith.cmpi slt, %50, %c1_i32 : i32
            scf.if %51 {
              %coord_50 = cute.make_coord(%arg38) : (i32) -> !cute.coord<"(0,?)">
              cute.memref.store(%view, %coord_50, %44#1) : (!memref_smem_i32, !cute.coord<"(0,?)">, i32) -> ()
              %coord_51 = cute.make_coord(%arg38) : (i32) -> !cute.coord<"(1,?)">
              cute.memref.store(%view, %coord_51, %44#2) : (!memref_smem_i32, !cute.coord<"(1,?)">, i32) -> ()
              %coord_52 = cute.make_coord(%arg38) : (i32) -> !cute.coord<"(2,?)">
              cute.memref.store(%view, %coord_52, %48) : (!memref_smem_i32, !cute.coord<"(2,?)">, i32) -> ()
              %coord_53 = cute.make_coord(%arg38) : (i32) -> !cute.coord<"(3,?)">
              cute.memref.store(%view, %coord_53, %49) : (!memref_smem_i32, !cute.coord<"(3,?)">, i32) -> ()
              %ptr_54 = cute.add_offset(%iter_9, %int_tuple_43) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %67 = builtin.unrealized_conversion_cast %ptr_54 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %68 = nvvm.mbarrier.arrive %67, %c1_i32 : !llvm.ptr<3> -> i64
            }
            iket.range_push("fetch_wait")
            %ptr_45 = cute.add_offset(%iter_9, %int_tuple_43) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %52 = builtin.unrealized_conversion_cast %ptr_45 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %52, %arg39, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            iket.range_pop
            iket.range_push("fetch_decode")
            %coord_46 = cute.make_coord(%arg38) : (i32) -> !cute.coord<"(0,?)">
            %53 = cute.memref.load(%view, %coord_46) : (!memref_smem_i32, !cute.coord<"(0,?)">) -> i32
            %coord_47 = cute.make_coord(%arg38) : (i32) -> !cute.coord<"(1,?)">
            %54 = cute.memref.load(%view, %coord_47) : (!memref_smem_i32, !cute.coord<"(1,?)">) -> i32
            %coord_48 = cute.make_coord(%arg38) : (i32) -> !cute.coord<"(2,?)">
            %55 = cute.memref.load(%view, %coord_48) : (!memref_smem_i32, !cute.coord<"(2,?)">) -> i32
            %coord_49 = cute.make_coord(%arg38) : (i32) -> !cute.coord<"(3,?)">
            %56 = cute.memref.load(%view, %coord_49) : (!memref_smem_i32, !cute.coord<"(3,?)">) -> i32
            nvvm.bar.warp.sync %c-1_i32 : i32
            %57 = nvvm.elect.sync -> i1
            scf.if %57 {
              %67 = nvvm.mbarrier.arrive %46, %c1_i32 : !llvm.ptr<3> -> i64
            }
            %58 = arith.addi %arg38, %c1_i32 : i32
            %59 = arith.addi %arg37, %c1_i32 : i32
            %60 = arith.cmpi eq, %58, %c1_i32 : i32
            %61 = arith.select %60, %c0_i32, %58 : i32
            %62 = scf.if %60 -> (i32) {
              scf.yield %45 : i32
            } else {
              scf.yield %arg39 : i32
            }
            %63 = arith.cmpi ne, %56, %c0_i32 : i32
            iket.range_pop
            %64 = arith.floordivsi %55, %c2_i32 : i32
            %65 = arith.muli %64, %c2_i32 : i32
            %66 = arith.subi %55, %65 : i32
            scf.yield %39#1, %39#2, %39#3, %53, %54, %66, %64, %63, %42, %40, %59, %61, %62 : i32, i32, i32, i32, i32, i32, i32, i1, i32, i32, i32, i32, i32
          }
          scf.if %13 {
            %int_tuple_27 = cute.make_int_tuple(%29#1) : (i32) -> !cute.int_tuple<"?">
            %ptr_28 = cute.add_offset(%ptr_6, %int_tuple_27) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %32 = builtin.unrealized_conversion_cast %ptr_28 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %32, %29#2, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            %33 = arith.addi %29#1, %c1_i32 : i32
            %34 = arith.cmpi eq, %33, %c6_i32 : i32
            %35 = arith.select %34, %c0_i32, %33 : i32
            %36 = scf.if %34 -> (i32) {
              %58 = arith.xori %29#2, %c1_i32 : i32
              scf.yield %58 : i32
            } else {
              scf.yield %29#2 : i32
            }
            %int_tuple_29 = cute.make_int_tuple(%35) : (i32) -> !cute.int_tuple<"?">
            %ptr_30 = cute.add_offset(%ptr_6, %int_tuple_29) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %37 = builtin.unrealized_conversion_cast %ptr_30 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %37, %36, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            %38 = arith.addi %35, %c1_i32 : i32
            %39 = arith.cmpi eq, %38, %c6_i32 : i32
            %40 = arith.select %39, %c0_i32, %38 : i32
            %41 = scf.if %39 -> (i32) {
              %58 = arith.xori %36, %c1_i32 : i32
              scf.yield %58 : i32
            } else {
              scf.yield %36 : i32
            }
            %int_tuple_31 = cute.make_int_tuple(%40) : (i32) -> !cute.int_tuple<"?">
            %ptr_32 = cute.add_offset(%ptr_6, %int_tuple_31) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %42 = builtin.unrealized_conversion_cast %ptr_32 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %42, %41, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            %43 = arith.addi %40, %c1_i32 : i32
            %44 = arith.cmpi eq, %43, %c6_i32 : i32
            %45 = arith.select %44, %c0_i32, %43 : i32
            %46 = scf.if %44 -> (i32) {
              %58 = arith.xori %41, %c1_i32 : i32
              scf.yield %58 : i32
            } else {
              scf.yield %41 : i32
            }
            %int_tuple_33 = cute.make_int_tuple(%45) : (i32) -> !cute.int_tuple<"?">
            %ptr_34 = cute.add_offset(%ptr_6, %int_tuple_33) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %47 = builtin.unrealized_conversion_cast %ptr_34 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %47, %46, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            %48 = arith.addi %45, %c1_i32 : i32
            %49 = arith.cmpi eq, %48, %c6_i32 : i32
            %50 = arith.select %49, %c0_i32, %48 : i32
            %51 = scf.if %49 -> (i32) {
              %58 = arith.xori %46, %c1_i32 : i32
              scf.yield %58 : i32
            } else {
              scf.yield %46 : i32
            }
            %int_tuple_35 = cute.make_int_tuple(%50) : (i32) -> !cute.int_tuple<"?">
            %ptr_36 = cute.add_offset(%ptr_6, %int_tuple_35) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %52 = builtin.unrealized_conversion_cast %ptr_36 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %52, %51, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            %53 = arith.addi %50, %c1_i32 : i32
            %54 = arith.cmpi eq, %53, %c6_i32 : i32
            %55 = arith.select %54, %c0_i32, %53 : i32
            %56 = scf.if %54 -> (i32) {
              %58 = arith.xori %51, %c1_i32 : i32
              scf.yield %58 : i32
            } else {
              scf.yield %51 : i32
            }
            %int_tuple_37 = cute.make_int_tuple(%55) : (i32) -> !cute.int_tuple<"?">
            %ptr_38 = cute.add_offset(%ptr_6, %int_tuple_37) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %57 = builtin.unrealized_conversion_cast %ptr_38 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %57, %56, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          }
          %30 = arith.xori %29#11, %c1_i32 : i32
          %int_tuple_25 = cute.make_int_tuple(%29#10) : (i32) -> !cute.int_tuple<"?">
          %ptr_26 = cute.add_offset(%ptr_11, %int_tuple_25) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %31 = builtin.unrealized_conversion_cast %ptr_26 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity %31, %30, %c10000000_i32 : !llvm.ptr<3>, i32, i32
        }
      }
      %22 = arith.cmpi slt, %12, %c8_i32 : i32
      scf.if %22 {
        nvvm.setmaxregister increase 232
        %23 = arith.extui %14 : i1 to i32
        %24 = arith.select %14, %23, %c0_i32 : i32
        %25 = arith.cmpi ne, %24, %c0_i32 : i32
        %26 = arith.floordivsi %0, %c128_i32 : i32
        %27 = cute_nvgpu.arch.make_warp_uniform(%26) : i32
        %shape_21 = cute.make_shape() : () -> !cute.shape<"2">
        %stride = cute.make_stride() : () -> !cute.stride<"128">
        %lay_22 = cute.make_layout(%shape_21, %stride) : !cute.layout<"2:128">
        %coord = cute.make_coord(%27) : (i32) -> !cute.coord<"?">
        %idx = cute.crd2idx(%coord, %lay_22) : (!cute.coord<"?">, !cute.layout<"2:128">) -> !cute.int_tuple<"?{div=128}">
        %e0_23 = cute.get_leaves(%idx) : !cute.int_tuple<"?{div=128}">
        %rmem = cute.memref.alloca() : !memref_rmem_f32
        %coord_24 = cute.make_coord(%e0_23) : (!cute.int_tuple<"?{div=128}">) -> !cute.coord<"?{div=128}">
        %ptn_A = cute.tiled.mma.partition A (%arg0, %view_16, %coord_24) : (!mma_bf16_bf16_f32_64x128x16, !memref_smem_bf16, !cute.coord<"?{div=128}">) -> !memref_smem_bf16_6
        %frg_A = cute.mma.make_fragment A (%arg0, %ptn_A) : (!mma_bf16_bf16_f32_64x128x16, !memref_smem_bf16_6) -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">
        %ptn_B = cute.tiled.mma.partition B (%arg0, %view_18, %coord_24) : (!mma_bf16_bf16_f32_64x128x16, !memref_smem_bf16, !cute.coord<"?{div=128}">) -> !memref_smem_bf16_7
        %frg_B = cute.mma.make_fragment B (%arg0, %ptn_B) : (!mma_bf16_bf16_f32_64x128x16, !memref_smem_bf16_7) -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">
        %28 = nvvm.read.ptx.sreg.ctaid.z : i32
        %int_tuple_25 = cute.make_int_tuple(%arg11, %arg12, %arg13) : (i32, i32, i32) -> !cute.int_tuple<"(?,?,?)">
        %sz_26 = cute.size(%int_tuple_25) : (!cute.int_tuple<"(?,?,?)">) -> !cute.int_tuple<"?">
        %e0_27 = cute.get_leaves(%sz_26) : !cute.int_tuple<"?">
        %int_tuple_28 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
        %mul = cute.tuple_mul(%e0_27, %int_tuple_28) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?{div=2}">
        %29 = cute.get_scalars(%mul) : !cute.int_tuple<"?{div=2}">
        %30 = arith.cmpi sgt, %29, %28 : i32
        %31:4 = scf.if %30 -> (i32, i32, i32, i32) {
          %33 = arith.floordivsi %28, %c2_i32 : i32
          %34 = arith.muli %33, %c2_i32 : i32
          %35 = arith.subi %28, %34 : i32
          %36 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %33, %arg15 : (i32, i32) -> i32
          %37 = arith.shrui %36, %arg16 : i32
          %38 = arith.cmpi eq, %arg15, %c0_i32 : i32
          %39 = arith.select %38, %33, %37 : i32
          %40 = arith.muli %39, %arg17 : i32
          %41 = arith.subi %33, %40 : i32
          %42 = arith.muli %39, %c2_i32 : i32
          %43 = arith.addi %42, %35 : i32
          %44 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %41, %arg25 : (i32, i32) -> i32
          %45 = arith.shrui %44, %arg26 : i32
          %46 = arith.cmpi eq, %arg25, %c0_i32 : i32
          %47 = arith.select %46, %41, %45 : i32
          %48 = arith.muli %47, %arg27 : i32
          %49 = arith.subi %41, %48 : i32
          %50 = arith.cmpi slt, %47, %arg18 : i32
          %51:2 = scf.if %50 -> (i32, i32) {
            %63 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %49, %arg19 : (i32, i32) -> i32
            %64 = arith.shrui %63, %arg20 : i32
            %65 = arith.cmpi eq, %arg19, %c0_i32 : i32
            %66 = arith.select %65, %49, %64 : i32
            %67 = arith.muli %66, %arg21 : i32
            %68 = arith.subi %49, %67 : i32
            scf.yield %66, %68 : i32, i32
          } else {
            %63 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %49, %arg22 : (i32, i32) -> i32
            %64 = arith.shrui %63, %arg23 : i32
            %65 = arith.cmpi eq, %arg22, %c0_i32 : i32
            %66 = arith.select %65, %49, %64 : i32
            %67 = arith.muli %66, %arg24 : i32
            %68 = arith.subi %49, %67 : i32
            scf.yield %66, %68 : i32, i32
          }
          %52 = arith.remsi %47, %c2_i32 : i32
          %53 = arith.cmpi eq, %52, %c1_i32 : i32
          %54 = scf.if %53 -> (i32) {
            %63 = arith.cmpi eq, %arg14, %c0_i32 : i32
            %64 = arith.select %63, %arg12, %arg11 : i32
            %65 = arith.subi %64, %c1_i32 : i32
            %66 = arith.subi %65, %51#0 : i32
            scf.yield %66 : i32
          } else {
            scf.yield %51#0 : i32
          }
          %55 = arith.muli %47, %arg21 : i32
          %56 = arith.addi %55, %51#1 : i32
          %57 = arith.cmpi eq, %arg14, %c1_i32 : i32
          %58 = arith.select %57, %54, %56 : i32
          %59 = arith.select %57, %56, %54 : i32
          %60 = arith.floordivsi %43, %c2_i32 : i32
          %61 = arith.muli %60, %c2_i32 : i32
          %62 = arith.subi %43, %61 : i32
          scf.yield %62, %58, %59, %60 : i32, i32, i32, i32
        } else {
          scf.yield %c0_i32, %c0_i32, %c0_i32, %c0_i32 : i32, i32, i32, i32
        }
        %32:11 = scf.while (%arg28 = %c0_i32, %arg29 = %c0_i32, %arg30 = %c0_i32, %arg31 = %31#1, %arg32 = %31#2, %arg33 = %31#0, %arg34 = %31#3, %arg35 = %30, %arg36 = %c0_i32, %arg37 = %c0_i32, %arg38 = %c0_i32, %arg39 = %c0_i32) : (i32, i32, i32, i32, i32, i32, i32, i1, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
          scf.condition(%arg35) %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg36, %arg37, %arg38, %arg39 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
        } do {
        ^bb0(%arg28: i32, %arg29: i32, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32):
          %int_tuple_29 = cute.make_int_tuple(%20) : (i32) -> !cute.int_tuple<"?">
          %tile = cute.make_tile() : () -> !cute.tile<"64:1">
          %shp = cute.ceil_div(%int_tuple_29, %tile) : !cute.int_tuple<"?">, !cute.tile<"64:1">
          %e0_30 = cute.get_leaves(%shp) : !cute.int_tuple<"?">
          %div = cute.tuple_div(%e0_30, %int_tuple_28) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?">
          %mul_31 = cute.tuple_mul(%div, %int_tuple_28) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?{div=2}">
          %sub = cute.tuple_sub(%e0_30, %mul_31) : (!cute.int_tuple<"?">, !cute.int_tuple<"?{div=2}">) -> !cute.int_tuple<"?">
          %33 = cute.get_scalars(%sub) : !cute.int_tuple<"?">
          %34 = arith.cmpi sgt, %33, %arg33 : i32
          %35 = arith.extui %34 : i1 to i32
          %int_tuple_32 = cute.make_int_tuple(%35) : (i32) -> !cute.int_tuple<"?">
          %add = cute.tuple_add(%div, %int_tuple_32) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
          %36 = cute.get_scalars(%add) : !cute.int_tuple<"?">
          iket.range_push("mma")
          %37 = arith.minsi %36, %c1_i32 : i32
          %38 = arith.cmpi sgt, %36, %c0_i32 : i32
          %39 = scf.if %38 -> (i1) {
            %int_tuple_161 = cute.make_int_tuple(%arg29) : (i32) -> !cute.int_tuple<"?">
            %ptr_162 = cute.add_offset(%iter_4, %int_tuple_161) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %99 = builtin.unrealized_conversion_cast %ptr_162 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            %100 = nvvm.mbarrier.wait.parity %99, %arg30 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
            scf.yield %100 : i1
          } else {
            scf.yield %true : i1
          }
          %40:5 = scf.for %arg39 = %c0_i32 to %37 step %c1_i32 iter_args(%arg40 = %true, %arg41 = %39, %arg42 = %arg28, %arg43 = %arg29, %arg44 = %arg30) -> (i1, i1, i32, i32, i32)  : i32 {
            %99 = arith.extui %arg41 : i1 to i32
            %100 = arith.cmpi eq, %99, %c0_i32 : i32
            scf.if %100 {
              %int_tuple_177 = cute.make_int_tuple(%arg43) : (i32) -> !cute.int_tuple<"?">
              %ptr_178 = cute.add_offset(%iter_4, %int_tuple_177) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %114 = builtin.unrealized_conversion_cast %ptr_178 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.mbarrier.try_wait.parity %114, %arg44, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            }
            %coord_161 = cute.make_coord(%arg43) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_162 = cute.slice(%frg_A, %coord_161) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">, !cute.coord<"(_,_,_,?)">
            %slice_163 = cute.slice(%frg_B, %coord_161) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">, !cute.coord<"(_,_,_,?)">
            nvvm.wgmma.fence.aligned
            %atom_164 = cute.make_atom(%false) : (i1) -> !cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>
            %101 = arith.cmpi eq, %arg40, %false : i1
            %102 = cute_nvgpu.atom.set_value<accum_c>(%atom_164, %101) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_165 = cute.make_coord() : () -> !cute.coord<"(_,_,0)">
            %slice_166 = cute.slice(%slice_162, %coord_165) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,0)">
            %slice_167 = cute.slice(%slice_163, %coord_165) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,0)">
            cute.gemm(%102, %rmem, %slice_166, %slice_167, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %103 = cute_nvgpu.atom.set_value<accum_c>(%102, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_168 = cute.make_coord() : () -> !cute.coord<"(_,_,1)">
            %slice_169 = cute.slice(%slice_162, %coord_168) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,1)">
            %slice_170 = cute.slice(%slice_163, %coord_168) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,1)">
            cute.gemm(%103, %rmem, %slice_169, %slice_170, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %104 = cute_nvgpu.atom.set_value<accum_c>(%103, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_171 = cute.make_coord() : () -> !cute.coord<"(_,_,2)">
            %slice_172 = cute.slice(%slice_162, %coord_171) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,2)">
            %slice_173 = cute.slice(%slice_163, %coord_171) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,2)">
            cute.gemm(%104, %rmem, %slice_172, %slice_173, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %105 = cute_nvgpu.atom.set_value<accum_c>(%104, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_174 = cute.make_coord() : () -> !cute.coord<"(_,_,3)">
            %slice_175 = cute.slice(%slice_162, %coord_174) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,3)">
            %slice_176 = cute.slice(%slice_163, %coord_174) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,3)">
            cute.gemm(%105, %rmem, %slice_175, %slice_176, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            nvvm.wgmma.commit.group.sync.aligned
            %106 = arith.addi %arg43, %c1_i32 : i32
            %107 = arith.addi %arg42, %c1_i32 : i32
            %108 = arith.cmpi eq, %106, %c6_i32 : i32
            %109 = arith.select %108, %c0_i32, %106 : i32
            %110 = scf.if %108 -> (i32) {
              %114 = arith.xori %arg44, %c1_i32 : i32
              scf.yield %114 : i32
            } else {
              scf.yield %arg44 : i32
            }
            %111 = arith.addi %arg39, %c1_i32 : i32
            %112 = arith.cmpi sgt, %36, %111 : i32
            %113 = scf.if %112 -> (i1) {
              %int_tuple_177 = cute.make_int_tuple(%109) : (i32) -> !cute.int_tuple<"?">
              %ptr_178 = cute.add_offset(%iter_4, %int_tuple_177) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %114 = builtin.unrealized_conversion_cast %ptr_178 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %115 = nvvm.mbarrier.wait.parity %114, %110 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
              scf.yield %115 : i1
            } else {
              scf.yield %true : i1
            }
            scf.yield %false, %113, %107, %109, %110 : i1, i1, i32, i32, i32
          }
          %41:8 = scf.for %arg39 = %37 to %36 step %c1_i32 iter_args(%arg40 = %40#0, %arg41 = %40#1, %arg42 = %40#2, %arg43 = %40#3, %arg44 = %40#4, %arg45 = %arg28, %arg46 = %arg29, %arg47 = %arg30) -> (i1, i1, i32, i32, i32, i32, i32, i32)  : i32 {
            %99 = arith.extui %arg41 : i1 to i32
            %100 = arith.cmpi eq, %99, %c0_i32 : i32
            scf.if %100 {
              %int_tuple_177 = cute.make_int_tuple(%arg43) : (i32) -> !cute.int_tuple<"?">
              %ptr_178 = cute.add_offset(%iter_4, %int_tuple_177) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %119 = builtin.unrealized_conversion_cast %ptr_178 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.mbarrier.try_wait.parity %119, %arg44, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            }
            %coord_161 = cute.make_coord(%arg43) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_162 = cute.slice(%frg_A, %coord_161) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">, !cute.coord<"(_,_,_,?)">
            %slice_163 = cute.slice(%frg_B, %coord_161) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">, !cute.coord<"(_,_,_,?)">
            nvvm.wgmma.fence.aligned
            %atom_164 = cute.make_atom(%false) : (i1) -> !cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>
            %101 = arith.cmpi eq, %arg40, %false : i1
            %102 = cute_nvgpu.atom.set_value<accum_c>(%atom_164, %101) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_165 = cute.make_coord() : () -> !cute.coord<"(_,_,0)">
            %slice_166 = cute.slice(%slice_162, %coord_165) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,0)">
            %slice_167 = cute.slice(%slice_163, %coord_165) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,0)">
            cute.gemm(%102, %rmem, %slice_166, %slice_167, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %103 = cute_nvgpu.atom.set_value<accum_c>(%102, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_168 = cute.make_coord() : () -> !cute.coord<"(_,_,1)">
            %slice_169 = cute.slice(%slice_162, %coord_168) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,1)">
            %slice_170 = cute.slice(%slice_163, %coord_168) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,1)">
            cute.gemm(%103, %rmem, %slice_169, %slice_170, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %104 = cute_nvgpu.atom.set_value<accum_c>(%103, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_171 = cute.make_coord() : () -> !cute.coord<"(_,_,2)">
            %slice_172 = cute.slice(%slice_162, %coord_171) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,2)">
            %slice_173 = cute.slice(%slice_163, %coord_171) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,2)">
            cute.gemm(%104, %rmem, %slice_172, %slice_173, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %105 = cute_nvgpu.atom.set_value<accum_c>(%104, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_174 = cute.make_coord() : () -> !cute.coord<"(_,_,3)">
            %slice_175 = cute.slice(%slice_162, %coord_174) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,3)">
            %slice_176 = cute.slice(%slice_163, %coord_174) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,3)">
            cute.gemm(%105, %rmem, %slice_175, %slice_176, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            nvvm.wgmma.commit.group.sync.aligned
            nvvm.wgmma.wait.group.sync.aligned 1
            scf.if %16 {
              %int_tuple_177 = cute.make_int_tuple(%arg46) : (i32) -> !cute.int_tuple<"?">
              %ptr_178 = cute.add_offset(%ptr_6, %int_tuple_177) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %119 = builtin.unrealized_conversion_cast %ptr_178 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %120 = nvvm.mbarrier.arrive %119, %c1_i32 : !llvm.ptr<3> -> i64
            }
            %106 = arith.addi %arg43, %c1_i32 : i32
            %107 = arith.addi %arg42, %c1_i32 : i32
            %108 = arith.cmpi eq, %106, %c6_i32 : i32
            %109 = arith.select %108, %c0_i32, %106 : i32
            %110 = scf.if %108 -> (i32) {
              %119 = arith.xori %arg44, %c1_i32 : i32
              scf.yield %119 : i32
            } else {
              scf.yield %arg44 : i32
            }
            %111 = arith.addi %arg46, %c1_i32 : i32
            %112 = arith.addi %arg45, %c1_i32 : i32
            %113 = arith.cmpi eq, %111, %c6_i32 : i32
            %114 = arith.select %113, %c0_i32, %111 : i32
            %115 = scf.if %113 -> (i32) {
              %119 = arith.xori %arg47, %c1_i32 : i32
              scf.yield %119 : i32
            } else {
              scf.yield %arg47 : i32
            }
            %116 = arith.addi %arg39, %c1_i32 : i32
            %117 = arith.cmpi sgt, %36, %116 : i32
            %118 = scf.if %117 -> (i1) {
              %int_tuple_177 = cute.make_int_tuple(%109) : (i32) -> !cute.int_tuple<"?">
              %ptr_178 = cute.add_offset(%iter_4, %int_tuple_177) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %119 = builtin.unrealized_conversion_cast %ptr_178 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %120 = nvvm.mbarrier.wait.parity %119, %110 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
              scf.yield %120 : i1
            } else {
              scf.yield %true : i1
            }
            scf.yield %false, %118, %107, %109, %110, %112, %114, %115 : i1, i1, i32, i32, i32, i32, i32, i32
          } {loop_annotation = #loop_annotation}
          nvvm.wgmma.wait.group.sync.aligned 0
          %42:3 = scf.for %arg39 = %c0_i32 to %37 step %c1_i32 iter_args(%arg40 = %41#5, %arg41 = %41#6, %arg42 = %41#7) -> (i32, i32, i32)  : i32 {
            scf.if %16 {
              %int_tuple_161 = cute.make_int_tuple(%arg41) : (i32) -> !cute.int_tuple<"?">
              %ptr_162 = cute.add_offset(%ptr_6, %int_tuple_161) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %104 = builtin.unrealized_conversion_cast %ptr_162 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %105 = nvvm.mbarrier.arrive %104, %c1_i32 : !llvm.ptr<3> -> i64
            }
            %99 = arith.addi %arg41, %c1_i32 : i32
            %100 = arith.addi %arg40, %c1_i32 : i32
            %101 = arith.cmpi eq, %99, %c6_i32 : i32
            %102 = arith.select %101, %c0_i32, %99 : i32
            %103 = scf.if %101 -> (i32) {
              %104 = arith.xori %arg42, %c1_i32 : i32
              scf.yield %104 : i32
            } else {
              scf.yield %arg42 : i32
            }
            scf.yield %100, %102, %103 : i32, i32, i32
          } {loop_annotation = #loop_annotation}
          %43 = arith.cmpi eq, %36, %c0_i32 : i32
          scf.if %43 {
            cute.memref.store_vec(%cst, %rmem) : (vector<64xf32>, !memref_rmem_f32) -> ()
          }
          %iter_33 = cute.get_iter(%rmem) : !memref_rmem_f32
          iket.range_pop
          iket.range_push("epilogue")
          %44 = arith.muli %arg34, %c2_i32 : i32
          %45 = arith.addi %44, %arg33 : i32
          %coord_34 = cute.make_coord(%45) : (i32) -> !cute.coord<"(_,_,?)">
          %slice = cute.slice(%arg6, %coord_34) : !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute.coord<"(_,_,?)">
          %tile_35 = cute.make_tile() : () -> !cute.tile<"[128:1;128:1]">
          %coord_36 = cute.make_coord(%arg31, %arg32) : (i32, i32) -> !cute.coord<"(?,?)">
          %tiled_view = cute.local_tile(%slice, %tile_35, %coord_36) : (!cute.coord_tensor<"(0,0,?)", "(?,?):(1@1,1@0)">, !cute.tile<"[128:1;128:1]">, !cute.coord<"(?,?)">) -> !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(128,128):(1@1,1@0)">
          %tile_37 = cute.make_tile() : () -> !cute.tile<"[128:1;32:1]">
          %div_38 = cute.zipped_divide(%tiled_view, %tile_37) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(128,128):(1@1,1@0)">, !cute.tile<"[128:1;32:1]">
          %shape_39 = cute.make_shape() : () -> !cute.shape<"1">
          %lay_40 = cute.make_layout(%shape_39) : !cute.layout<"1:0">
          %grouped = cute.group_modes(%view_20) <0, 2> : (!memref_smem_f32) -> !memref_smem_f32_1
          %grouped_41 = cute.group_modes(%div_38) <0, 1> : (!cute.coord_tensor<"(?{div=128},?{div=128},?)", "((128,32),(1,4)):((1@1,1@0),(0,32@0))">) -> !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((128,32)),(1,4)):(((1@1,1@0)),(0,32@0))">
          %coord_42 = cute.make_coord() : () -> !cute.coord<"0">
          %res_smem_tensor, %res_target_tensors = cute_nvgpu.atom.tma_partition(%arg5, %coord_42, %lay_40, %grouped, %grouped_41) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f32, copy_bits = 131072, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = F32_RN>, !cute.coord<"0">, !cute.layout<"1:0">, !memref_smem_f32_1, !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((128,32)),(1,4)):(((1@1,1@0)),(0,32@0))">) -> (!memref_smem_f32_2, !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1,4)):(((1@0,1@1),0),(0,32@0))">)
          %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32>
          %46 = cute.make_tiled_copy(%atom) : !copy_simt
          %coord_43 = cute.make_coord(%0) : (i32) -> !cute.coord<"?">
          %dst_partitioned = cute.tiled.copy.partition_D(%46, %view_20, %coord_43) : (!copy_simt, !memref_smem_f32, !cute.coord<"?">) -> !memref_smem_f32_3
          %rmem_44 = cute.memref.alloca() : !memref_rmem_f32_1
          %iter_45 = cute.get_iter(%rmem_44) : !memref_rmem_f32_1
          %shape_46 = cute.make_shape() : () -> !cute.shape<"((2,2,2),1,(8,1))">
          %stride_47 = cute.make_stride() : () -> !cute.stride<"((1,2,4),0,(8,0))">
          %lay_48 = cute.make_layout(%shape_46, %stride_47) : !cute.layout<"((2,2,2),1,(8,1)):((1,2,4),0,(8,0))">
          %view_49 = cute.make_view(%iter_33, %lay_48) : !memref_rmem_f32_2
          %tile_50 = cute.make_tile() : () -> !cute.tile<"[[2:1;2:1;2:1];1:0;2:1]">
          %div_51 = cute.flat_divide(%view_49, %tile_50) : !memref_rmem_f32_2, !cute.tile<"[[2:1;2:1;2:1];1:0;2:1]">
          %coord_52 = cute.make_coord() : () -> !cute.coord<"(_,_,_,0,_,_)">
          %slice_53 = cute.slice(%div_51, %coord_52) : !memref_rmem_f32_3, !cute.coord<"(_,_,_,0,_,_)">
          %grouped_54 = cute.group_modes(%slice_53) <3, 5> : (!memref_rmem_f32_4) -> !memref_rmem_f32_5
          %retiled = cute.tiled.copy.retile(%46, %grouped_54) : (!copy_simt, !memref_rmem_f32_5) -> !memref_rmem_f32_6
          %47 = arith.muli %arg35, %c4_i32 : i32
          %lay_55 = cute.get_layout(%rmem_44) : !memref_rmem_f32_1
          %coord_56 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,0))">
          %slice_57 = cute.slice(%retiled, %coord_56) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,0))">
          %lay_58 = cute.get_layout(%slice_57) : !memref_rmem_f32_7
          %rinv = cute.right_inverse(%lay_55) : (!cute.layout<"((1,(2,2,2)),1,2):((0,(1,2,4)),0,8)">) -> !cute.layout<"16:1">
          %48 = cute.composition(%lay_58, %rinv) : (!cute.layout<"((1,8),1,2):((0,1),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
          %coalesce = cute.coalesce(%48) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
          %49 = cute.composition(%rinv, %coalesce) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
          %div_59 = cute.logical_divide(%slice_57, %49) : !memref_rmem_f32_7, !cute.layout<"16:1">
          %div_60 = cute.logical_divide(%rmem_44, %49) : !memref_rmem_f32_1, !cute.layout<"16:1">
          %shape_61 = cute.make_shape() : () -> !cute.shape<"8">
          %lay_62 = cute.make_layout(%shape_61) : !cute.layout<"8:1">
          %div_63 = cute.logical_divide(%div_59, %lay_62) : !memref_rmem_f32_8, !cute.layout<"8:1">
          %div_64 = cute.logical_divide(%div_60, %lay_62) : !memref_rmem_f32_8, !cute.layout<"8:1">
          %atom_65 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 256 b>
          cute.copy(%atom_65, %div_63, %div_64) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
          iket.range_push("epi_math")
          iket.range_pop
          iket.range_push("epi_store_acq")
          scf.if %25 {
            nvvm.cp.async.bulk.wait_group 1 {read}
          }
          nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
          iket.range_pop
          %50 = arith.remsi %47, %c2_i32 : i32
          iket.range_push("epi_r2s")
          %coord_66 = cute.make_coord(%50) : (i32) -> !cute.coord<"(_,_,_,?)">
          %slice_67 = cute.slice(%dst_partitioned, %coord_66) : !memref_smem_f32_3, !cute.coord<"(_,_,_,?)">
          %iter_68 = cute.get_iter(%slice_67) : !memref_smem_f32_4
          %shape_69 = cute.make_shape() : () -> !cute.shape<"(1,2,2,2,1,2)">
          %stride_70 = cute.make_stride() : () -> !cute.stride<"(0,1,2,4,0,8)">
          %lay_71 = cute.make_layout(%shape_69, %stride_70) : !cute.layout<"(1,2,2,2,1,2):(0,1,2,4,0,8)">
          %view_72 = cute.make_view(%iter_45, %lay_71) : !memref_rmem_f32_10
          %stride_73 = cute.make_stride() : () -> !cute.stride<"(0,1,256,8,0,16)">
          %lay_74 = cute.make_layout(%shape_69, %stride_73) : !cute.layout<"(1,2,2,2,1,2):(0,1,256,8,0,16)">
          %view_75 = cute.make_view(%iter_68, %lay_74) : !memref_smem_f32_5
          %coord_76 = cute.make_coord() : () -> !cute.coord<"(_,0,_,_,_,_)">
          %slice_77 = cute.slice(%view_72, %coord_76) : !memref_rmem_f32_10, !cute.coord<"(_,0,_,_,_,_)">
          %51 = cute.memref.load_vec(%slice_77) : (!memref_rmem_f32_11) -> vector<8xf32>
          %coord_78 = cute.make_coord() : () -> !cute.coord<"(_,1,_,_,_,_)">
          %slice_79 = cute.slice(%view_72, %coord_78) : !memref_rmem_f32_10, !cute.coord<"(_,1,_,_,_,_)">
          %52 = cute.memref.load_vec(%slice_79) : (!memref_rmem_f32_12) -> vector<8xf32>
          %53 = arith.shrsi %0, %c1_i32 : i32
          %54 = arith.andi %53, %c1_i32 : i32
          %55 = arith.cmpi ne, %54, %c0_i32 : i32
          %rmem_80 = cute.memref.alloca() : !memref_rmem_f32_13
          %rmem_81 = cute.memref.alloca() : !memref_rmem_f32_13
          %56 = arith.select %55, %52, %51 : vector<8xf32>
          cute.memref.store_vec(%56, %rmem_80) : (vector<8xf32>, !memref_rmem_f32_13) -> ()
          %57 = arith.select %55, %51, %52 : vector<8xf32>
          cute.memref.store_vec(%57, %rmem_81) : (vector<8xf32>, !memref_rmem_f32_13) -> ()
          %slice_82 = cute.slice(%view_75, %coord_76) : !memref_smem_f32_5, !cute.coord<"(_,0,_,_,_,_)">
          %iter_83 = cute.get_iter(%slice_82) : !memref_smem_f32_6
          %int_tuple_84 = cute.make_int_tuple(%54) : (i32) -> !cute.int_tuple<"?">
          %ptr_85 = cute.add_offset(%iter_83, %int_tuple_84) : (!cute.ptr<f32, smem, align<8>, S<3,4,3>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, smem, S<3,4,3>>
          %lay_86 = cute.get_layout(%slice_82) : !memref_smem_f32_6
          %view_87 = cute.make_view(%ptr_85, %lay_86) : !memref_smem_f32_7
          %atom_88 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
          cute.copy(%atom_88, %rmem_80, %view_87) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !memref_rmem_f32_13, !memref_smem_f32_7)
          %58 = arith.subi %c1_i32, %54 : i32
          %int_tuple_89 = cute.make_int_tuple(%58) : (i32) -> !cute.int_tuple<"?">
          %ptr_90 = cute.add_offset(%iter_83, %int_tuple_89) : (!cute.ptr<f32, smem, align<8>, S<3,4,3>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, smem, S<3,4,3>>
          %view_91 = cute.make_view(%ptr_90, %lay_86) : !memref_smem_f32_7
          cute.copy(%atom_88, %rmem_81, %view_91) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !memref_rmem_f32_13, !memref_smem_f32_7)
          iket.range_pop
          iket.range_push("epi_store_issue")
          nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
          nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
          scf.if %25 {
            %coord_161 = cute.make_coord(%50) : (i32) -> !cute.coord<"(_,?)">
            %slice_162 = cute.slice(%res_smem_tensor, %coord_161) : !memref_smem_f32_2, !cute.coord<"(_,?)">
            %iter_163 = cute.get_iter(%slice_162) : !memref_smem_f32_8
            %coord_164 = cute.make_coord() : () -> !cute.coord<"(_,(0,0))">
            %slice_165 = cute.slice(%res_target_tensors, %coord_164) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1,4)):(((1@0,1@1),0),(0,32@0))">, !cute.coord<"(_,(0,0))">
            %iter_166 = cute.get_iter(%slice_165) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
            %tup = cute.deref_arith_tuple_iter(%iter_166) : !cute.arith_tuple_iter<"(?{div=128},?{div=128},?)">
            %e0_167, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=128},?{div=128},?)">
            %lay_168 = cute.get_layout(%slice_162) : !memref_smem_f32_8
            %lay_169 = cute.get_layout(%slice_165) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
            %append = cute.append_to_rank<2> (%lay_168, %lay_40) : !cute.layout<"((4096,1)):((1,0))">, !cute.layout<"1:0">
            %view_170 = cute.make_view(%iter_163, %append) : !memref_smem_f32_9
            %grouped_171 = cute.group_modes(%view_170) <1, 2> : (!memref_smem_f32_9) -> !memref_smem_f32_10
            %append_172 = cute.append_to_rank<2> (%lay_169, %lay_40) : !cute.layout<"(((32,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
            %int_tuple_173 = cute.make_int_tuple(%e0_167, %e1, %e2) : (!cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=128},?{div=128},?)">
            %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_173) : (!cute.int_tuple<"(?{div=128},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=128},?{div=128},?)">
            %view_174 = cute.make_view(%int_tup_iter, %append_172) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">
            %grouped_175 = cute.group_modes(%view_174) <1, 2> : (!cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">
            %99 = cute_nvgpu.atom.make_exec_tma(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f32, copy_bits = 131072, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = F32_RN>) -> !cute_nvgpu.atom.tma_store<f32, copy_bits = 131072, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>
            cute.copy(%99, %grouped_171, %grouped_175) : (!cute_nvgpu.atom.tma_store<f32, copy_bits = 131072, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>, !memref_smem_f32_10, !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">)
            nvvm.cp.async.bulk.commit.group
          }
          iket.range_pop
          %coord_92 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,1))">
          %slice_93 = cute.slice(%retiled, %coord_92) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,1))">
          %lay_94 = cute.get_layout(%slice_93) : !memref_rmem_f32_7
          %59 = cute.composition(%lay_94, %rinv) : (!cute.layout<"((1,8),1,2):((0,1),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
          %coalesce_95 = cute.coalesce(%59) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
          %60 = cute.composition(%rinv, %coalesce_95) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
          %div_96 = cute.logical_divide(%slice_93, %60) : !memref_rmem_f32_7, !cute.layout<"16:1">
          %div_97 = cute.logical_divide(%rmem_44, %60) : !memref_rmem_f32_1, !cute.layout<"16:1">
          %div_98 = cute.logical_divide(%div_96, %lay_62) : !memref_rmem_f32_8, !cute.layout<"8:1">
          %div_99 = cute.logical_divide(%div_97, %lay_62) : !memref_rmem_f32_8, !cute.layout<"8:1">
          cute.copy(%atom_65, %div_98, %div_99) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
          iket.range_push("epi_math")
          iket.range_pop
          iket.range_push("epi_store_acq")
          scf.if %25 {
            nvvm.cp.async.bulk.wait_group 1 {read}
          }
          nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
          iket.range_pop
          %61 = arith.addi %47, %c1_i32 : i32
          %62 = arith.remsi %61, %c2_i32 : i32
          iket.range_push("epi_r2s")
          %coord_100 = cute.make_coord(%62) : (i32) -> !cute.coord<"(_,_,_,?)">
          %slice_101 = cute.slice(%dst_partitioned, %coord_100) : !memref_smem_f32_3, !cute.coord<"(_,_,_,?)">
          %iter_102 = cute.get_iter(%slice_101) : !memref_smem_f32_4
          %view_103 = cute.make_view(%iter_102, %lay_74) : !memref_smem_f32_5
          %63 = cute.memref.load_vec(%slice_77) : (!memref_rmem_f32_11) -> vector<8xf32>
          %64 = cute.memref.load_vec(%slice_79) : (!memref_rmem_f32_12) -> vector<8xf32>
          %rmem_104 = cute.memref.alloca() : !memref_rmem_f32_13
          %rmem_105 = cute.memref.alloca() : !memref_rmem_f32_13
          %65 = arith.select %55, %64, %63 : vector<8xf32>
          cute.memref.store_vec(%65, %rmem_104) : (vector<8xf32>, !memref_rmem_f32_13) -> ()
          %66 = arith.select %55, %63, %64 : vector<8xf32>
          cute.memref.store_vec(%66, %rmem_105) : (vector<8xf32>, !memref_rmem_f32_13) -> ()
          %slice_106 = cute.slice(%view_103, %coord_76) : !memref_smem_f32_5, !cute.coord<"(_,0,_,_,_,_)">
          %iter_107 = cute.get_iter(%slice_106) : !memref_smem_f32_6
          %ptr_108 = cute.add_offset(%iter_107, %int_tuple_84) : (!cute.ptr<f32, smem, align<8>, S<3,4,3>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, smem, S<3,4,3>>
          %lay_109 = cute.get_layout(%slice_106) : !memref_smem_f32_6
          %view_110 = cute.make_view(%ptr_108, %lay_109) : !memref_smem_f32_7
          cute.copy(%atom_88, %rmem_104, %view_110) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !memref_rmem_f32_13, !memref_smem_f32_7)
          %ptr_111 = cute.add_offset(%iter_107, %int_tuple_89) : (!cute.ptr<f32, smem, align<8>, S<3,4,3>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, smem, S<3,4,3>>
          %view_112 = cute.make_view(%ptr_111, %lay_109) : !memref_smem_f32_7
          cute.copy(%atom_88, %rmem_105, %view_112) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !memref_rmem_f32_13, !memref_smem_f32_7)
          iket.range_pop
          iket.range_push("epi_store_issue")
          nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
          nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
          scf.if %25 {
            %coord_161 = cute.make_coord(%62) : (i32) -> !cute.coord<"(_,?)">
            %slice_162 = cute.slice(%res_smem_tensor, %coord_161) : !memref_smem_f32_2, !cute.coord<"(_,?)">
            %iter_163 = cute.get_iter(%slice_162) : !memref_smem_f32_8
            %coord_164 = cute.make_coord() : () -> !cute.coord<"(_,(0,1))">
            %slice_165 = cute.slice(%res_target_tensors, %coord_164) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1,4)):(((1@0,1@1),0),(0,32@0))">, !cute.coord<"(_,(0,1))">
            %iter_166 = cute.get_iter(%slice_165) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
            %tup = cute.deref_arith_tuple_iter(%iter_166) : !cute.arith_tuple_iter<"(?{div=32},?{div=128},?)">
            %e0_167, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=32},?{div=128},?)">
            %lay_168 = cute.get_layout(%slice_162) : !memref_smem_f32_8
            %lay_169 = cute.get_layout(%slice_165) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
            %append = cute.append_to_rank<2> (%lay_168, %lay_40) : !cute.layout<"((4096,1)):((1,0))">, !cute.layout<"1:0">
            %view_170 = cute.make_view(%iter_163, %append) : !memref_smem_f32_9
            %grouped_171 = cute.group_modes(%view_170) <1, 2> : (!memref_smem_f32_9) -> !memref_smem_f32_10
            %append_172 = cute.append_to_rank<2> (%lay_169, %lay_40) : !cute.layout<"(((32,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
            %int_tuple_173 = cute.make_int_tuple(%e0_167, %e1, %e2) : (!cute.int_tuple<"?{div=32}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=32},?{div=128},?)">
            %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_173) : (!cute.int_tuple<"(?{div=32},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=32},?{div=128},?)">
            %view_174 = cute.make_view(%int_tup_iter, %append_172) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">
            %grouped_175 = cute.group_modes(%view_174) <1, 2> : (!cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">
            %99 = cute_nvgpu.atom.make_exec_tma(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f32, copy_bits = 131072, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = F32_RN>) -> !cute_nvgpu.atom.tma_store<f32, copy_bits = 131072, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>
            cute.copy(%99, %grouped_171, %grouped_175) : (!cute_nvgpu.atom.tma_store<f32, copy_bits = 131072, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>, !memref_smem_f32_10, !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">)
            nvvm.cp.async.bulk.commit.group
          }
          iket.range_pop
          %coord_113 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,2))">
          %slice_114 = cute.slice(%retiled, %coord_113) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,2))">
          %lay_115 = cute.get_layout(%slice_114) : !memref_rmem_f32_7
          %67 = cute.composition(%lay_115, %rinv) : (!cute.layout<"((1,8),1,2):((0,1),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
          %coalesce_116 = cute.coalesce(%67) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
          %68 = cute.composition(%rinv, %coalesce_116) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
          %div_117 = cute.logical_divide(%slice_114, %68) : !memref_rmem_f32_7, !cute.layout<"16:1">
          %div_118 = cute.logical_divide(%rmem_44, %68) : !memref_rmem_f32_1, !cute.layout<"16:1">
          %div_119 = cute.logical_divide(%div_117, %lay_62) : !memref_rmem_f32_8, !cute.layout<"8:1">
          %div_120 = cute.logical_divide(%div_118, %lay_62) : !memref_rmem_f32_8, !cute.layout<"8:1">
          cute.copy(%atom_65, %div_119, %div_120) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
          iket.range_push("epi_math")
          iket.range_pop
          iket.range_push("epi_store_acq")
          scf.if %25 {
            nvvm.cp.async.bulk.wait_group 1 {read}
          }
          nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
          iket.range_pop
          %69 = arith.addi %47, %c2_i32 : i32
          %70 = arith.remsi %69, %c2_i32 : i32
          iket.range_push("epi_r2s")
          %coord_121 = cute.make_coord(%70) : (i32) -> !cute.coord<"(_,_,_,?)">
          %slice_122 = cute.slice(%dst_partitioned, %coord_121) : !memref_smem_f32_3, !cute.coord<"(_,_,_,?)">
          %iter_123 = cute.get_iter(%slice_122) : !memref_smem_f32_4
          %view_124 = cute.make_view(%iter_123, %lay_74) : !memref_smem_f32_5
          %71 = cute.memref.load_vec(%slice_77) : (!memref_rmem_f32_11) -> vector<8xf32>
          %72 = cute.memref.load_vec(%slice_79) : (!memref_rmem_f32_12) -> vector<8xf32>
          %rmem_125 = cute.memref.alloca() : !memref_rmem_f32_13
          %rmem_126 = cute.memref.alloca() : !memref_rmem_f32_13
          %73 = arith.select %55, %72, %71 : vector<8xf32>
          cute.memref.store_vec(%73, %rmem_125) : (vector<8xf32>, !memref_rmem_f32_13) -> ()
          %74 = arith.select %55, %71, %72 : vector<8xf32>
          cute.memref.store_vec(%74, %rmem_126) : (vector<8xf32>, !memref_rmem_f32_13) -> ()
          %slice_127 = cute.slice(%view_124, %coord_76) : !memref_smem_f32_5, !cute.coord<"(_,0,_,_,_,_)">
          %iter_128 = cute.get_iter(%slice_127) : !memref_smem_f32_6
          %ptr_129 = cute.add_offset(%iter_128, %int_tuple_84) : (!cute.ptr<f32, smem, align<8>, S<3,4,3>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, smem, S<3,4,3>>
          %lay_130 = cute.get_layout(%slice_127) : !memref_smem_f32_6
          %view_131 = cute.make_view(%ptr_129, %lay_130) : !memref_smem_f32_7
          cute.copy(%atom_88, %rmem_125, %view_131) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !memref_rmem_f32_13, !memref_smem_f32_7)
          %ptr_132 = cute.add_offset(%iter_128, %int_tuple_89) : (!cute.ptr<f32, smem, align<8>, S<3,4,3>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, smem, S<3,4,3>>
          %view_133 = cute.make_view(%ptr_132, %lay_130) : !memref_smem_f32_7
          cute.copy(%atom_88, %rmem_126, %view_133) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !memref_rmem_f32_13, !memref_smem_f32_7)
          iket.range_pop
          iket.range_push("epi_store_issue")
          nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
          nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
          scf.if %25 {
            %coord_161 = cute.make_coord(%70) : (i32) -> !cute.coord<"(_,?)">
            %slice_162 = cute.slice(%res_smem_tensor, %coord_161) : !memref_smem_f32_2, !cute.coord<"(_,?)">
            %iter_163 = cute.get_iter(%slice_162) : !memref_smem_f32_8
            %coord_164 = cute.make_coord() : () -> !cute.coord<"(_,(0,2))">
            %slice_165 = cute.slice(%res_target_tensors, %coord_164) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1,4)):(((1@0,1@1),0),(0,32@0))">, !cute.coord<"(_,(0,2))">
            %iter_166 = cute.get_iter(%slice_165) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
            %tup = cute.deref_arith_tuple_iter(%iter_166) : !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
            %e0_167, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=64},?{div=128},?)">
            %lay_168 = cute.get_layout(%slice_162) : !memref_smem_f32_8
            %lay_169 = cute.get_layout(%slice_165) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
            %append = cute.append_to_rank<2> (%lay_168, %lay_40) : !cute.layout<"((4096,1)):((1,0))">, !cute.layout<"1:0">
            %view_170 = cute.make_view(%iter_163, %append) : !memref_smem_f32_9
            %grouped_171 = cute.group_modes(%view_170) <1, 2> : (!memref_smem_f32_9) -> !memref_smem_f32_10
            %append_172 = cute.append_to_rank<2> (%lay_169, %lay_40) : !cute.layout<"(((32,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
            %int_tuple_173 = cute.make_int_tuple(%e0_167, %e1, %e2) : (!cute.int_tuple<"?{div=64}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=64},?{div=128},?)">
            %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_173) : (!cute.int_tuple<"(?{div=64},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
            %view_174 = cute.make_view(%int_tup_iter, %append_172) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">
            %grouped_175 = cute.group_modes(%view_174) <1, 2> : (!cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">
            %99 = cute_nvgpu.atom.make_exec_tma(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f32, copy_bits = 131072, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = F32_RN>) -> !cute_nvgpu.atom.tma_store<f32, copy_bits = 131072, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>
            cute.copy(%99, %grouped_171, %grouped_175) : (!cute_nvgpu.atom.tma_store<f32, copy_bits = 131072, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>, !memref_smem_f32_10, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">)
            nvvm.cp.async.bulk.commit.group
          }
          iket.range_pop
          %coord_134 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,3))">
          %slice_135 = cute.slice(%retiled, %coord_134) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,3))">
          %lay_136 = cute.get_layout(%slice_135) : !memref_rmem_f32_7
          %75 = cute.composition(%lay_136, %rinv) : (!cute.layout<"((1,8),1,2):((0,1),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
          %coalesce_137 = cute.coalesce(%75) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
          %76 = cute.composition(%rinv, %coalesce_137) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
          %div_138 = cute.logical_divide(%slice_135, %76) : !memref_rmem_f32_7, !cute.layout<"16:1">
          %div_139 = cute.logical_divide(%rmem_44, %76) : !memref_rmem_f32_1, !cute.layout<"16:1">
          %div_140 = cute.logical_divide(%div_138, %lay_62) : !memref_rmem_f32_8, !cute.layout<"8:1">
          %div_141 = cute.logical_divide(%div_139, %lay_62) : !memref_rmem_f32_8, !cute.layout<"8:1">
          cute.copy(%atom_65, %div_140, %div_141) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
          iket.range_push("epi_math")
          iket.range_pop
          iket.range_push("epi_store_acq")
          scf.if %25 {
            nvvm.cp.async.bulk.wait_group 1 {read}
          }
          nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
          iket.range_pop
          %77 = arith.addi %47, %c3_i32 : i32
          %78 = arith.remsi %77, %c2_i32 : i32
          iket.range_push("epi_r2s")
          %coord_142 = cute.make_coord(%78) : (i32) -> !cute.coord<"(_,_,_,?)">
          %slice_143 = cute.slice(%dst_partitioned, %coord_142) : !memref_smem_f32_3, !cute.coord<"(_,_,_,?)">
          %iter_144 = cute.get_iter(%slice_143) : !memref_smem_f32_4
          %view_145 = cute.make_view(%iter_144, %lay_74) : !memref_smem_f32_5
          %79 = cute.memref.load_vec(%slice_77) : (!memref_rmem_f32_11) -> vector<8xf32>
          %80 = cute.memref.load_vec(%slice_79) : (!memref_rmem_f32_12) -> vector<8xf32>
          %rmem_146 = cute.memref.alloca() : !memref_rmem_f32_13
          %rmem_147 = cute.memref.alloca() : !memref_rmem_f32_13
          %81 = arith.select %55, %80, %79 : vector<8xf32>
          cute.memref.store_vec(%81, %rmem_146) : (vector<8xf32>, !memref_rmem_f32_13) -> ()
          %82 = arith.select %55, %79, %80 : vector<8xf32>
          cute.memref.store_vec(%82, %rmem_147) : (vector<8xf32>, !memref_rmem_f32_13) -> ()
          %slice_148 = cute.slice(%view_145, %coord_76) : !memref_smem_f32_5, !cute.coord<"(_,0,_,_,_,_)">
          %iter_149 = cute.get_iter(%slice_148) : !memref_smem_f32_6
          %ptr_150 = cute.add_offset(%iter_149, %int_tuple_84) : (!cute.ptr<f32, smem, align<8>, S<3,4,3>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, smem, S<3,4,3>>
          %lay_151 = cute.get_layout(%slice_148) : !memref_smem_f32_6
          %view_152 = cute.make_view(%ptr_150, %lay_151) : !memref_smem_f32_7
          cute.copy(%atom_88, %rmem_146, %view_152) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !memref_rmem_f32_13, !memref_smem_f32_7)
          %ptr_153 = cute.add_offset(%iter_149, %int_tuple_89) : (!cute.ptr<f32, smem, align<8>, S<3,4,3>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, smem, S<3,4,3>>
          %view_154 = cute.make_view(%ptr_153, %lay_151) : !memref_smem_f32_7
          cute.copy(%atom_88, %rmem_147, %view_154) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !memref_rmem_f32_13, !memref_smem_f32_7)
          iket.range_pop
          iket.range_push("epi_store_issue")
          nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
          nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
          scf.if %25 {
            %coord_161 = cute.make_coord(%78) : (i32) -> !cute.coord<"(_,?)">
            %slice_162 = cute.slice(%res_smem_tensor, %coord_161) : !memref_smem_f32_2, !cute.coord<"(_,?)">
            %iter_163 = cute.get_iter(%slice_162) : !memref_smem_f32_8
            %coord_164 = cute.make_coord() : () -> !cute.coord<"(_,(0,3))">
            %slice_165 = cute.slice(%res_target_tensors, %coord_164) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1,4)):(((1@0,1@1),0),(0,32@0))">, !cute.coord<"(_,(0,3))">
            %iter_166 = cute.get_iter(%slice_165) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
            %tup = cute.deref_arith_tuple_iter(%iter_166) : !cute.arith_tuple_iter<"(?{div=32},?{div=128},?)">
            %e0_167, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=32},?{div=128},?)">
            %lay_168 = cute.get_layout(%slice_162) : !memref_smem_f32_8
            %lay_169 = cute.get_layout(%slice_165) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
            %append = cute.append_to_rank<2> (%lay_168, %lay_40) : !cute.layout<"((4096,1)):((1,0))">, !cute.layout<"1:0">
            %view_170 = cute.make_view(%iter_163, %append) : !memref_smem_f32_9
            %grouped_171 = cute.group_modes(%view_170) <1, 2> : (!memref_smem_f32_9) -> !memref_smem_f32_10
            %append_172 = cute.append_to_rank<2> (%lay_169, %lay_40) : !cute.layout<"(((32,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
            %int_tuple_173 = cute.make_int_tuple(%e0_167, %e1, %e2) : (!cute.int_tuple<"?{div=32}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=32},?{div=128},?)">
            %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_173) : (!cute.int_tuple<"(?{div=32},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=32},?{div=128},?)">
            %view_174 = cute.make_view(%int_tup_iter, %append_172) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">
            %grouped_175 = cute.group_modes(%view_174) <1, 2> : (!cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">
            %99 = cute_nvgpu.atom.make_exec_tma(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f32, copy_bits = 131072, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = F32_RN>) -> !cute_nvgpu.atom.tma_store<f32, copy_bits = 131072, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>
            cute.copy(%99, %grouped_171, %grouped_175) : (!cute_nvgpu.atom.tma_store<f32, copy_bits = 131072, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>, !memref_smem_f32_10, !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">)
            nvvm.cp.async.bulk.commit.group
          }
          iket.range_pop
          iket.range_pop
          %83 = arith.addi %arg35, %c1_i32 : i32
          iket.range_push("fetch_wait")
          %int_tuple_155 = cute.make_int_tuple(%arg37) : (i32) -> !cute.int_tuple<"?">
          %ptr_156 = cute.add_offset(%iter_9, %int_tuple_155) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %84 = builtin.unrealized_conversion_cast %ptr_156 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity %84, %arg38, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          iket.range_pop
          iket.range_push("fetch_decode")
          %coord_157 = cute.make_coord(%arg37) : (i32) -> !cute.coord<"(0,?)">
          %85 = cute.memref.load(%view, %coord_157) : (!memref_smem_i32, !cute.coord<"(0,?)">) -> i32
          %coord_158 = cute.make_coord(%arg37) : (i32) -> !cute.coord<"(1,?)">
          %86 = cute.memref.load(%view, %coord_158) : (!memref_smem_i32, !cute.coord<"(1,?)">) -> i32
          %coord_159 = cute.make_coord(%arg37) : (i32) -> !cute.coord<"(2,?)">
          %87 = cute.memref.load(%view, %coord_159) : (!memref_smem_i32, !cute.coord<"(2,?)">) -> i32
          %coord_160 = cute.make_coord(%arg37) : (i32) -> !cute.coord<"(3,?)">
          %88 = cute.memref.load(%view, %coord_160) : (!memref_smem_i32, !cute.coord<"(3,?)">) -> i32
          nvvm.bar.warp.sync %c-1_i32 : i32
          %89 = nvvm.elect.sync -> i1
          scf.if %89 {
            %ptr_161 = cute.add_offset(%ptr_11, %int_tuple_155) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %99 = builtin.unrealized_conversion_cast %ptr_161 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            %100 = nvvm.mbarrier.arrive %99, %c1_i32 : !llvm.ptr<3> -> i64
          }
          %90 = arith.addi %arg37, %c1_i32 : i32
          %91 = arith.addi %arg36, %c1_i32 : i32
          %92 = arith.cmpi eq, %90, %c1_i32 : i32
          %93 = arith.select %92, %c0_i32, %90 : i32
          %94 = scf.if %92 -> (i32) {
            %99 = arith.xori %arg38, %c1_i32 : i32
            scf.yield %99 : i32
          } else {
            scf.yield %arg38 : i32
          }
          %95 = arith.cmpi ne, %88, %c0_i32 : i32
          iket.range_pop
          %96 = arith.floordivsi %87, %c2_i32 : i32
          %97 = arith.muli %96, %c2_i32 : i32
          %98 = arith.subi %87, %97 : i32
          scf.yield %41#2, %41#3, %41#4, %85, %86, %98, %96, %95, %83, %91, %93, %94 : i32, i32, i32, i32, i32, i32, i32, i1, i32, i32, i32, i32
        }
        nvvm.griddepcontrol launch_dependents
        scf.if %25 {
          nvvm.cp.async.bulk.wait_group 0 {read}
        }
      }
      return
    }
  }
  func.func @cutlass___call___quackgemm_default_epiGemmDefaultSm90_object_at__FakeTensorBFloat16_div1_div1_div1_i64_div8_i64_div8_1_FakeTensorBFloat16div1div1div1i64div8i64div81_FakeTensorFloat(%arg0: !memref_gmem_bf16, %arg1: !memref_gmem_bf16, %arg2: !memref_gmem_f32, %arg3: i32, %arg4: i32, %arg5: !cuda.stream) -> i32 attributes {llvm.emit_c_interface} {
    %c31_i32 = arith.constant 31 : i32
    %c384_i32 = arith.constant 384 : i32
    %c232448_i64 = arith.constant 232448 : i64
    %c4294967295_i64 = arith.constant 4294967295 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = cute.static : !cute.swizzle<"S<3,4,3>">
    %false = arith.constant false
    %iter = cute.get_iter(%arg0) : !memref_gmem_bf16
    %iter_0 = cute.get_iter(%arg1) : !memref_gmem_bf16
    %iter_1 = cute.get_iter(%arg2) : !memref_gmem_f32
    %lay = cute.get_layout(%arg0) : !memref_gmem_bf16
    %1 = cute.select<[1, 2, 0]> (%lay) : (!cute.layout<"(?,?,?):(?{i64 div=8},?{i64 div=8},1)">) -> !cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">
    %view = cute.make_view(%iter, %1) : !memref_gmem_bf16_1
    %lay_2 = cute.get_layout(%arg1) : !memref_gmem_bf16
    %2 = cute.select<[1, 2, 0]> (%lay_2) : (!cute.layout<"(?,?,?):(?{i64 div=8},?{i64 div=8},1)">) -> !cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">
    %view_3 = cute.make_view(%iter_0, %2) : !memref_gmem_bf16_1
    %lay_4 = cute.get_layout(%arg2) : !memref_gmem_f32
    %3 = cute.select<[1, 2, 0]> (%lay_4) : (!cute.layout<"(?,?,?):(?{i64 div=4},?{i64 div=4},1)">) -> !cute.layout<"(?,?,?):(?{i64 div=4},1,?{i64 div=4})">
    %view_5 = cute.make_view(%iter_1, %3) : !memref_gmem_f32_1
    %4 = cute.get_shape(%1) : (!cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">) -> !cute.shape<"(?,?,?)">
    %e0, %e1, %e2 = cute.get_leaves(%4) : !cute.shape<"(?,?,?)">
    %itup = cute.to_int_tuple(%e0) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_6 = cute.to_int_tuple(%e1) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_7 = cute.to_int_tuple(%e2) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %5 = cute.get_shape(%2) : (!cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">) -> !cute.shape<"(?,?,?)">
    %e0_8, %e1_9, %e2_10 = cute.get_leaves(%5) : !cute.shape<"(?,?,?)">
    %itup_11 = cute.to_int_tuple(%e0_8) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_12 = cute.to_int_tuple(%e1_9) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_13 = cute.to_int_tuple(%e2_10) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %6 = cute.get_scalars(%itup_13) : !cute.int_tuple<"?">
    %7 = cute.get_shape(%3) : (!cute.layout<"(?,?,?):(?{i64 div=4},1,?{i64 div=4})">) -> !cute.shape<"(?,?,?)">
    %e0_14, %e1_15, %e2_16 = cute.get_leaves(%7) : !cute.shape<"(?,?,?)">
    %itup_17 = cute.to_int_tuple(%e0_14) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_18 = cute.to_int_tuple(%e1_15) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_19 = cute.to_int_tuple(%e2_16) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %atom = cute.make_atom(%false) : (i1) -> !cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>
    %8 = cute.make_tiled_mma(%atom) : !mma_bf16_bf16_f32_64x128x16
    %shape = cute.make_shape() : () -> !cute.shape<"(1,1,1)">
    %lay_20 = cute.make_layout(%shape) : !cute.layout<"(1,1,1):(0,0,0)">
    %shape_21 = cute.make_shape() : () -> !cute.shape<"(8,64)">
    %stride = cute.make_stride() : () -> !cute.stride<"(64,1)">
    %lay_22 = cute.make_layout(%shape_21, %stride) : !cute.layout<"(8,64):(64,1)">
    %int_tuple = cute.make_int_tuple() : () -> !cute.int_tuple<"0">
    %lay_23 = cute.make_composed_layout(%0, %int_tuple, %lay_22) : !cute.composed_layout<"S<3,4,3> o 0 o (8,64):(64,1)">
    %shape_24 = cute.make_shape() : () -> !cute.shape<"(128,64,6)">
    %int_tuple_25 = cute.make_int_tuple() : () -> !cute.int_tuple<"(0,1,2)">
    %tile_to_shape = cute.tile_to_shape(%lay_23, %shape_24, %int_tuple_25) : (!cute.composed_layout<"S<3,4,3> o 0 o (8,64):(64,1)">, !cute.shape<"(128,64,6)">, !cute.int_tuple<"(0,1,2)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">
    %shape_26 = cute.make_shape() : () -> !cute.shape<"(8,32)">
    %stride_27 = cute.make_stride() : () -> !cute.stride<"(32,1)">
    %lay_28 = cute.make_layout(%shape_26, %stride_27) : !cute.layout<"(8,32):(32,1)">
    %lay_29 = cute.make_composed_layout(%0, %int_tuple, %lay_28) : !cute.composed_layout<"S<3,4,3> o 0 o (8,32):(32,1)">
    %shape_30 = cute.make_shape() : () -> !cute.shape<"(128,32,2)">
    %tile_to_shape_31 = cute.tile_to_shape(%lay_29, %shape_30, %int_tuple_25) : (!cute.composed_layout<"S<3,4,3> o 0 o (8,32):(32,1)">, !cute.shape<"(128,32,2)">, !cute.int_tuple<"(0,1,2)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(32,1),(1,2)):((32,256),(1,0),(0,4096))">
    %coord = cute.make_coord() : () -> !cute.coord<"(_,_,0)">
    %slice = cute.slice(%tile_to_shape, %coord) : !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">, !cute.coord<"(_,_,0)">
    %shape_32 = cute.make_shape(%itup, %itup_6, %itup_7) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?,?)">
    %9 = cute.make_identity_layout(%shape_32) : !cute.layout<"(?,?,?):(1@0,1@1,1@2)">
    %tile = cute.make_tile() : () -> !cute.tile<"[128:1;64:1]">
    %10 = cute.composition(%9, %tile) : (!cute.layout<"(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;64:1]">) -> !cute.layout<"(128,64):(1@0,1@1)">
    %non_exec_atom, %tma_tensor = cute_nvgpu.atom.make_non_exec_tiled_tma_load(%view, %slice, %10) <{kind = <sm_90> num_multicast = 1}> : (!memref_gmem_bf16_1, !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1)):((64,512),(1,0))">, !cute.layout<"(128,64):(1@0,1@1)">) -> (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">)
    %shape_33 = cute.make_shape(%itup_11, %itup_12, %itup_13) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?,?)">
    %11 = cute.make_identity_layout(%shape_33) : !cute.layout<"(?,?,?):(1@0,1@1,1@2)">
    %12 = cute.composition(%11, %tile) : (!cute.layout<"(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;64:1]">) -> !cute.layout<"(128,64):(1@0,1@1)">
    %non_exec_atom_34, %tma_tensor_35 = cute_nvgpu.atom.make_non_exec_tiled_tma_load(%view_3, %slice, %12) <{kind = <sm_90> num_multicast = 1}> : (!memref_gmem_bf16_1, !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1)):((64,512),(1,0))">, !cute.layout<"(128,64):(1@0,1@1)">) -> (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">)
    %slice_36 = cute.slice(%tile_to_shape_31, %coord) : !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(32,1),(1,2)):((32,256),(1,0),(0,4096))">, !cute.coord<"(_,_,0)">
    %shape_37 = cute.make_shape(%itup_17, %itup_18, %itup_19) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?,?)">
    %13 = cute.make_identity_layout(%shape_37) : !cute.layout<"(?,?,?):(1@0,1@1,1@2)">
    %tile_38 = cute.make_tile() : () -> !cute.tile<"[128:1;32:1]">
    %14 = cute.composition(%13, %tile_38) : (!cute.layout<"(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;32:1]">) -> !cute.layout<"(128,32):(1@0,1@1)">
    %15 = cute.composition(%13, %14) : (!cute.layout<"(?,?,?):(1@0,1@1,1@2)">, !cute.layout<"(128,32):(1@0,1@1)">) -> !cute.layout<"(128,32):(1@0,1@1)">
    %non_exec_atom_39, %tma_tensor_40 = cute_nvgpu.atom.make_non_exec_tiled_tma_store(%view_5, %slice_36, %15) : (!memref_gmem_f32_1, !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(32,1)):((32,256),(1,0))">, !cute.layout<"(128,32):(1@0,1@1)">) -> (!cute_nvgpu.atom.non_exec_tiled_tma_store<f32, copy_bits = 131072, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = F32_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">)
    %sz = cute.size(%view) <{mode = [0]}> : (!memref_gmem_bf16_1) -> !cute.int_tuple<"?">
    %e0_41 = cute.get_leaves(%sz) : !cute.int_tuple<"?">
    %int_tuple_42 = cute.make_int_tuple(%e0_41) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %tile_43 = cute.make_tile() : () -> !cute.tile<"128:1">
    %shp = cute.ceil_div(%int_tuple_42, %tile_43) : !cute.int_tuple<"?">, !cute.tile<"128:1">
    %e0_44 = cute.get_leaves(%shp) : !cute.int_tuple<"?">
    %sz_45 = cute.size(%view_3) <{mode = [0]}> : (!memref_gmem_bf16_1) -> !cute.int_tuple<"?">
    %e0_46 = cute.get_leaves(%sz_45) : !cute.int_tuple<"?">
    %int_tuple_47 = cute.make_int_tuple(%e0_46) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %shp_48 = cute.ceil_div(%int_tuple_47, %tile_43) : !cute.int_tuple<"?">, !cute.tile<"128:1">
    %e0_49 = cute.get_leaves(%shp_48) : !cute.int_tuple<"?">
    %int_tuple_50 = cute.make_int_tuple(%e0_44) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %tile_51 = cute.make_tile() : () -> !cute.tile<"1:0">
    %shp_52 = cute.ceil_div(%int_tuple_50, %tile_51) : !cute.int_tuple<"?">, !cute.tile<"1:0">
    %e0_53 = cute.get_leaves(%shp_52) : !cute.int_tuple<"?">
    %16 = cute.get_scalars(%e0_53) : !cute.int_tuple<"?">
    %int_tuple_54 = cute.make_int_tuple(%e0_49) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %shp_55 = cute.ceil_div(%int_tuple_54, %tile_51) : !cute.int_tuple<"?">, !cute.tile<"1:0">
    %e0_56 = cute.get_leaves(%shp_55) : !cute.int_tuple<"?">
    %17 = cute.get_scalars(%e0_56) : !cute.int_tuple<"?">
    %int_tuple_57 = cute.make_int_tuple(%e0_53, %e0_56) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?,?)">
    %sz_58 = cute.size(%int_tuple_57) : (!cute.int_tuple<"(?,?)">) -> !cute.int_tuple<"?">
    %e0_59 = cute.get_leaves(%sz_58) : !cute.int_tuple<"?">
    %18 = cute.get_scalars(%e0_59) : !cute.int_tuple<"?">
    %int_tuple_60 = cute.make_int_tuple(%arg4) : (i32) -> !cute.int_tuple<"?">
    %add = cute.tuple_add(%e0_53, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %int_tuple_61 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
    %sub = cute.tuple_sub(%add, %int_tuple_61) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %div = cute.tuple_div(%sub, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %mul = cute.tuple_mul(%div, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %19 = cute.get_scalars(%mul) : !cute.int_tuple<"?">
    %add_62 = cute.tuple_add(%e0_56, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %sub_63 = cute.tuple_sub(%add_62, %int_tuple_61) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %div_64 = cute.tuple_div(%sub_63, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %mul_65 = cute.tuple_mul(%div_64, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %20 = cute.get_scalars(%mul_65) : !cute.int_tuple<"?">
    %21 = arith.cmpi sle, %20, %19 : i32
    %22 = arith.extui %21 : i1 to i32
    %23 = arith.cmpi eq, %22, %c0_i32 : i32
    %24 = arith.select %23, %16, %17 : i32
    %25 = arith.select %23, %17, %16 : i32
    %26 = arith.minsi %arg4, %24 : i32
    %27 = arith.remsi %24, %26 : i32
    %28 = arith.floordivsi %24, %26 : i32
    %29 = arith.muli %26, %25 : i32
    %sub_66 = cute.tuple_sub(%e0_59, %int_tuple_61) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %30 = cute.get_scalars(%sub_66) : !cute.int_tuple<"?">
    %31 = "llvm.intr.ctlz"(%30) <{is_zero_poison = false}> : (i32) -> i32
    %32 = arith.subi %c31_i32, %31 : i32
    %33 = arith.maxsi %32, %c0_i32 : i32
    %34 = arith.shli %c1_i32, %33 : i32
    %35 = arith.extui %34 : i32 to i64
    %36 = arith.muli %35, %c4294967296_i64 : i64
    %37 = arith.extui %18 : i32 to i64
    %38 = arith.addi %36, %37 : i64
    %39 = arith.subi %38, %c1_i64 : i64
    %40 = arith.divui %39, %37 : i64
    %41 = arith.andi %40, %c4294967295_i64 : i64
    %42 = arith.trunci %41 : i64 to i32
    %43 = arith.subi %26, %c1_i32 : i32
    %44 = "llvm.intr.ctlz"(%43) <{is_zero_poison = false}> : (i32) -> i32
    %45 = arith.subi %c31_i32, %44 : i32
    %46 = arith.maxsi %45, %c0_i32 : i32
    %47 = arith.shli %c1_i32, %46 : i32
    %48 = arith.extui %47 : i32 to i64
    %49 = arith.muli %48, %c4294967296_i64 : i64
    %50 = arith.extui %26 : i32 to i64
    %51 = arith.addi %49, %50 : i64
    %52 = arith.subi %51, %c1_i64 : i64
    %53 = arith.divui %52, %50 : i64
    %54 = arith.andi %53, %c4294967295_i64 : i64
    %55 = arith.trunci %54 : i64 to i32
    %56 = arith.cmpi sgt, %27, %c0_i32 : i32
    %57 = arith.select %56, %27, %c1_i32 : i32
    %58 = arith.subi %57, %c1_i32 : i32
    %59 = "llvm.intr.ctlz"(%58) <{is_zero_poison = false}> : (i32) -> i32
    %60 = arith.subi %c31_i32, %59 : i32
    %61 = arith.maxsi %60, %c0_i32 : i32
    %62 = arith.shli %c1_i32, %61 : i32
    %63 = arith.extui %62 : i32 to i64
    %64 = arith.muli %63, %c4294967296_i64 : i64
    %65 = arith.extui %57 : i32 to i64
    %66 = arith.addi %64, %65 : i64
    %67 = arith.subi %66, %c1_i64 : i64
    %68 = arith.divui %67, %65 : i64
    %69 = arith.andi %68, %c4294967295_i64 : i64
    %70 = arith.trunci %69 : i64 to i32
    %71 = arith.subi %29, %c1_i32 : i32
    %72 = "llvm.intr.ctlz"(%71) <{is_zero_poison = false}> : (i32) -> i32
    %73 = arith.subi %c31_i32, %72 : i32
    %74 = arith.maxsi %73, %c0_i32 : i32
    %75 = arith.shli %c1_i32, %74 : i32
    %76 = arith.extui %75 : i32 to i64
    %77 = arith.muli %76, %c4294967296_i64 : i64
    %78 = arith.extui %29 : i32 to i64
    %79 = arith.addi %77, %78 : i64
    %80 = arith.subi %79, %c1_i64 : i64
    %81 = arith.divui %80, %78 : i64
    %82 = arith.andi %81, %c4294967295_i64 : i64
    %83 = arith.trunci %82 : i64 to i32
    %int_tuple_67 = cute.make_int_tuple(%e0_53, %e0_56, %itup_13) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?,?,?)">
    %sz_68 = cute.size(%int_tuple_67) : (!cute.int_tuple<"(?,?,?)">) -> !cute.int_tuple<"?">
    %e0_69 = cute.get_leaves(%sz_68) : !cute.int_tuple<"?">
    %mul_70 = cute.tuple_mul(%e0_69, %int_tuple_61) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %int_tuple_71 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
    %mul_72 = cute.tuple_mul(%mul_70, %int_tuple_71) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?{div=2}">
    %84 = cute.get_scalars(%mul_72) : !cute.int_tuple<"?{div=2}">
    %85 = arith.minsi %84, %arg3 : i32
    %86 = cute.kernel_smem_size @kernels::@kernel_cutlass_kernel_quackgemm_default_epiGemmDefaultSm90_object_at__TiledMMA_ThrLayoutVMNK128211112800_PermutationMNK____MMAAtom_ThrID1281_ShapeMNK6412816_TVLayoutA12864160164_TVLayoutB_0 : i64
    %87 = arith.cmpi sgt, %86, %c232448_i64 : i64
    scf.if %87 {
      cute.print("\0AError: kernel '@kernels::@kernel_cutlass_kernel_quackgemm_default_epiGemmDefaultSm90_object_at__TiledMMA_ThrLayoutVMNK128211112800_PermutationMNK____MMAAtom_ThrID1281_ShapeMNK6412816_TVLayoutA12864160164_TVLayoutB_0' launch shared memory exceeds current GPU arch sm_90a allowed. Allocated: {} bytes. Max: 232448 bytes.\0A\0A", %86) : i64
    }
    %88 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c384_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %86, gridDim = (%c1_i32, %c1_i32, %85), stream = %arg5) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17>
    cuda.launch_cfg.programmatic_stream_serialization_allowed[%88] %c1_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    cuda.launch_cfg.cluster_dim[%88] (%c1_i32, %c1_i32, %c1_i32) : !cuda.launch_cfg<max_attrs = 17>, i32, i32, i32
    cuda.launch_cfg.cooperative[%88] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    %89 = cuda.launch_ex @kernels::@kernel_cutlass_kernel_quackgemm_default_epiGemmDefaultSm90_object_at__TiledMMA_ThrLayoutVMNK128211112800_PermutationMNK____MMAAtom_ThrID1281_ShapeMNK6412816_TVLayoutA12864160164_TVLayoutB_0<%88> (%8, %non_exec_atom, %tma_tensor, %non_exec_atom_34, %tma_tensor_35, %non_exec_atom_39, %tma_tensor_40, %lay_20, %tile_to_shape, %tile_to_shape, %tile_to_shape_31, %16, %17, %6, %22, %42, %33, %18, %28, %55, %46, %26, %70, %61, %57, %83, %74, %29) {assume_kernel_attr = #cuda.assume_kernel_attr<true>} : !cuda.launch_cfg<max_attrs = 17>, (!mma_bf16_bf16_f32_64x128x16, !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute_nvgpu.atom.non_exec_tiled_tma_store<f32, copy_bits = 131072, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = F32_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute.layout<"(1,1,1):(0,0,0)">, !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">, !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">, !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(32,1),(1,2)):((32,256),(1,0),(0,4096))">, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !cuda.result
    %90 = cuda.cast %89 : !cuda.result -> i32
    cuda.return_if_error %90 : i32
    return %c0_i32 : i32
  }
}

