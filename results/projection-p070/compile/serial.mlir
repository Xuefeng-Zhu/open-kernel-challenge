!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 128 b>, layout_copy_tv = <"(256,4):(4,1)">, tiler_mn = <"[1024:1]">>
!copy_stsm_4 = !cute.tiled_copy<!cute_nvgpu.atom.stsm<bf16, mode = <"(8,8)">, num_matrices = 4, n>, layout_copy_tv = <"((4,64),((2,2,2),1)):((128,1),((512,64,1024),0))">, tiler_mn = <"[(8,8,2):(1,16,8);(4,2,2):(2,1,8)]">>
!memref_gmem_bf16 = !cute.memref<bf16, gmem, align<16>, "(?,?,?):(?{i64 div=8},?{i64 div=8},1)">
!memref_gmem_bf16_1 = !cute.memref<bf16, gmem, align<16>, "(?,?,?):(?{i64 div=8},1,?{i64 div=8})">
!memref_gmem_f32 = !cute.memref<f32, gmem, align<16>, "(?,?,?,?):(1,?{i64 div=4},?{i64 div=4},?{i64 div=4})">
!memref_gmem_f32_1 = !cute.memref<f32, gmem, align<16>, "4096:1">
!memref_gmem_f32_2 = !cute.memref<f32, gmem, align<16>, "((4,1),(4)):((1,0),(1024))">
!memref_gmem_f32_3 = !cute.memref<f32, gmem, align<16>, "((4,1)):((1,0))">
!memref_gmem_f32_4 = !cute.memref<f32, gmem, align<16>, "((4,1),1):((1,0),0)">
!memref_gmem_f32_5 = !cute.memref<f32, gmem, align<16>, "((4,1),(1)):((1,0),(0))">
!memref_gmem_i32 = !cute.memref<i32, gmem, "(?,?,?):(?{i64},1,?{i64})">
!memref_rmem_bf16 = !cute.memref<bf16, rmem, align<32>, "(((2,2,2),1),1,2):(((1,2,4),0),0,8)">
!memref_rmem_bf16_1 = !cute.memref<bf16, rmem, align<32>, "(((2,2,2),1),(1,2)):(((1,2,4),0),(0,8))">
!memref_rmem_f32 = !cute.memref<f32, rmem, align<32>, "((2,2,16),1,1):((1,2,4),0,0)">
!memref_rmem_f32_1 = !cute.memref<f32, rmem, align<32>, "(((2,2,2),1),1,2):(((1,2,4),0),0,8)">
!memref_rmem_f32_2 = !cute.memref<f32, rmem, align<32>, "((2,2,2),1,(8,1)):((1,2,4),0,(8,0))">
!memref_rmem_f32_3 = !cute.memref<f32, rmem, align<32>, "((2,2,2),1,2,(1,1,1),1,4):((1,2,4),0,8,(0,0,0),0,16)">
!memref_rmem_f32_4 = !cute.memref<f32, rmem, align<32>, "((2,2,2),1,2,1,4):((1,2,4),0,8,0,16)">
!memref_rmem_f32_5 = !cute.memref<f32, rmem, align<32>, "((2,2,2),1,2,(1,4)):((1,2,4),0,8,(0,16))">
!memref_rmem_f32_6 = !cute.memref<f32, rmem, align<32>, "((8,1),1,2,(1,4)):((1,0),0,8,(0,16))">
!memref_rmem_f32_7 = !cute.memref<f32, rmem, align<32>, "((8,1),1,2):((1,0),0,8)">
!memref_rmem_f32_8 = !cute.memref<f32, rmem, align<32>, "(16,1):(1,0)">
!memref_rmem_f32_9 = !cute.memref<f32, rmem, align<32>, "(8,2):(1,8)">
!memref_rmem_f32_10 = !cute.memref<f32, rmem, align<32>, "(4,1):(1,0)">
!memref_rmem_f32_11 = !cute.memref<f32, rmem, align<32>, "(4,(1)):(1,(0))">
!memref_rmem_f32_12 = !cute.memref<f32, rmem, align<16>, "(4,1):(1,0)">
!memref_rmem_f32_13 = !cute.memref<f32, rmem, align<16>, "(4,(1)):(1,(0))">
!memref_rmem_f32_14 = !cute.memref<f32, rmem, align<32>, "4:1">
!memref_rmem_f32_15 = !cute.memref<f32, rmem, align<16>, "4:1">
!memref_smem_bf16 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">
!memref_smem_bf16_1 = !cute.memref<bf16, smem, align<1024>, S<2,4,3>, "((8,16),(32,1),(1,4)):((32,256),(1,0),(0,4096))">
!memref_smem_bf16_2 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "(((8,16),(64,1)),(1,6)):(((64,512),(1,0)),(0,8192))">
!memref_smem_bf16_3 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "(((8,16),(64,1))):(((64,512),(1,0)))">
!memref_smem_bf16_4 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((8192,1)):((1,0))">
!memref_smem_bf16_5 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((8192,1),1):((1,0),0)">
!memref_smem_bf16_6 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((8192,1),(1)):((1,0),(0))">
!memref_smem_bf16_7 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((64,16),1,4,(1,6)):((64,1),0,16,(0,8192))">
!memref_smem_bf16_8 = !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((128,16),1,4,(1,6)):((64,1),0,16,(0,8192))">
!memref_smem_bf16_9 = !cute.memref<bf16, smem, align<1024>, S<2,4,3>, "(((8,16),(32,1)),(1,4)):(((32,256),(1,0)),(0,4096))">
!memref_smem_bf16_10 = !cute.memref<bf16, smem, align<1024>, S<2,4,3>, "((4096,1),(1,4)):((1,0),(0,4096))">
!memref_smem_bf16_11 = !cute.memref<bf16, smem, align<16>, S<2,4,3>, "(((2,4),1),1,2,(1,4)):(((1,2),0),0,16,(0,4096))">
!memref_smem_bf16_12 = !cute.memref<bf16, smem, align<16>, S<2,4,3>, "(((2,4),1),1,2):(((1,2),0),0,16)">
!memref_smem_bf16_13 = !cute.memref<bf16, smem, align<16>, S<2,4,3>, "(((2,4),1),(1,2)):(((1,2),0),(0,16))">
!memref_smem_bf16_14 = !cute.memref<bf16, smem, align<1024>, S<2,4,3>, "((4096,1)):((1,0))">
!memref_smem_bf16_15 = !cute.memref<bf16, smem, align<1024>, S<2,4,3>, "((4096,1),1):((1,0),0)">
!memref_smem_bf16_16 = !cute.memref<bf16, smem, align<1024>, S<2,4,3>, "((4096,1),(1)):((1,0),(0))">
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
    cuda.kernel @kernel_cutlass_kernel_quackgemm_default_epiGemmDefaultSm90_object_at__TiledMMA_ThrLayoutVMNK128211112800_PermutationMNK____MMAAtom_ThrID1281_ShapeMNK6412816_TVLayoutA12864160164_TVLayoutB_0(%arg0: !mma_bf16_bf16_f32_64x128x16, %arg1: !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN> {cute_nvgpu.grid_constant}, %arg2: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, %arg3: !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN> {cute_nvgpu.grid_constant}, %arg4: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, %arg5: !cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 65536, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN> {cute_nvgpu.grid_constant}, %arg6: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, %arg7: f32, %arg8: !memref_gmem_i32, %arg9: !memref_gmem_f32, %arg10: !cute.layout<"(1,1,1):(0,0,0)">, %arg11: !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">, %arg12: !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">, %arg13: !cute.composed_layout<"S<2,4,3> o 0 o ((8,16),(32,1),(1,4)):((32,256),(1,0),(0,4096))">, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: i32) attributes {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, cute.kernel, cutlass_launch_facts = {cluster_launch = true, cooperative_launch = false, exact_block_dim = array<i64: 384, 1, 1>, exact_cluster_dim = array<i64: 1, 1, 1>, schema_version = 1 : i64}, gpu.kernel, nvvm.minctasm = 1 : i32, nvvm.reqntid = array<i32: 384, 1, 1>, smem.hint_smem_base_uniform, smem.partition_num = 2 : i32} {
      %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
      %c4_i32 = arith.constant 4 : i32
      %c3_i32 = arith.constant 3 : i32
      %c256_i32 = arith.constant 256 : i32
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
      %iter = cute.get_iter(%arg8) : !memref_gmem_i32
      %iter_0 = cute.get_iter(%arg9) : !memref_gmem_f32
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
        cute_nvgpu.prefetch_tma_desc(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 65536, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> ()
      }
      %smem = cute.memref.alloca() {smem.partition_id = 1 : i32, smem.struct_fields = ["sD:32768:0", "sC:0:32768", "epi:0:32768", "sA:98304:32768", "sB:98304:131072", "sAuxA:0:229376", "sSFA:0:229376", "sSFB:0:229376"]} : !memref_smem_i8
      %iter_1 = cute.get_iter(%smem) : !memref_smem_i8
      %int_tuple = cute.make_int_tuple() : () -> !cute.int_tuple<"32768">
      %ptr = cute.add_offset(%iter_1, %int_tuple) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"32768">) -> !cute.ptr<i8, smem, align<1024>>
      %int_tuple_2 = cute.make_int_tuple() : () -> !cute.int_tuple<"131072">
      %ptr_3 = cute.add_offset(%iter_1, %int_tuple_2) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"131072">) -> !cute.ptr<i8, smem, align<1024>>
      %smem_4 = cute.memref.alloca() {smem.partition_id = 0 : i32} : !memref_smem_i8_1
      %iter_5 = cute.get_iter(%smem_4) : !memref_smem_i8_1
      %iter_6 = cute.recast_iter(%iter_5) : !cute.ptr<i8, smem, align<8>> to !cute.ptr<i64, smem>
      %14 = arith.cmpi eq, %12, %c0_i32 : i32
      scf.if %14 {
        %23 = nvvm.elect.sync -> i1
        scf.if %23 {
          %24 = builtin.unrealized_conversion_cast %iter_6 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %24, %c1_i32 : !llvm.ptr<3>, i32
          %int_tuple_23 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
          %ptr_24 = cute.add_offset(%iter_6, %int_tuple_23) : (!cute.ptr<i64, smem>, !cute.int_tuple<"1">) -> !cute.ptr<i64, smem>
          %25 = builtin.unrealized_conversion_cast %ptr_24 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %25, %c1_i32 : !llvm.ptr<3>, i32
          %int_tuple_25 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
          %ptr_26 = cute.add_offset(%iter_6, %int_tuple_25) : (!cute.ptr<i64, smem>, !cute.int_tuple<"2">) -> !cute.ptr<i64, smem>
          %26 = builtin.unrealized_conversion_cast %ptr_26 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %26, %c1_i32 : !llvm.ptr<3>, i32
          %int_tuple_27 = cute.make_int_tuple() : () -> !cute.int_tuple<"3">
          %ptr_28 = cute.add_offset(%iter_6, %int_tuple_27) : (!cute.ptr<i64, smem>, !cute.int_tuple<"3">) -> !cute.ptr<i64, smem>
          %27 = builtin.unrealized_conversion_cast %ptr_28 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %27, %c1_i32 : !llvm.ptr<3>, i32
          %int_tuple_29 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
          %ptr_30 = cute.add_offset(%iter_6, %int_tuple_29) : (!cute.ptr<i64, smem>, !cute.int_tuple<"4">) -> !cute.ptr<i64, smem>
          %28 = builtin.unrealized_conversion_cast %ptr_30 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %28, %c1_i32 : !llvm.ptr<3>, i32
          %int_tuple_31 = cute.make_int_tuple() : () -> !cute.int_tuple<"5">
          %ptr_32 = cute.add_offset(%iter_6, %int_tuple_31) : (!cute.ptr<i64, smem>, !cute.int_tuple<"5">) -> !cute.ptr<i64, smem>
          %29 = builtin.unrealized_conversion_cast %ptr_32 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %29, %c1_i32 : !llvm.ptr<3>, i32
        }
      }
      %int_tuple_7 = cute.make_int_tuple() : () -> !cute.int_tuple<"6">
      %ptr_8 = cute.add_offset(%iter_6, %int_tuple_7) : (!cute.ptr<i64, smem>, !cute.int_tuple<"6">) -> !cute.ptr<i64, smem>
      scf.if %14 {
        %23 = nvvm.elect.sync -> i1
        scf.if %23 {
          %24 = builtin.unrealized_conversion_cast %ptr_8 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %24, %c8_i32 : !llvm.ptr<3>, i32
          %int_tuple_23 = cute.make_int_tuple() : () -> !cute.int_tuple<"7">
          %ptr_24 = cute.add_offset(%iter_6, %int_tuple_23) : (!cute.ptr<i64, smem>, !cute.int_tuple<"7">) -> !cute.ptr<i64, smem>
          %25 = builtin.unrealized_conversion_cast %ptr_24 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %25, %c8_i32 : !llvm.ptr<3>, i32
          %int_tuple_25 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
          %ptr_26 = cute.add_offset(%iter_6, %int_tuple_25) : (!cute.ptr<i64, smem>, !cute.int_tuple<"8">) -> !cute.ptr<i64, smem>
          %26 = builtin.unrealized_conversion_cast %ptr_26 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %26, %c8_i32 : !llvm.ptr<3>, i32
          %int_tuple_27 = cute.make_int_tuple() : () -> !cute.int_tuple<"9">
          %ptr_28 = cute.add_offset(%iter_6, %int_tuple_27) : (!cute.ptr<i64, smem>, !cute.int_tuple<"9">) -> !cute.ptr<i64, smem>
          %27 = builtin.unrealized_conversion_cast %ptr_28 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %27, %c8_i32 : !llvm.ptr<3>, i32
          %int_tuple_29 = cute.make_int_tuple() : () -> !cute.int_tuple<"10">
          %ptr_30 = cute.add_offset(%iter_6, %int_tuple_29) : (!cute.ptr<i64, smem>, !cute.int_tuple<"10">) -> !cute.ptr<i64, smem>
          %28 = builtin.unrealized_conversion_cast %ptr_30 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %28, %c8_i32 : !llvm.ptr<3>, i32
          %int_tuple_31 = cute.make_int_tuple() : () -> !cute.int_tuple<"11">
          %ptr_32 = cute.add_offset(%iter_6, %int_tuple_31) : (!cute.ptr<i64, smem>, !cute.int_tuple<"11">) -> !cute.ptr<i64, smem>
          %29 = builtin.unrealized_conversion_cast %ptr_32 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %29, %c8_i32 : !llvm.ptr<3>, i32
        }
      }
      %15 = arith.remsi %0, %c32_i32 : i32
      %16 = arith.cmpi slt, %15, %c1_i32 : i32
      %smem_9 = cute.memref.alloca() {smem.partition_id = 0 : i32} : !memref_smem_i8_2
      %iter_10 = cute.get_iter(%smem_9) : !memref_smem_i8_2
      %iter_11 = cute.recast_iter(%iter_10) : !cute.ptr<i8, smem, align<8>> to !cute.ptr<i64, smem>
      scf.if %14 {
        %23 = nvvm.elect.sync -> i1
        scf.if %23 {
          %24 = builtin.unrealized_conversion_cast %iter_11 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %24, %c1_i32 : !llvm.ptr<3>, i32
        }
      }
      %int_tuple_12 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
      %ptr_13 = cute.add_offset(%iter_11, %int_tuple_12) : (!cute.ptr<i64, smem>, !cute.int_tuple<"1">) -> !cute.ptr<i64, smem>
      scf.if %14 {
        %23 = nvvm.elect.sync -> i1
        scf.if %23 {
          %24 = builtin.unrealized_conversion_cast %ptr_13 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.init %24, %c9_i32 : !llvm.ptr<3>, i32
        }
      }
      %shape = cute.make_shape() : () -> !cute.shape<"(4,1)">
      %lay = cute.make_layout(%shape) : !cute.layout<"(4,1):(1,0)">
      %smem_14 = cute.memref.alloca() {smem.partition_id = 0 : i32} : !memref_smem_i8_3
      %iter_15 = cute.get_iter(%smem_14) : !memref_smem_i8_3
      %iter_16 = cute.recast_iter(%iter_15) : !cute.ptr<i8, smem, align<16>> to !cute.ptr<i32, smem, align<16>>
      %view = cute.make_view(%iter_16, %lay) : !memref_smem_i32
      nvvm.fence.mbarrier.init
      %17 = cute.composed_get_outer(%arg11) : (!cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">) -> !cute.layout<"((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">
      %iter_17 = cute.recast_iter(%ptr) : !cute.ptr<i8, smem, align<1024>> to !cute.ptr<bf16, smem, align<1024>, S<3,4,3>>
      %view_18 = cute.make_view(%iter_17, %17) : !memref_smem_bf16
      %18 = cute.composed_get_outer(%arg12) : (!cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">) -> !cute.layout<"((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">
      %iter_19 = cute.recast_iter(%ptr_3) : !cute.ptr<i8, smem, align<1024>> to !cute.ptr<bf16, smem, align<1024>, S<3,4,3>>
      %view_20 = cute.make_view(%iter_19, %18) : !memref_smem_bf16
      %19 = cute.composed_get_outer(%arg13) : (!cute.composed_layout<"S<2,4,3> o 0 o ((8,16),(32,1),(1,4)):((32,256),(1,0),(0,4096))">) -> !cute.layout<"((8,16),(32,1),(1,4)):((32,256),(1,0),(0,4096))">
      %iter_21 = cute.recast_iter(%iter_1) : !cute.ptr<i8, smem, align<1024>> to !cute.ptr<bf16, smem, align<1024>, S<2,4,3>>
      %view_22 = cute.make_view(%iter_21, %19) : !memref_smem_bf16_1
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
          %int_tuple_23 = cute.make_int_tuple(%arg14, %arg15, %arg16) : (i32, i32, i32) -> !cute.int_tuple<"(?,?,?)">
          %sz_24 = cute.size(%int_tuple_23) : (!cute.int_tuple<"(?,?,?)">) -> !cute.int_tuple<"?">
          %e0_25 = cute.get_leaves(%sz_24) : !cute.int_tuple<"?">
          %int_tuple_26 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
          %mul = cute.tuple_mul(%e0_25, %int_tuple_26) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?{div=2}">
          %26 = cute.get_scalars(%mul) : !cute.int_tuple<"?{div=2}">
          %27 = arith.cmpi sgt, %26, %25 : i32
          %28:4 = scf.if %27 -> (i32, i32, i32, i32) {
            %32 = arith.floordivsi %25, %c2_i32 : i32
            %33 = arith.muli %32, %c2_i32 : i32
            %34 = arith.subi %25, %33 : i32
            %35 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %32, %arg18 : (i32, i32) -> i32
            %36 = arith.shrui %35, %arg19 : i32
            %37 = arith.cmpi eq, %arg18, %c0_i32 : i32
            %38 = arith.select %37, %32, %36 : i32
            %39 = arith.muli %38, %arg20 : i32
            %40 = arith.subi %32, %39 : i32
            %41 = arith.muli %38, %c2_i32 : i32
            %42 = arith.addi %41, %34 : i32
            %43 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %40, %arg28 : (i32, i32) -> i32
            %44 = arith.shrui %43, %arg29 : i32
            %45 = arith.cmpi eq, %arg28, %c0_i32 : i32
            %46 = arith.select %45, %40, %44 : i32
            %47 = arith.muli %46, %arg30 : i32
            %48 = arith.subi %40, %47 : i32
            %49 = arith.cmpi slt, %46, %arg21 : i32
            %50:2 = scf.if %49 -> (i32, i32) {
              %62 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %48, %arg22 : (i32, i32) -> i32
              %63 = arith.shrui %62, %arg23 : i32
              %64 = arith.cmpi eq, %arg22, %c0_i32 : i32
              %65 = arith.select %64, %48, %63 : i32
              %66 = arith.muli %65, %arg24 : i32
              %67 = arith.subi %48, %66 : i32
              scf.yield %65, %67 : i32, i32
            } else {
              %62 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %48, %arg25 : (i32, i32) -> i32
              %63 = arith.shrui %62, %arg26 : i32
              %64 = arith.cmpi eq, %arg25, %c0_i32 : i32
              %65 = arith.select %64, %48, %63 : i32
              %66 = arith.muli %65, %arg27 : i32
              %67 = arith.subi %48, %66 : i32
              scf.yield %65, %67 : i32, i32
            }
            %51 = arith.remsi %46, %c2_i32 : i32
            %52 = arith.cmpi eq, %51, %c1_i32 : i32
            %53 = scf.if %52 -> (i32) {
              %62 = arith.cmpi eq, %arg17, %c0_i32 : i32
              %63 = arith.select %62, %arg15, %arg14 : i32
              %64 = arith.subi %63, %c1_i32 : i32
              %65 = arith.subi %64, %50#0 : i32
              scf.yield %65 : i32
            } else {
              scf.yield %50#0 : i32
            }
            %54 = arith.muli %46, %arg24 : i32
            %55 = arith.addi %54, %50#1 : i32
            %56 = arith.cmpi eq, %arg17, %c1_i32 : i32
            %57 = arith.select %56, %53, %55 : i32
            %58 = arith.select %56, %55, %53 : i32
            %59 = arith.floordivsi %42, %c2_i32 : i32
            %60 = arith.muli %59, %c2_i32 : i32
            %61 = arith.subi %42, %60 : i32
            scf.yield %61, %57, %58, %59 : i32, i32, i32, i32
          } else {
            scf.yield %c0_i32, %c0_i32, %c0_i32, %c0_i32 : i32, i32, i32, i32
          }
          %29:12 = scf.while (%arg31 = %c0_i32, %arg32 = %c0_i32, %arg33 = %c1_i32, %arg34 = %28#1, %arg35 = %28#2, %arg36 = %28#0, %arg37 = %28#3, %arg38 = %27, %arg39 = %25, %arg40 = %c0_i32, %arg41 = %c0_i32, %arg42 = %c0_i32, %arg43 = %c0_i32) : (i32, i32, i32, i32, i32, i32, i32, i1, i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
            scf.condition(%arg38) %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg39, %arg40, %arg41, %arg42, %arg43 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
          } do {
          ^bb0(%arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32, %arg40: i32, %arg41: i32, %arg42: i32):
            iket.range_push("tma_load")
            %coord = cute.make_coord(%arg37) : (i32) -> !cute.coord<"(_,_,?)">
            %slice = cute.slice(%arg2, %coord) : !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute.coord<"(_,_,?)">
            %tile = cute.make_tile() : () -> !cute.tile<"[128:1;64:1]">
            %coord_29 = cute.make_coord(%arg34) : (i32) -> !cute.coord<"(?,_)">
            %tiled_view = cute.local_tile(%slice, %tile, %coord_29) : (!cute.coord_tensor<"(0,0,?)", "(?,?):(1@1,1@0)">, !cute.tile<"[128:1;64:1]">, !cute.coord<"(?,_)">) -> !cute.coord_tensor<"(0,?{div=128},?)", "(128,64,?):(1@1,1@0,64@0)">
            %grouped = cute.group_modes(%view_18) <0, 2> : (!memref_smem_bf16) -> !memref_smem_bf16_2
            %grouped_30 = cute.group_modes(%tiled_view) <0, 2> : (!cute.coord_tensor<"(0,?{div=128},?)", "(128,64,?):(1@1,1@0,64@0)">) -> !cute.coord_tensor<"(0,?{div=128},?)", "((128,64),?):((1@1,1@0),64@0)">
            %slice_31 = cute.slice(%arg4, %coord) : !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute.coord<"(_,_,?)">
            %coord_32 = cute.make_coord(%arg35) : (i32) -> !cute.coord<"(?,_)">
            %tiled_view_33 = cute.local_tile(%slice_31, %tile, %coord_32) : (!cute.coord_tensor<"(0,0,?)", "(?,?):(1@1,1@0)">, !cute.tile<"[128:1;64:1]">, !cute.coord<"(?,_)">) -> !cute.coord_tensor<"(0,?{div=128},?)", "(128,64,?):(1@1,1@0,64@0)">
            %grouped_34 = cute.group_modes(%view_20) <0, 2> : (!memref_smem_bf16) -> !memref_smem_bf16_2
            %grouped_35 = cute.group_modes(%tiled_view_33) <0, 2> : (!cute.coord_tensor<"(0,?{div=128},?)", "(128,64,?):(1@1,1@0,64@0)">) -> !cute.coord_tensor<"(0,?{div=128},?)", "((128,64),?):((1@1,1@0),64@0)">
            %int_tuple_36 = cute.make_int_tuple(%20) : (i32) -> !cute.int_tuple<"?">
            %tile_37 = cute.make_tile() : () -> !cute.tile<"64:1">
            %shp = cute.ceil_div(%int_tuple_36, %tile_37) : !cute.int_tuple<"?">, !cute.tile<"64:1">
            %e0_38 = cute.get_leaves(%shp) : !cute.int_tuple<"?">
            %div = cute.tuple_div(%e0_38, %int_tuple_26) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?">
            %mul_39 = cute.tuple_mul(%div, %int_tuple_26) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?{div=2}">
            %sub = cute.tuple_sub(%e0_38, %mul_39) : (!cute.int_tuple<"?">, !cute.int_tuple<"?{div=2}">) -> !cute.int_tuple<"?">
            %32 = cute.get_scalars(%sub) : !cute.int_tuple<"?">
            %int_tuple_40 = cute.make_int_tuple(%arg36) : (i32) -> !cute.int_tuple<"?">
            %mul_41 = cute.tuple_mul(%int_tuple_40, %div) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
            %33 = arith.minsi %arg36, %32 : i32
            %int_tuple_42 = cute.make_int_tuple(%33) : (i32) -> !cute.int_tuple<"?">
            %add = cute.tuple_add(%mul_41, %int_tuple_42) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
            %34 = arith.cmpi sgt, %32, %arg36 : i32
            %35 = arith.extui %34 : i1 to i32
            %int_tuple_43 = cute.make_int_tuple(%35) : (i32) -> !cute.int_tuple<"?">
            %add_44 = cute.tuple_add(%div, %int_tuple_43) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
            %36 = cute.get_scalars(%add_44) : !cute.int_tuple<"?">
            %37 = arith.cmpi sgt, %36, %c0_i32 : i32
            %38 = scf.if %37 -> (i1) {
              %int_tuple_52 = cute.make_int_tuple(%arg32) : (i32) -> !cute.int_tuple<"?">
              %ptr_53 = cute.add_offset(%ptr_8, %int_tuple_52) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %67 = builtin.unrealized_conversion_cast %ptr_53 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %68 = nvvm.mbarrier.wait.parity %67, %arg33 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
              scf.yield %68 : i1
            } else {
              scf.yield %true : i1
            }
            %39:4 = scf.for %arg43 = %c0_i32 to %36 step %c1_i32 iter_args(%arg44 = %38, %arg45 = %arg31, %arg46 = %arg32, %arg47 = %arg33) -> (i1, i32, i32, i32)  : i32 {
              %67 = arith.extui %arg44 : i1 to i32
              %68 = arith.cmpi eq, %67, %c0_i32 : i32
              scf.if %68 {
                %int_tuple_97 = cute.make_int_tuple(%arg46) : (i32) -> !cute.int_tuple<"?">
                %ptr_98 = cute.add_offset(%ptr_8, %int_tuple_97) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
                %82 = builtin.unrealized_conversion_cast %ptr_98 : !cute.ptr<i64, smem> to !llvm.ptr<3>
                nvvm.mbarrier.try_wait.parity %82, %arg47, %c10000000_i32 : !llvm.ptr<3>, i32, i32
              }
              %69 = nvvm.elect.sync -> i1
              scf.if %69 {
                %int_tuple_97 = cute.make_int_tuple(%arg46) : (i32) -> !cute.int_tuple<"?">
                %ptr_98 = cute.add_offset(%iter_6, %int_tuple_97) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
                %82 = builtin.unrealized_conversion_cast %ptr_98 : !cute.ptr<i64, smem> to !llvm.ptr<3>
                %83 = nvvm.mbarrier.arrive.expect_tx %82, %c32768_i32 : !llvm.ptr<3>, i32 -> i64
              }
              %int_tuple_52 = cute.make_int_tuple(%arg46) : (i32) -> !cute.int_tuple<"?">
              %ptr_53 = cute.add_offset(%iter_6, %int_tuple_52) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %int_tuple_54 = cute.make_int_tuple(%arg43) : (i32) -> !cute.int_tuple<"?">
              %add_55 = cute.tuple_add(%add, %int_tuple_54) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
              %coord_56 = cute.make_coord(%add_55) : (!cute.int_tuple<"?">) -> !cute.coord<"(_,?)">
              %slice_57 = cute.slice(%grouped_30, %coord_56) : !cute.coord_tensor<"(0,?{div=128},?)", "((128,64),?):((1@1,1@0),64@0)">, !cute.coord<"(_,?)">
              %coord_58 = cute.make_coord(%arg46) : (i32) -> !cute.coord<"(_,?)">
              %slice_59 = cute.slice(%grouped, %coord_58) : !memref_smem_bf16_2, !cute.coord<"(_,?)">
              %shape_60 = cute.make_shape() : () -> !cute.shape<"1">
              %lay_61 = cute.make_layout(%shape_60) : !cute.layout<"1:0">
              %coord_62 = cute.make_coord() : () -> !cute.coord<"0">
              %res_smem_tensor, %res_target_tensors = cute_nvgpu.atom.tma_partition(%arg1, %coord_62, %lay_61, %slice_59, %slice_57) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord<"0">, !cute.layout<"1:0">, !memref_smem_bf16_3, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "((128,64)):((1@1,1@0))">) -> (!memref_smem_bf16_4, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">)
              %filtered = cute.filter_zeros(%res_smem_tensor) : !memref_smem_bf16_4
              %iter_63 = cute.get_iter(%filtered) : !memref_smem_bf16_4
              %filtered_64 = cute.filter_zeros(%res_target_tensors) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %iter_65 = cute.get_iter(%filtered_64) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %tup = cute.deref_arith_tuple_iter(%iter_65) : !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
              %e0_66, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=64},?{div=128},?)">
              %lay_67 = cute.get_layout(%filtered_64) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %lay_68 = cute.get_layout(%filtered) : !memref_smem_bf16_4
              %append = cute.append_to_rank<2> (%lay_67, %lay_61) : !cute.layout<"(((64,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
              %int_tuple_69 = cute.make_int_tuple(%e0_66, %e1, %e2) : (!cute.int_tuple<"?{div=64}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=64},?{div=128},?)">
              %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_69) : (!cute.int_tuple<"(?{div=64},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
              %view_70 = cute.make_view(%int_tup_iter, %append) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),1):(((1@0,1@1),0),0)">
              %grouped_71 = cute.group_modes(%view_70) <1, 2> : (!cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),(1)):(((1@0,1@1),0),(0))">
              %append_72 = cute.append_to_rank<2> (%lay_68, %lay_61) : !cute.layout<"((8192,1)):((1,0))">, !cute.layout<"1:0">
              %view_73 = cute.make_view(%iter_63, %append_72) : !memref_smem_bf16_5
              %grouped_74 = cute.group_modes(%view_73) <1, 2> : (!memref_smem_bf16_5) -> !memref_smem_bf16_6
              %70 = cute_nvgpu.atom.make_exec_tma(%arg1) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> !cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>
              %71 = cute_nvgpu.atom.set_value<tma_bar>(%70, %ptr_53) : (!cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>, !cute.ptr<i64, smem>)
              cute.copy(%71, %grouped_71, %grouped_74) {from_block_api, multicast_layout = #cute.layout<"(1,1):(1,0)">} : (!cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),(1)):(((1@0,1@1),0),(0))">, !memref_smem_bf16_6)
              %slice_75 = cute.slice(%grouped_35, %coord_56) : !cute.coord_tensor<"(0,?{div=128},?)", "((128,64),?):((1@1,1@0),64@0)">, !cute.coord<"(_,?)">
              %slice_76 = cute.slice(%grouped_34, %coord_58) : !memref_smem_bf16_2, !cute.coord<"(_,?)">
              %res_smem_tensor_77, %res_target_tensors_78 = cute_nvgpu.atom.tma_partition(%arg3, %coord_62, %lay_61, %slice_76, %slice_75) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord<"0">, !cute.layout<"1:0">, !memref_smem_bf16_3, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "((128,64)):((1@1,1@0))">) -> (!memref_smem_bf16_4, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">)
              %filtered_79 = cute.filter_zeros(%res_smem_tensor_77) : !memref_smem_bf16_4
              %iter_80 = cute.get_iter(%filtered_79) : !memref_smem_bf16_4
              %filtered_81 = cute.filter_zeros(%res_target_tensors_78) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %iter_82 = cute.get_iter(%filtered_81) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %tup_83 = cute.deref_arith_tuple_iter(%iter_82) : !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
              %e0_84, %e1_85, %e2_86 = cute.get_leaves(%tup_83) : !cute.int_tuple<"(?{div=64},?{div=128},?)">
              %lay_87 = cute.get_layout(%filtered_81) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1)):(((1@0,1@1),0))">
              %lay_88 = cute.get_layout(%filtered_79) : !memref_smem_bf16_4
              %append_89 = cute.append_to_rank<2> (%lay_87, %lay_61) : !cute.layout<"(((64,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
              %int_tuple_90 = cute.make_int_tuple(%e0_84, %e1_85, %e2_86) : (!cute.int_tuple<"?{div=64}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=64},?{div=128},?)">
              %int_tup_iter_91 = cute.make_arith_tuple_iter(%int_tuple_90) : (!cute.int_tuple<"(?{div=64},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
              %view_92 = cute.make_view(%int_tup_iter_91, %append_89) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),1):(((1@0,1@1),0),0)">
              %grouped_93 = cute.group_modes(%view_92) <1, 2> : (!cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),(1)):(((1@0,1@1),0),(0))">
              %append_94 = cute.append_to_rank<2> (%lay_88, %lay_61) : !cute.layout<"((8192,1)):((1,0))">, !cute.layout<"1:0">
              %view_95 = cute.make_view(%iter_80, %append_94) : !memref_smem_bf16_5
              %grouped_96 = cute.group_modes(%view_95) <1, 2> : (!memref_smem_bf16_5) -> !memref_smem_bf16_6
              %72 = cute_nvgpu.atom.make_exec_tma(%arg3) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> !cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>
              %73 = cute_nvgpu.atom.set_value<tma_bar>(%72, %ptr_53) : (!cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>, !cute.ptr<i64, smem>)
              cute.copy(%73, %grouped_93, %grouped_96) {from_block_api, multicast_layout = #cute.layout<"(1,1):(0,1)">} : (!cute_nvgpu.atom.tma_load<bf16, copy_bits = 131072, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">>, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((64,128),1),(1)):(((1@0,1@1),0),(0))">, !memref_smem_bf16_6)
              %74 = arith.addi %arg46, %c1_i32 : i32
              %75 = arith.addi %arg45, %c1_i32 : i32
              %76 = arith.cmpi eq, %74, %c6_i32 : i32
              %77 = arith.select %76, %c0_i32, %74 : i32
              %78 = scf.if %76 -> (i32) {
                %82 = arith.xori %arg47, %c1_i32 : i32
                scf.yield %82 : i32
              } else {
                scf.yield %arg47 : i32
              }
              %79 = arith.addi %arg43, %c1_i32 : i32
              %80 = arith.cmpi sgt, %36, %79 : i32
              %81 = scf.if %80 -> (i1) {
                %int_tuple_97 = cute.make_int_tuple(%77) : (i32) -> !cute.int_tuple<"?">
                %ptr_98 = cute.add_offset(%ptr_8, %int_tuple_97) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
                %82 = builtin.unrealized_conversion_cast %ptr_98 : !cute.ptr<i64, smem> to !llvm.ptr<3>
                %83 = nvvm.mbarrier.wait.parity %82, %78 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
                scf.yield %83 : i1
              } else {
                scf.yield %true : i1
              }
              scf.yield %81, %75, %77, %78 : i1, i32, i32, i32
            } {loop_annotation = #loop_annotation}
            iket.range_pop
            %40 = arith.addi %arg39, %c1_i32 : i32
            %41 = nvvm.read.ptx.sreg.nctaid.z : i32
            %42 = arith.addi %arg38, %41 : i32
            %43 = arith.cmpi sgt, %26, %42 : i32
            %44:4 = scf.if %43 -> (i32, i32, i32, i32) {
              %67 = arith.floordivsi %42, %c2_i32 : i32
              %68 = arith.muli %67, %c2_i32 : i32
              %69 = arith.subi %42, %68 : i32
              %70 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %67, %arg18 : (i32, i32) -> i32
              %71 = arith.shrui %70, %arg19 : i32
              %72 = arith.cmpi eq, %arg18, %c0_i32 : i32
              %73 = arith.select %72, %67, %71 : i32
              %74 = arith.muli %73, %arg20 : i32
              %75 = arith.subi %67, %74 : i32
              %76 = arith.muli %73, %c2_i32 : i32
              %77 = arith.addi %76, %69 : i32
              %78 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %75, %arg28 : (i32, i32) -> i32
              %79 = arith.shrui %78, %arg29 : i32
              %80 = arith.cmpi eq, %arg28, %c0_i32 : i32
              %81 = arith.select %80, %75, %79 : i32
              %82 = arith.muli %81, %arg30 : i32
              %83 = arith.subi %75, %82 : i32
              %84 = arith.cmpi slt, %81, %arg21 : i32
              %85:2 = scf.if %84 -> (i32, i32) {
                %97 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %83, %arg22 : (i32, i32) -> i32
                %98 = arith.shrui %97, %arg23 : i32
                %99 = arith.cmpi eq, %arg22, %c0_i32 : i32
                %100 = arith.select %99, %83, %98 : i32
                %101 = arith.muli %100, %arg24 : i32
                %102 = arith.subi %83, %101 : i32
                scf.yield %100, %102 : i32, i32
              } else {
                %97 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %83, %arg25 : (i32, i32) -> i32
                %98 = arith.shrui %97, %arg26 : i32
                %99 = arith.cmpi eq, %arg25, %c0_i32 : i32
                %100 = arith.select %99, %83, %98 : i32
                %101 = arith.muli %100, %arg27 : i32
                %102 = arith.subi %83, %101 : i32
                scf.yield %100, %102 : i32, i32
              }
              %86 = arith.remsi %81, %c2_i32 : i32
              %87 = arith.cmpi eq, %86, %c1_i32 : i32
              %88 = scf.if %87 -> (i32) {
                %97 = arith.cmpi eq, %arg17, %c0_i32 : i32
                %98 = arith.select %97, %arg15, %arg14 : i32
                %99 = arith.subi %98, %c1_i32 : i32
                %100 = arith.subi %99, %85#0 : i32
                scf.yield %100 : i32
              } else {
                scf.yield %85#0 : i32
              }
              %89 = arith.muli %81, %arg24 : i32
              %90 = arith.addi %89, %85#1 : i32
              %91 = arith.cmpi eq, %arg17, %c1_i32 : i32
              %92 = arith.select %91, %88, %90 : i32
              %93 = arith.select %91, %90, %88 : i32
              %94 = arith.floordivsi %77, %c2_i32 : i32
              %95 = arith.muli %94, %c2_i32 : i32
              %96 = arith.subi %77, %95 : i32
              scf.yield %96, %92, %93, %94 : i32, i32, i32, i32
            } else {
              scf.yield %c0_i32, %c0_i32, %c0_i32, %c0_i32 : i32, i32, i32, i32
            }
            %45 = arith.xori %arg42, %c1_i32 : i32
            %int_tuple_45 = cute.make_int_tuple(%arg41) : (i32) -> !cute.int_tuple<"?">
            %ptr_46 = cute.add_offset(%ptr_13, %int_tuple_45) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %46 = builtin.unrealized_conversion_cast %ptr_46 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %46, %45, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            %47 = arith.muli %44#3, %c2_i32 : i32
            %48 = arith.addi %47, %44#0 : i32
            %49 = arith.extui %43 : i1 to i32
            %50 = nvvm.read.ptx.sreg.laneid : i32
            %51 = arith.cmpi slt, %50, %c1_i32 : i32
            scf.if %51 {
              %coord_52 = cute.make_coord(%arg41) : (i32) -> !cute.coord<"(0,?)">
              cute.memref.store(%view, %coord_52, %44#1) : (!memref_smem_i32, !cute.coord<"(0,?)">, i32) -> ()
              %coord_53 = cute.make_coord(%arg41) : (i32) -> !cute.coord<"(1,?)">
              cute.memref.store(%view, %coord_53, %44#2) : (!memref_smem_i32, !cute.coord<"(1,?)">, i32) -> ()
              %coord_54 = cute.make_coord(%arg41) : (i32) -> !cute.coord<"(2,?)">
              cute.memref.store(%view, %coord_54, %48) : (!memref_smem_i32, !cute.coord<"(2,?)">, i32) -> ()
              %coord_55 = cute.make_coord(%arg41) : (i32) -> !cute.coord<"(3,?)">
              cute.memref.store(%view, %coord_55, %49) : (!memref_smem_i32, !cute.coord<"(3,?)">, i32) -> ()
              %ptr_56 = cute.add_offset(%iter_11, %int_tuple_45) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %67 = builtin.unrealized_conversion_cast %ptr_56 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %68 = nvvm.mbarrier.arrive %67, %c1_i32 : !llvm.ptr<3> -> i64
            }
            iket.range_push("fetch_wait")
            %ptr_47 = cute.add_offset(%iter_11, %int_tuple_45) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %52 = builtin.unrealized_conversion_cast %ptr_47 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %52, %arg42, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            iket.range_pop
            iket.range_push("fetch_decode")
            %coord_48 = cute.make_coord(%arg41) : (i32) -> !cute.coord<"(0,?)">
            %53 = cute.memref.load(%view, %coord_48) : (!memref_smem_i32, !cute.coord<"(0,?)">) -> i32
            %coord_49 = cute.make_coord(%arg41) : (i32) -> !cute.coord<"(1,?)">
            %54 = cute.memref.load(%view, %coord_49) : (!memref_smem_i32, !cute.coord<"(1,?)">) -> i32
            %coord_50 = cute.make_coord(%arg41) : (i32) -> !cute.coord<"(2,?)">
            %55 = cute.memref.load(%view, %coord_50) : (!memref_smem_i32, !cute.coord<"(2,?)">) -> i32
            %coord_51 = cute.make_coord(%arg41) : (i32) -> !cute.coord<"(3,?)">
            %56 = cute.memref.load(%view, %coord_51) : (!memref_smem_i32, !cute.coord<"(3,?)">) -> i32
            nvvm.bar.warp.sync %c-1_i32 : i32
            %57 = nvvm.elect.sync -> i1
            scf.if %57 {
              %67 = nvvm.mbarrier.arrive %46, %c1_i32 : !llvm.ptr<3> -> i64
            }
            %58 = arith.addi %arg41, %c1_i32 : i32
            %59 = arith.addi %arg40, %c1_i32 : i32
            %60 = arith.cmpi eq, %58, %c1_i32 : i32
            %61 = arith.select %60, %c0_i32, %58 : i32
            %62 = scf.if %60 -> (i32) {
              scf.yield %45 : i32
            } else {
              scf.yield %arg42 : i32
            }
            %63 = arith.cmpi ne, %56, %c0_i32 : i32
            iket.range_pop
            %64 = arith.floordivsi %55, %c2_i32 : i32
            %65 = arith.muli %64, %c2_i32 : i32
            %66 = arith.subi %55, %65 : i32
            scf.yield %39#1, %39#2, %39#3, %53, %54, %66, %64, %63, %42, %40, %59, %61, %62 : i32, i32, i32, i32, i32, i32, i32, i1, i32, i32, i32, i32, i32
          }
          scf.if %13 {
            %int_tuple_29 = cute.make_int_tuple(%29#1) : (i32) -> !cute.int_tuple<"?">
            %ptr_30 = cute.add_offset(%ptr_8, %int_tuple_29) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %32 = builtin.unrealized_conversion_cast %ptr_30 : !cute.ptr<i64, smem> to !llvm.ptr<3>
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
            %int_tuple_31 = cute.make_int_tuple(%35) : (i32) -> !cute.int_tuple<"?">
            %ptr_32 = cute.add_offset(%ptr_8, %int_tuple_31) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %37 = builtin.unrealized_conversion_cast %ptr_32 : !cute.ptr<i64, smem> to !llvm.ptr<3>
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
            %int_tuple_33 = cute.make_int_tuple(%40) : (i32) -> !cute.int_tuple<"?">
            %ptr_34 = cute.add_offset(%ptr_8, %int_tuple_33) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %42 = builtin.unrealized_conversion_cast %ptr_34 : !cute.ptr<i64, smem> to !llvm.ptr<3>
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
            %int_tuple_35 = cute.make_int_tuple(%45) : (i32) -> !cute.int_tuple<"?">
            %ptr_36 = cute.add_offset(%ptr_8, %int_tuple_35) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %47 = builtin.unrealized_conversion_cast %ptr_36 : !cute.ptr<i64, smem> to !llvm.ptr<3>
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
            %int_tuple_37 = cute.make_int_tuple(%50) : (i32) -> !cute.int_tuple<"?">
            %ptr_38 = cute.add_offset(%ptr_8, %int_tuple_37) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %52 = builtin.unrealized_conversion_cast %ptr_38 : !cute.ptr<i64, smem> to !llvm.ptr<3>
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
            %int_tuple_39 = cute.make_int_tuple(%55) : (i32) -> !cute.int_tuple<"?">
            %ptr_40 = cute.add_offset(%ptr_8, %int_tuple_39) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %57 = builtin.unrealized_conversion_cast %ptr_40 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.try_wait.parity %57, %56, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          }
          %30 = arith.xori %29#11, %c1_i32 : i32
          %int_tuple_27 = cute.make_int_tuple(%29#10) : (i32) -> !cute.int_tuple<"?">
          %ptr_28 = cute.add_offset(%ptr_13, %int_tuple_27) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %31 = builtin.unrealized_conversion_cast %ptr_28 : !cute.ptr<i64, smem> to !llvm.ptr<3>
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
        %shape_23 = cute.make_shape() : () -> !cute.shape<"2">
        %stride = cute.make_stride() : () -> !cute.stride<"128">
        %lay_24 = cute.make_layout(%shape_23, %stride) : !cute.layout<"2:128">
        %coord = cute.make_coord(%27) : (i32) -> !cute.coord<"?">
        %idx = cute.crd2idx(%coord, %lay_24) : (!cute.coord<"?">, !cute.layout<"2:128">) -> !cute.int_tuple<"?{div=128}">
        %e0_25 = cute.get_leaves(%idx) : !cute.int_tuple<"?{div=128}">
        %rmem = cute.memref.alloca() : !memref_rmem_f32
        %coord_26 = cute.make_coord(%e0_25) : (!cute.int_tuple<"?{div=128}">) -> !cute.coord<"?{div=128}">
        %ptn_A = cute.tiled.mma.partition A (%arg0, %view_18, %coord_26) : (!mma_bf16_bf16_f32_64x128x16, !memref_smem_bf16, !cute.coord<"?{div=128}">) -> !memref_smem_bf16_7
        %frg_A = cute.mma.make_fragment A (%arg0, %ptn_A) : (!mma_bf16_bf16_f32_64x128x16, !memref_smem_bf16_7) -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">
        %ptn_B = cute.tiled.mma.partition B (%arg0, %view_20, %coord_26) : (!mma_bf16_bf16_f32_64x128x16, !memref_smem_bf16, !cute.coord<"?{div=128}">) -> !memref_smem_bf16_8
        %frg_B = cute.mma.make_fragment B (%arg0, %ptn_B) : (!mma_bf16_bf16_f32_64x128x16, !memref_smem_bf16_8) -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">
        %28 = nvvm.read.ptx.sreg.ctaid.z : i32
        %int_tuple_27 = cute.make_int_tuple(%arg14, %arg15, %arg16) : (i32, i32, i32) -> !cute.int_tuple<"(?,?,?)">
        %sz_28 = cute.size(%int_tuple_27) : (!cute.int_tuple<"(?,?,?)">) -> !cute.int_tuple<"?">
        %e0_29 = cute.get_leaves(%sz_28) : !cute.int_tuple<"?">
        %int_tuple_30 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
        %mul = cute.tuple_mul(%e0_29, %int_tuple_30) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?{div=2}">
        %29 = cute.get_scalars(%mul) : !cute.int_tuple<"?{div=2}">
        %30 = arith.cmpi sgt, %29, %28 : i32
        %31:4 = scf.if %30 -> (i32, i32, i32, i32) {
          %33 = arith.floordivsi %28, %c2_i32 : i32
          %34 = arith.muli %33, %c2_i32 : i32
          %35 = arith.subi %28, %34 : i32
          %36 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %33, %arg18 : (i32, i32) -> i32
          %37 = arith.shrui %36, %arg19 : i32
          %38 = arith.cmpi eq, %arg18, %c0_i32 : i32
          %39 = arith.select %38, %33, %37 : i32
          %40 = arith.muli %39, %arg20 : i32
          %41 = arith.subi %33, %40 : i32
          %42 = arith.muli %39, %c2_i32 : i32
          %43 = arith.addi %42, %35 : i32
          %44 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %41, %arg28 : (i32, i32) -> i32
          %45 = arith.shrui %44, %arg29 : i32
          %46 = arith.cmpi eq, %arg28, %c0_i32 : i32
          %47 = arith.select %46, %41, %45 : i32
          %48 = arith.muli %47, %arg30 : i32
          %49 = arith.subi %41, %48 : i32
          %50 = arith.cmpi slt, %47, %arg21 : i32
          %51:2 = scf.if %50 -> (i32, i32) {
            %63 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %49, %arg22 : (i32, i32) -> i32
            %64 = arith.shrui %63, %arg23 : i32
            %65 = arith.cmpi eq, %arg22, %c0_i32 : i32
            %66 = arith.select %65, %49, %64 : i32
            %67 = arith.muli %66, %arg24 : i32
            %68 = arith.subi %49, %67 : i32
            scf.yield %66, %68 : i32, i32
          } else {
            %63 = llvm.inline_asm asm_dialect = att "mul.hi.u32 $0, $1, $2;", "=r,r,r" %49, %arg25 : (i32, i32) -> i32
            %64 = arith.shrui %63, %arg26 : i32
            %65 = arith.cmpi eq, %arg25, %c0_i32 : i32
            %66 = arith.select %65, %49, %64 : i32
            %67 = arith.muli %66, %arg27 : i32
            %68 = arith.subi %49, %67 : i32
            scf.yield %66, %68 : i32, i32
          }
          %52 = arith.remsi %47, %c2_i32 : i32
          %53 = arith.cmpi eq, %52, %c1_i32 : i32
          %54 = scf.if %53 -> (i32) {
            %63 = arith.cmpi eq, %arg17, %c0_i32 : i32
            %64 = arith.select %63, %arg15, %arg14 : i32
            %65 = arith.subi %64, %c1_i32 : i32
            %66 = arith.subi %65, %51#0 : i32
            scf.yield %66 : i32
          } else {
            scf.yield %51#0 : i32
          }
          %55 = arith.muli %47, %arg24 : i32
          %56 = arith.addi %55, %51#1 : i32
          %57 = arith.cmpi eq, %arg17, %c1_i32 : i32
          %58 = arith.select %57, %54, %56 : i32
          %59 = arith.select %57, %56, %54 : i32
          %60 = arith.floordivsi %43, %c2_i32 : i32
          %61 = arith.muli %60, %c2_i32 : i32
          %62 = arith.subi %43, %61 : i32
          scf.yield %62, %58, %59, %60 : i32, i32, i32, i32
        } else {
          scf.yield %c0_i32, %c0_i32, %c0_i32, %c0_i32 : i32, i32, i32, i32
        }
        %32:11 = scf.while (%arg31 = %c0_i32, %arg32 = %c0_i32, %arg33 = %c0_i32, %arg34 = %31#1, %arg35 = %31#2, %arg36 = %31#0, %arg37 = %31#3, %arg38 = %30, %arg39 = %c0_i32, %arg40 = %c0_i32, %arg41 = %c0_i32, %arg42 = %c0_i32) : (i32, i32, i32, i32, i32, i32, i32, i1, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
          scf.condition(%arg38) %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg39, %arg40, %arg41, %arg42 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
        } do {
        ^bb0(%arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32, %arg40: i32, %arg41: i32):
          %int_tuple_31 = cute.make_int_tuple(%20) : (i32) -> !cute.int_tuple<"?">
          %tile = cute.make_tile() : () -> !cute.tile<"64:1">
          %shp = cute.ceil_div(%int_tuple_31, %tile) : !cute.int_tuple<"?">, !cute.tile<"64:1">
          %e0_32 = cute.get_leaves(%shp) : !cute.int_tuple<"?">
          %div = cute.tuple_div(%e0_32, %int_tuple_30) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?">
          %mul_33 = cute.tuple_mul(%div, %int_tuple_30) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?{div=2}">
          %sub = cute.tuple_sub(%e0_32, %mul_33) : (!cute.int_tuple<"?">, !cute.int_tuple<"?{div=2}">) -> !cute.int_tuple<"?">
          %33 = cute.get_scalars(%sub) : !cute.int_tuple<"?">
          %34 = arith.cmpi sgt, %33, %arg36 : i32
          %35 = arith.extui %34 : i1 to i32
          %int_tuple_34 = cute.make_int_tuple(%35) : (i32) -> !cute.int_tuple<"?">
          %add = cute.tuple_add(%div, %int_tuple_34) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
          %36 = cute.get_scalars(%add) : !cute.int_tuple<"?">
          iket.range_push("mma")
          %37 = arith.minsi %36, %c1_i32 : i32
          %38 = arith.cmpi sgt, %36, %c0_i32 : i32
          %39 = scf.if %38 -> (i1) {
            %int_tuple_75 = cute.make_int_tuple(%arg32) : (i32) -> !cute.int_tuple<"?">
            %ptr_76 = cute.add_offset(%iter_6, %int_tuple_75) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %62 = builtin.unrealized_conversion_cast %ptr_76 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            %63 = nvvm.mbarrier.wait.parity %62, %arg33 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
            scf.yield %63 : i1
          } else {
            scf.yield %true : i1
          }
          %40:5 = scf.for %arg42 = %c0_i32 to %37 step %c1_i32 iter_args(%arg43 = %true, %arg44 = %39, %arg45 = %arg31, %arg46 = %arg32, %arg47 = %arg33) -> (i1, i1, i32, i32, i32)  : i32 {
            %62 = arith.extui %arg44 : i1 to i32
            %63 = arith.cmpi eq, %62, %c0_i32 : i32
            scf.if %63 {
              %int_tuple_91 = cute.make_int_tuple(%arg46) : (i32) -> !cute.int_tuple<"?">
              %ptr_92 = cute.add_offset(%iter_6, %int_tuple_91) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %77 = builtin.unrealized_conversion_cast %ptr_92 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.mbarrier.try_wait.parity %77, %arg47, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            }
            %coord_75 = cute.make_coord(%arg46) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_76 = cute.slice(%frg_A, %coord_75) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">, !cute.coord<"(_,_,_,?)">
            %slice_77 = cute.slice(%frg_B, %coord_75) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">, !cute.coord<"(_,_,_,?)">
            nvvm.wgmma.fence.aligned
            %atom_78 = cute.make_atom(%false) : (i1) -> !cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>
            %64 = arith.cmpi eq, %arg43, %false : i1
            %65 = cute_nvgpu.atom.set_value<accum_c>(%atom_78, %64) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_79 = cute.make_coord() : () -> !cute.coord<"(_,_,0)">
            %slice_80 = cute.slice(%slice_76, %coord_79) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,0)">
            %slice_81 = cute.slice(%slice_77, %coord_79) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,0)">
            cute.gemm(%65, %rmem, %slice_80, %slice_81, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %66 = cute_nvgpu.atom.set_value<accum_c>(%65, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_82 = cute.make_coord() : () -> !cute.coord<"(_,_,1)">
            %slice_83 = cute.slice(%slice_76, %coord_82) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,1)">
            %slice_84 = cute.slice(%slice_77, %coord_82) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,1)">
            cute.gemm(%66, %rmem, %slice_83, %slice_84, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %67 = cute_nvgpu.atom.set_value<accum_c>(%66, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_85 = cute.make_coord() : () -> !cute.coord<"(_,_,2)">
            %slice_86 = cute.slice(%slice_76, %coord_85) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,2)">
            %slice_87 = cute.slice(%slice_77, %coord_85) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,2)">
            cute.gemm(%67, %rmem, %slice_86, %slice_87, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %68 = cute_nvgpu.atom.set_value<accum_c>(%67, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_88 = cute.make_coord() : () -> !cute.coord<"(_,_,3)">
            %slice_89 = cute.slice(%slice_76, %coord_88) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,3)">
            %slice_90 = cute.slice(%slice_77, %coord_88) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,3)">
            cute.gemm(%68, %rmem, %slice_89, %slice_90, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            nvvm.wgmma.commit.group.sync.aligned
            %69 = arith.addi %arg46, %c1_i32 : i32
            %70 = arith.addi %arg45, %c1_i32 : i32
            %71 = arith.cmpi eq, %69, %c6_i32 : i32
            %72 = arith.select %71, %c0_i32, %69 : i32
            %73 = scf.if %71 -> (i32) {
              %77 = arith.xori %arg47, %c1_i32 : i32
              scf.yield %77 : i32
            } else {
              scf.yield %arg47 : i32
            }
            %74 = arith.addi %arg42, %c1_i32 : i32
            %75 = arith.cmpi sgt, %36, %74 : i32
            %76 = scf.if %75 -> (i1) {
              %int_tuple_91 = cute.make_int_tuple(%72) : (i32) -> !cute.int_tuple<"?">
              %ptr_92 = cute.add_offset(%iter_6, %int_tuple_91) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %77 = builtin.unrealized_conversion_cast %ptr_92 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %78 = nvvm.mbarrier.wait.parity %77, %73 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
              scf.yield %78 : i1
            } else {
              scf.yield %true : i1
            }
            scf.yield %false, %76, %70, %72, %73 : i1, i1, i32, i32, i32
          }
          %41:8 = scf.for %arg42 = %37 to %36 step %c1_i32 iter_args(%arg43 = %40#0, %arg44 = %40#1, %arg45 = %40#2, %arg46 = %40#3, %arg47 = %40#4, %arg48 = %arg31, %arg49 = %arg32, %arg50 = %arg33) -> (i1, i1, i32, i32, i32, i32, i32, i32)  : i32 {
            %62 = arith.extui %arg44 : i1 to i32
            %63 = arith.cmpi eq, %62, %c0_i32 : i32
            scf.if %63 {
              %int_tuple_91 = cute.make_int_tuple(%arg46) : (i32) -> !cute.int_tuple<"?">
              %ptr_92 = cute.add_offset(%iter_6, %int_tuple_91) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %82 = builtin.unrealized_conversion_cast %ptr_92 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.mbarrier.try_wait.parity %82, %arg47, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            }
            %coord_75 = cute.make_coord(%arg46) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_76 = cute.slice(%frg_A, %coord_75) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">, !cute.coord<"(_,_,_,?)">
            %slice_77 = cute.slice(%frg_B, %coord_75) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,(1,6)):(0,0,2,(0,1024))">, !cute.coord<"(_,_,_,?)">
            nvvm.wgmma.fence.aligned
            %atom_78 = cute.make_atom(%false) : (i1) -> !cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>
            %64 = arith.cmpi eq, %arg43, %false : i1
            %65 = cute_nvgpu.atom.set_value<accum_c>(%atom_78, %64) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_79 = cute.make_coord() : () -> !cute.coord<"(_,_,0)">
            %slice_80 = cute.slice(%slice_76, %coord_79) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,0)">
            %slice_81 = cute.slice(%slice_77, %coord_79) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,0)">
            cute.gemm(%65, %rmem, %slice_80, %slice_81, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %66 = cute_nvgpu.atom.set_value<accum_c>(%65, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_82 = cute.make_coord() : () -> !cute.coord<"(_,_,1)">
            %slice_83 = cute.slice(%slice_76, %coord_82) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,1)">
            %slice_84 = cute.slice(%slice_77, %coord_82) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,1)">
            cute.gemm(%66, %rmem, %slice_83, %slice_84, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %67 = cute_nvgpu.atom.set_value<accum_c>(%66, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_85 = cute.make_coord() : () -> !cute.coord<"(_,_,2)">
            %slice_86 = cute.slice(%slice_76, %coord_85) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,2)">
            %slice_87 = cute.slice(%slice_77, %coord_85) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,2)">
            cute.gemm(%67, %rmem, %slice_86, %slice_87, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            %68 = cute_nvgpu.atom.set_value<accum_c>(%67, %true) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, i1)
            %coord_88 = cute.make_coord() : () -> !cute.coord<"(_,_,3)">
            %slice_89 = cute.slice(%slice_76, %coord_88) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,3)">
            %slice_90 = cute.slice(%slice_77, %coord_88) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4):(0,0,2)">, !cute.coord<"(_,_,3)">
            cute.gemm(%68, %rmem, %slice_89, %slice_90, %rmem) : (!cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>, !memref_rmem_f32, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_rmem_f32)
            nvvm.wgmma.commit.group.sync.aligned
            nvvm.wgmma.wait.group.sync.aligned 1
            scf.if %16 {
              %int_tuple_91 = cute.make_int_tuple(%arg49) : (i32) -> !cute.int_tuple<"?">
              %ptr_92 = cute.add_offset(%ptr_8, %int_tuple_91) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %82 = builtin.unrealized_conversion_cast %ptr_92 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %83 = nvvm.mbarrier.arrive %82, %c1_i32 : !llvm.ptr<3> -> i64
            }
            %69 = arith.addi %arg46, %c1_i32 : i32
            %70 = arith.addi %arg45, %c1_i32 : i32
            %71 = arith.cmpi eq, %69, %c6_i32 : i32
            %72 = arith.select %71, %c0_i32, %69 : i32
            %73 = scf.if %71 -> (i32) {
              %82 = arith.xori %arg47, %c1_i32 : i32
              scf.yield %82 : i32
            } else {
              scf.yield %arg47 : i32
            }
            %74 = arith.addi %arg49, %c1_i32 : i32
            %75 = arith.addi %arg48, %c1_i32 : i32
            %76 = arith.cmpi eq, %74, %c6_i32 : i32
            %77 = arith.select %76, %c0_i32, %74 : i32
            %78 = scf.if %76 -> (i32) {
              %82 = arith.xori %arg50, %c1_i32 : i32
              scf.yield %82 : i32
            } else {
              scf.yield %arg50 : i32
            }
            %79 = arith.addi %arg42, %c1_i32 : i32
            %80 = arith.cmpi sgt, %36, %79 : i32
            %81 = scf.if %80 -> (i1) {
              %int_tuple_91 = cute.make_int_tuple(%72) : (i32) -> !cute.int_tuple<"?">
              %ptr_92 = cute.add_offset(%iter_6, %int_tuple_91) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %82 = builtin.unrealized_conversion_cast %ptr_92 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %83 = nvvm.mbarrier.wait.parity %82, %73 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
              scf.yield %83 : i1
            } else {
              scf.yield %true : i1
            }
            scf.yield %false, %81, %70, %72, %73, %75, %77, %78 : i1, i1, i32, i32, i32, i32, i32, i32
          } {loop_annotation = #loop_annotation}
          nvvm.wgmma.wait.group.sync.aligned 0
          %42:3 = scf.for %arg42 = %c0_i32 to %37 step %c1_i32 iter_args(%arg43 = %41#5, %arg44 = %41#6, %arg45 = %41#7) -> (i32, i32, i32)  : i32 {
            scf.if %16 {
              %int_tuple_75 = cute.make_int_tuple(%arg44) : (i32) -> !cute.int_tuple<"?">
              %ptr_76 = cute.add_offset(%ptr_8, %int_tuple_75) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %67 = builtin.unrealized_conversion_cast %ptr_76 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %68 = nvvm.mbarrier.arrive %67, %c1_i32 : !llvm.ptr<3> -> i64
            }
            %62 = arith.addi %arg44, %c1_i32 : i32
            %63 = arith.addi %arg43, %c1_i32 : i32
            %64 = arith.cmpi eq, %62, %c6_i32 : i32
            %65 = arith.select %64, %c0_i32, %62 : i32
            %66 = scf.if %64 -> (i32) {
              %67 = arith.xori %arg45, %c1_i32 : i32
              scf.yield %67 : i32
            } else {
              scf.yield %arg45 : i32
            }
            scf.yield %63, %65, %66 : i32, i32, i32
          } {loop_annotation = #loop_annotation}
          %43 = arith.cmpi eq, %36, %c0_i32 : i32
          scf.if %43 {
            cute.memref.store_vec(%cst, %rmem) : (vector<64xf32>, !memref_rmem_f32) -> ()
          }
          %iter_35 = cute.get_iter(%rmem) : !memref_rmem_f32
          iket.range_pop
          iket.range_push("epilogue")
          %coord_36 = cute.make_coord(%arg37) : (i32) -> !cute.coord<"(_,_,?)">
          %slice = cute.slice(%arg6, %coord_36) : !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute.coord<"(_,_,?)">
          %tile_37 = cute.make_tile() : () -> !cute.tile<"[128:1;128:1]">
          %coord_38 = cute.make_coord(%arg34, %arg35) : (i32, i32) -> !cute.coord<"(?,?)">
          %tiled_view = cute.local_tile(%slice, %tile_37, %coord_38) : (!cute.coord_tensor<"(0,0,?)", "(?,?):(1@1,1@0)">, !cute.tile<"[128:1;128:1]">, !cute.coord<"(?,?)">) -> !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(128,128):(1@1,1@0)">
          %tile_39 = cute.make_tile() : () -> !cute.tile<"[128:1;32:1]">
          %div_40 = cute.zipped_divide(%tiled_view, %tile_39) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(128,128):(1@1,1@0)">, !cute.tile<"[128:1;32:1]">
          %shape_41 = cute.make_shape() : () -> !cute.shape<"1">
          %lay_42 = cute.make_layout(%shape_41) : !cute.layout<"1:0">
          %grouped = cute.group_modes(%view_22) <0, 2> : (!memref_smem_bf16_1) -> !memref_smem_bf16_9
          %grouped_43 = cute.group_modes(%div_40) <0, 1> : (!cute.coord_tensor<"(?{div=128},?{div=128},?)", "((128,32),(1,4)):((1@1,1@0),(0,32@0))">) -> !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((128,32)),(1,4)):(((1@1,1@0)),(0,32@0))">
          %coord_44 = cute.make_coord() : () -> !cute.coord<"0">
          %res_smem_tensor, %res_target_tensors = cute_nvgpu.atom.tma_partition(%arg5, %coord_44, %lay_42, %grouped, %grouped_43) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 65536, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord<"0">, !cute.layout<"1:0">, !memref_smem_bf16_9, !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((128,32)),(1,4)):(((1@1,1@0)),(0,32@0))">) -> (!memref_smem_bf16_10, !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1,4)):(((1@0,1@1),0),(0,32@0))">)
          %atom = cute.make_atom() : () -> !cute_nvgpu.atom.stsm<bf16, mode = <"(8,8)">, num_matrices = 4, n>
          %44 = cute.make_tiled_copy(%atom) : !copy_stsm_4
          %coord_45 = cute.make_coord(%0) : (i32) -> !cute.coord<"?">
          %dst_partitioned = cute.tiled.copy.partition_D(%44, %view_22, %coord_45) : (!copy_stsm_4, !memref_smem_bf16_1, !cute.coord<"?">) -> !memref_smem_bf16_11
          %rmem_46 = cute.memref.alloca() : !memref_rmem_f32_1
          %iter_47 = cute.get_iter(%rmem_46) : !memref_rmem_f32_1
          %shape_48 = cute.make_shape() : () -> !cute.shape<"((2,2,2),1,(8,1))">
          %stride_49 = cute.make_stride() : () -> !cute.stride<"((1,2,4),0,(8,0))">
          %lay_50 = cute.make_layout(%shape_48, %stride_49) : !cute.layout<"((2,2,2),1,(8,1)):((1,2,4),0,(8,0))">
          %view_51 = cute.make_view(%iter_35, %lay_50) : !memref_rmem_f32_2
          %tile_52 = cute.make_tile() : () -> !cute.tile<"[[2:1;2:1;2:1];1:0;2:1]">
          %div_53 = cute.flat_divide(%view_51, %tile_52) : !memref_rmem_f32_2, !cute.tile<"[[2:1;2:1;2:1];1:0;2:1]">
          %coord_54 = cute.make_coord() : () -> !cute.coord<"(_,_,_,0,_,_)">
          %slice_55 = cute.slice(%div_53, %coord_54) : !memref_rmem_f32_3, !cute.coord<"(_,_,_,0,_,_)">
          %grouped_56 = cute.group_modes(%slice_55) <3, 5> : (!memref_rmem_f32_4) -> !memref_rmem_f32_5
          %retiled = cute.tiled.copy.retile(%44, %grouped_56) : (!copy_stsm_4, !memref_rmem_f32_5) -> !memref_rmem_f32_6
          %lay_57 = cute.get_layout(%arg8) : !memref_gmem_i32
          %coord_58 = cute.make_coord(%arg34, %arg35, %arg37) : (i32, i32, i32) -> !cute.coord<"(?,?,?)">
          %idx_59 = cute.crd2idx(%coord_58, %lay_57) : (!cute.coord<"(?,?,?)">, !cute.layout<"(?,?,?):(?{i64},1,?{i64})">) -> !cute.int_tuple<"?{i64}">
          %e0_60 = cute.get_leaves(%idx_59) : !cute.int_tuple<"?{i64}">
          %int_tuple_61 = cute.make_int_tuple(%e0_60) : (!cute.int_tuple<"?{i64}">) -> !cute.int_tuple<"?{i64}">
          %ptr_62 = cute.add_offset(%iter, %int_tuple_61) : (!cute.ptr<i32, gmem>, !cute.int_tuple<"?{i64}">) -> !cute.ptr<i32, gmem>
          %lay_63 = cute.get_layout(%arg9) : !memref_gmem_f32
          %coord_64 = cute.make_coord(%arg34, %arg35, %arg37) : (i32, i32, i32) -> !cute.coord<"(0,?,?,?)">
          %idx_65 = cute.crd2idx(%coord_64, %lay_63) : (!cute.coord<"(0,?,?,?)">, !cute.layout<"(?,?,?,?):(1,?{i64 div=4},?{i64 div=4},?{i64 div=4})">) -> !cute.int_tuple<"?{i64 div=4}">
          %e0_66 = cute.get_leaves(%idx_65) : !cute.int_tuple<"?{i64 div=4}">
          %int_tuple_67 = cute.make_int_tuple(%e0_66) : (!cute.int_tuple<"?{i64 div=4}">) -> !cute.int_tuple<"?{i64 div=4}">
          %ptr_68 = cute.add_offset(%iter_0, %int_tuple_67) : (!cute.ptr<f32, gmem, align<16>>, !cute.int_tuple<"?{i64 div=4}">) -> !cute.ptr<f32, gmem, align<16>>
          %45 = arith.cmpi slt, %arg36, %c1_i32 : i32
          scf.if %45 {
            %62 = arith.cmpi ne, %arg36, %c0_i32 : i32
            scf.if %62 {
              %74 = arith.cmpi eq, %0, %c0_i32 : i32
              scf.if %74 {
                %75 = builtin.unrealized_conversion_cast %ptr_62 : !cute.ptr<i32, gmem> to !llvm.ptr<1>
                %76 = nvvm.load.ext %75 {order = #nvvm.mem_order<acquire>, scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<1> -> i32
                scf.while (%arg42 = %76) : (i32) -> () {
                  %77 = arith.cmpi ne, %arg42, %arg36 : i32
                  scf.condition(%77)
                } do {
                  %77 = nvvm.load.ext %75 {order = #nvvm.mem_order<acquire>, scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<1> -> i32
                  scf.yield %77 : i32
                }
              }
            }
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            %coord_75 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,0))">
            %slice_76 = cute.slice(%retiled, %coord_75) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,0))">
            %lay_77 = cute.get_layout(%slice_76) : !memref_rmem_f32_7
            %lay_78 = cute.get_layout(%rmem_46) : !memref_rmem_f32_1
            %rinv = cute.right_inverse(%lay_78) : (!cute.layout<"(((2,2,2),1),1,2):(((1,2,4),0),0,8)">) -> !cute.layout<"16:1">
            %63 = cute.composition(%lay_77, %rinv) : (!cute.layout<"((8,1),1,2):((1,0),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %coalesce = cute.coalesce(%63) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
            %64 = cute.composition(%rinv, %coalesce) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %div_79 = cute.logical_divide(%slice_76, %64) : !memref_rmem_f32_7, !cute.layout<"16:1">
            %div_80 = cute.logical_divide(%rmem_46, %64) : !memref_rmem_f32_1, !cute.layout<"16:1">
            %shape_81 = cute.make_shape() : () -> !cute.shape<"8">
            %lay_82 = cute.make_layout(%shape_81) : !cute.layout<"8:1">
            %div_83 = cute.logical_divide(%div_79, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            %div_84 = cute.logical_divide(%div_80, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            %atom_85 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 256 b>
            cute.copy(%atom_85, %div_83, %div_84) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
            %65 = arith.cmpi eq, %arg36, %c0_i32 : i32
            scf.if %65 {
              %shape_116 = cute.make_shape() : () -> !cute.shape<"4096">
              %lay_117 = cute.make_layout(%shape_116) : !cute.layout<"4096:1">
              %view_118 = cute.make_view(%ptr_68, %lay_117) : !memref_gmem_f32_1
              %atom_119 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
              %74 = cute.make_tiled_copy(%atom_119) : !copy_simt
              %dst_partitioned_120 = cute.tiled.copy.partition_D(%74, %view_118, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
              %shape_121 = cute.make_shape() : () -> !cute.shape<"4">
              %lay_122 = cute.make_layout(%shape_121) : !cute.layout<"4:1">
              %coord_123 = cute.make_coord() : () -> !cute.coord<"(_,0)">
              %slice_124 = cute.slice(%dst_partitioned_120, %coord_123) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
              %iter_125 = cute.get_iter(%slice_124) : !memref_gmem_f32_3
              %lay_126 = cute.get_layout(%slice_124) : !memref_gmem_f32_3
              %append = cute.append_to_rank<2> (%lay_122, %lay_42) : !cute.layout<"4:1">, !cute.layout<"1:0">
              %view_127 = cute.make_view(%iter_47, %append) : !memref_rmem_f32_10
              %grouped_128 = cute.group_modes(%view_127) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
              %append_129 = cute.append_to_rank<2> (%lay_126, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_130 = cute.make_view(%iter_125, %append_129) : !memref_gmem_f32_4
              %grouped_131 = cute.group_modes(%view_130) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_128, %grouped_131) : (!copy_simt, !memref_rmem_f32_11, !memref_gmem_f32_5)
              %int_tuple_132 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
              %ptr_133 = cute.add_offset(%iter_47, %int_tuple_132) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
              %coord_134 = cute.make_coord() : () -> !cute.coord<"(_,1)">
              %slice_135 = cute.slice(%dst_partitioned_120, %coord_134) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
              %iter_136 = cute.get_iter(%slice_135) : !memref_gmem_f32_3
              %lay_137 = cute.get_layout(%slice_135) : !memref_gmem_f32_3
              %view_138 = cute.make_view(%ptr_133, %append) : !memref_rmem_f32_12
              %grouped_139 = cute.group_modes(%view_138) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
              %append_140 = cute.append_to_rank<2> (%lay_137, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_141 = cute.make_view(%iter_136, %append_140) : !memref_gmem_f32_4
              %grouped_142 = cute.group_modes(%view_141) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_139, %grouped_142) : (!copy_simt, !memref_rmem_f32_13, !memref_gmem_f32_5)
              %int_tuple_143 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
              %ptr_144 = cute.add_offset(%iter_47, %int_tuple_143) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
              %coord_145 = cute.make_coord() : () -> !cute.coord<"(_,2)">
              %slice_146 = cute.slice(%dst_partitioned_120, %coord_145) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
              %iter_147 = cute.get_iter(%slice_146) : !memref_gmem_f32_3
              %lay_148 = cute.get_layout(%slice_146) : !memref_gmem_f32_3
              %view_149 = cute.make_view(%ptr_144, %append) : !memref_rmem_f32_10
              %grouped_150 = cute.group_modes(%view_149) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
              %append_151 = cute.append_to_rank<2> (%lay_148, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_152 = cute.make_view(%iter_147, %append_151) : !memref_gmem_f32_4
              %grouped_153 = cute.group_modes(%view_152) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_150, %grouped_153) : (!copy_simt, !memref_rmem_f32_11, !memref_gmem_f32_5)
              %int_tuple_154 = cute.make_int_tuple() : () -> !cute.int_tuple<"12">
              %ptr_155 = cute.add_offset(%iter_47, %int_tuple_154) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
              %coord_156 = cute.make_coord() : () -> !cute.coord<"(_,3)">
              %slice_157 = cute.slice(%dst_partitioned_120, %coord_156) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
              %iter_158 = cute.get_iter(%slice_157) : !memref_gmem_f32_3
              %lay_159 = cute.get_layout(%slice_157) : !memref_gmem_f32_3
              %view_160 = cute.make_view(%ptr_155, %append) : !memref_rmem_f32_12
              %grouped_161 = cute.group_modes(%view_160) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
              %append_162 = cute.append_to_rank<2> (%lay_159, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_163 = cute.make_view(%iter_158, %append_162) : !memref_gmem_f32_4
              %grouped_164 = cute.group_modes(%view_163) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_161, %grouped_164) : (!copy_simt, !memref_rmem_f32_13, !memref_gmem_f32_5)
            } else {
              %shape_116 = cute.make_shape() : () -> !cute.shape<"4096">
              %lay_117 = cute.make_layout(%shape_116) : !cute.layout<"4096:1">
              %view_118 = cute.make_view(%ptr_68, %lay_117) : !memref_gmem_f32_1
              %atom_119 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
              %74 = cute.make_tiled_copy(%atom_119) : !copy_simt
              %dst_partitioned_120 = cute.tiled.copy.partition_D(%74, %view_118, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
              %shape_121 = cute.make_shape() : () -> !cute.shape<"4">
              %lay_122 = cute.make_layout(%shape_121) : !cute.layout<"4:1">
              %view_123 = cute.make_view(%iter_47, %lay_122) : !memref_rmem_f32_14
              %coord_124 = cute.make_coord() : () -> !cute.coord<"(_,0)">
              %slice_125 = cute.slice(%dst_partitioned_120, %coord_124) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
              %iter_126 = cute.get_iter(%slice_125) : !memref_gmem_f32_3
              %75 = cute.memref.load_vec(%view_123) : (!memref_rmem_f32_14) -> vector<4xf32>
              %76 = builtin.unrealized_conversion_cast %iter_126 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %76, %75 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_127 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
              %ptr_128 = cute.add_offset(%iter_47, %int_tuple_127) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
              %view_129 = cute.make_view(%ptr_128, %lay_122) : !memref_rmem_f32_15
              %coord_130 = cute.make_coord() : () -> !cute.coord<"(_,1)">
              %slice_131 = cute.slice(%dst_partitioned_120, %coord_130) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
              %iter_132 = cute.get_iter(%slice_131) : !memref_gmem_f32_3
              %77 = cute.memref.load_vec(%view_129) : (!memref_rmem_f32_15) -> vector<4xf32>
              %78 = builtin.unrealized_conversion_cast %iter_132 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %78, %77 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_133 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
              %ptr_134 = cute.add_offset(%iter_47, %int_tuple_133) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
              %view_135 = cute.make_view(%ptr_134, %lay_122) : !memref_rmem_f32_14
              %coord_136 = cute.make_coord() : () -> !cute.coord<"(_,2)">
              %slice_137 = cute.slice(%dst_partitioned_120, %coord_136) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
              %iter_138 = cute.get_iter(%slice_137) : !memref_gmem_f32_3
              %79 = cute.memref.load_vec(%view_135) : (!memref_rmem_f32_14) -> vector<4xf32>
              %80 = builtin.unrealized_conversion_cast %iter_138 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %80, %79 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_139 = cute.make_int_tuple() : () -> !cute.int_tuple<"12">
              %ptr_140 = cute.add_offset(%iter_47, %int_tuple_139) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
              %view_141 = cute.make_view(%ptr_140, %lay_122) : !memref_rmem_f32_15
              %coord_142 = cute.make_coord() : () -> !cute.coord<"(_,3)">
              %slice_143 = cute.slice(%dst_partitioned_120, %coord_142) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
              %iter_144 = cute.get_iter(%slice_143) : !memref_gmem_f32_3
              %81 = cute.memref.load_vec(%view_141) : (!memref_rmem_f32_15) -> vector<4xf32>
              %82 = builtin.unrealized_conversion_cast %iter_144 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %82, %81 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
            }
            %coord_86 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,1))">
            %slice_87 = cute.slice(%retiled, %coord_86) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,1))">
            %lay_88 = cute.get_layout(%slice_87) : !memref_rmem_f32_7
            %66 = cute.composition(%lay_88, %rinv) : (!cute.layout<"((8,1),1,2):((1,0),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %coalesce_89 = cute.coalesce(%66) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
            %67 = cute.composition(%rinv, %coalesce_89) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %div_90 = cute.logical_divide(%slice_87, %67) : !memref_rmem_f32_7, !cute.layout<"16:1">
            %div_91 = cute.logical_divide(%rmem_46, %67) : !memref_rmem_f32_1, !cute.layout<"16:1">
            %div_92 = cute.logical_divide(%div_90, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            %div_93 = cute.logical_divide(%div_91, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            cute.copy(%atom_85, %div_92, %div_93) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
            %int_tuple_94 = cute.make_int_tuple() : () -> !cute.int_tuple<"4096">
            %ptr_95 = cute.add_offset(%ptr_68, %int_tuple_94) : (!cute.ptr<f32, gmem, align<16>>, !cute.int_tuple<"4096">) -> !cute.ptr<f32, gmem, align<16>>
            scf.if %65 {
              %shape_116 = cute.make_shape() : () -> !cute.shape<"4096">
              %lay_117 = cute.make_layout(%shape_116) : !cute.layout<"4096:1">
              %view_118 = cute.make_view(%ptr_95, %lay_117) : !memref_gmem_f32_1
              %atom_119 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
              %74 = cute.make_tiled_copy(%atom_119) : !copy_simt
              %dst_partitioned_120 = cute.tiled.copy.partition_D(%74, %view_118, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
              %shape_121 = cute.make_shape() : () -> !cute.shape<"4">
              %lay_122 = cute.make_layout(%shape_121) : !cute.layout<"4:1">
              %coord_123 = cute.make_coord() : () -> !cute.coord<"(_,0)">
              %slice_124 = cute.slice(%dst_partitioned_120, %coord_123) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
              %iter_125 = cute.get_iter(%slice_124) : !memref_gmem_f32_3
              %lay_126 = cute.get_layout(%slice_124) : !memref_gmem_f32_3
              %append = cute.append_to_rank<2> (%lay_122, %lay_42) : !cute.layout<"4:1">, !cute.layout<"1:0">
              %view_127 = cute.make_view(%iter_47, %append) : !memref_rmem_f32_10
              %grouped_128 = cute.group_modes(%view_127) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
              %append_129 = cute.append_to_rank<2> (%lay_126, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_130 = cute.make_view(%iter_125, %append_129) : !memref_gmem_f32_4
              %grouped_131 = cute.group_modes(%view_130) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_128, %grouped_131) : (!copy_simt, !memref_rmem_f32_11, !memref_gmem_f32_5)
              %int_tuple_132 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
              %ptr_133 = cute.add_offset(%iter_47, %int_tuple_132) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
              %coord_134 = cute.make_coord() : () -> !cute.coord<"(_,1)">
              %slice_135 = cute.slice(%dst_partitioned_120, %coord_134) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
              %iter_136 = cute.get_iter(%slice_135) : !memref_gmem_f32_3
              %lay_137 = cute.get_layout(%slice_135) : !memref_gmem_f32_3
              %view_138 = cute.make_view(%ptr_133, %append) : !memref_rmem_f32_12
              %grouped_139 = cute.group_modes(%view_138) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
              %append_140 = cute.append_to_rank<2> (%lay_137, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_141 = cute.make_view(%iter_136, %append_140) : !memref_gmem_f32_4
              %grouped_142 = cute.group_modes(%view_141) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_139, %grouped_142) : (!copy_simt, !memref_rmem_f32_13, !memref_gmem_f32_5)
              %int_tuple_143 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
              %ptr_144 = cute.add_offset(%iter_47, %int_tuple_143) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
              %coord_145 = cute.make_coord() : () -> !cute.coord<"(_,2)">
              %slice_146 = cute.slice(%dst_partitioned_120, %coord_145) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
              %iter_147 = cute.get_iter(%slice_146) : !memref_gmem_f32_3
              %lay_148 = cute.get_layout(%slice_146) : !memref_gmem_f32_3
              %view_149 = cute.make_view(%ptr_144, %append) : !memref_rmem_f32_10
              %grouped_150 = cute.group_modes(%view_149) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
              %append_151 = cute.append_to_rank<2> (%lay_148, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_152 = cute.make_view(%iter_147, %append_151) : !memref_gmem_f32_4
              %grouped_153 = cute.group_modes(%view_152) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_150, %grouped_153) : (!copy_simt, !memref_rmem_f32_11, !memref_gmem_f32_5)
              %int_tuple_154 = cute.make_int_tuple() : () -> !cute.int_tuple<"12">
              %ptr_155 = cute.add_offset(%iter_47, %int_tuple_154) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
              %coord_156 = cute.make_coord() : () -> !cute.coord<"(_,3)">
              %slice_157 = cute.slice(%dst_partitioned_120, %coord_156) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
              %iter_158 = cute.get_iter(%slice_157) : !memref_gmem_f32_3
              %lay_159 = cute.get_layout(%slice_157) : !memref_gmem_f32_3
              %view_160 = cute.make_view(%ptr_155, %append) : !memref_rmem_f32_12
              %grouped_161 = cute.group_modes(%view_160) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
              %append_162 = cute.append_to_rank<2> (%lay_159, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_163 = cute.make_view(%iter_158, %append_162) : !memref_gmem_f32_4
              %grouped_164 = cute.group_modes(%view_163) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_161, %grouped_164) : (!copy_simt, !memref_rmem_f32_13, !memref_gmem_f32_5)
            } else {
              %shape_116 = cute.make_shape() : () -> !cute.shape<"4096">
              %lay_117 = cute.make_layout(%shape_116) : !cute.layout<"4096:1">
              %view_118 = cute.make_view(%ptr_95, %lay_117) : !memref_gmem_f32_1
              %atom_119 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
              %74 = cute.make_tiled_copy(%atom_119) : !copy_simt
              %dst_partitioned_120 = cute.tiled.copy.partition_D(%74, %view_118, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
              %shape_121 = cute.make_shape() : () -> !cute.shape<"4">
              %lay_122 = cute.make_layout(%shape_121) : !cute.layout<"4:1">
              %view_123 = cute.make_view(%iter_47, %lay_122) : !memref_rmem_f32_14
              %coord_124 = cute.make_coord() : () -> !cute.coord<"(_,0)">
              %slice_125 = cute.slice(%dst_partitioned_120, %coord_124) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
              %iter_126 = cute.get_iter(%slice_125) : !memref_gmem_f32_3
              %75 = cute.memref.load_vec(%view_123) : (!memref_rmem_f32_14) -> vector<4xf32>
              %76 = builtin.unrealized_conversion_cast %iter_126 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %76, %75 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_127 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
              %ptr_128 = cute.add_offset(%iter_47, %int_tuple_127) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
              %view_129 = cute.make_view(%ptr_128, %lay_122) : !memref_rmem_f32_15
              %coord_130 = cute.make_coord() : () -> !cute.coord<"(_,1)">
              %slice_131 = cute.slice(%dst_partitioned_120, %coord_130) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
              %iter_132 = cute.get_iter(%slice_131) : !memref_gmem_f32_3
              %77 = cute.memref.load_vec(%view_129) : (!memref_rmem_f32_15) -> vector<4xf32>
              %78 = builtin.unrealized_conversion_cast %iter_132 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %78, %77 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_133 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
              %ptr_134 = cute.add_offset(%iter_47, %int_tuple_133) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
              %view_135 = cute.make_view(%ptr_134, %lay_122) : !memref_rmem_f32_14
              %coord_136 = cute.make_coord() : () -> !cute.coord<"(_,2)">
              %slice_137 = cute.slice(%dst_partitioned_120, %coord_136) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
              %iter_138 = cute.get_iter(%slice_137) : !memref_gmem_f32_3
              %79 = cute.memref.load_vec(%view_135) : (!memref_rmem_f32_14) -> vector<4xf32>
              %80 = builtin.unrealized_conversion_cast %iter_138 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %80, %79 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_139 = cute.make_int_tuple() : () -> !cute.int_tuple<"12">
              %ptr_140 = cute.add_offset(%iter_47, %int_tuple_139) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
              %view_141 = cute.make_view(%ptr_140, %lay_122) : !memref_rmem_f32_15
              %coord_142 = cute.make_coord() : () -> !cute.coord<"(_,3)">
              %slice_143 = cute.slice(%dst_partitioned_120, %coord_142) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
              %iter_144 = cute.get_iter(%slice_143) : !memref_gmem_f32_3
              %81 = cute.memref.load_vec(%view_141) : (!memref_rmem_f32_15) -> vector<4xf32>
              %82 = builtin.unrealized_conversion_cast %iter_144 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %82, %81 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
            }
            %coord_96 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,2))">
            %slice_97 = cute.slice(%retiled, %coord_96) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,2))">
            %lay_98 = cute.get_layout(%slice_97) : !memref_rmem_f32_7
            %68 = cute.composition(%lay_98, %rinv) : (!cute.layout<"((8,1),1,2):((1,0),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %coalesce_99 = cute.coalesce(%68) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
            %69 = cute.composition(%rinv, %coalesce_99) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %div_100 = cute.logical_divide(%slice_97, %69) : !memref_rmem_f32_7, !cute.layout<"16:1">
            %div_101 = cute.logical_divide(%rmem_46, %69) : !memref_rmem_f32_1, !cute.layout<"16:1">
            %div_102 = cute.logical_divide(%div_100, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            %div_103 = cute.logical_divide(%div_101, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            cute.copy(%atom_85, %div_102, %div_103) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
            %int_tuple_104 = cute.make_int_tuple() : () -> !cute.int_tuple<"8192">
            %ptr_105 = cute.add_offset(%ptr_68, %int_tuple_104) : (!cute.ptr<f32, gmem, align<16>>, !cute.int_tuple<"8192">) -> !cute.ptr<f32, gmem, align<16>>
            scf.if %65 {
              %shape_116 = cute.make_shape() : () -> !cute.shape<"4096">
              %lay_117 = cute.make_layout(%shape_116) : !cute.layout<"4096:1">
              %view_118 = cute.make_view(%ptr_105, %lay_117) : !memref_gmem_f32_1
              %atom_119 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
              %74 = cute.make_tiled_copy(%atom_119) : !copy_simt
              %dst_partitioned_120 = cute.tiled.copy.partition_D(%74, %view_118, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
              %shape_121 = cute.make_shape() : () -> !cute.shape<"4">
              %lay_122 = cute.make_layout(%shape_121) : !cute.layout<"4:1">
              %coord_123 = cute.make_coord() : () -> !cute.coord<"(_,0)">
              %slice_124 = cute.slice(%dst_partitioned_120, %coord_123) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
              %iter_125 = cute.get_iter(%slice_124) : !memref_gmem_f32_3
              %lay_126 = cute.get_layout(%slice_124) : !memref_gmem_f32_3
              %append = cute.append_to_rank<2> (%lay_122, %lay_42) : !cute.layout<"4:1">, !cute.layout<"1:0">
              %view_127 = cute.make_view(%iter_47, %append) : !memref_rmem_f32_10
              %grouped_128 = cute.group_modes(%view_127) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
              %append_129 = cute.append_to_rank<2> (%lay_126, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_130 = cute.make_view(%iter_125, %append_129) : !memref_gmem_f32_4
              %grouped_131 = cute.group_modes(%view_130) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_128, %grouped_131) : (!copy_simt, !memref_rmem_f32_11, !memref_gmem_f32_5)
              %int_tuple_132 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
              %ptr_133 = cute.add_offset(%iter_47, %int_tuple_132) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
              %coord_134 = cute.make_coord() : () -> !cute.coord<"(_,1)">
              %slice_135 = cute.slice(%dst_partitioned_120, %coord_134) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
              %iter_136 = cute.get_iter(%slice_135) : !memref_gmem_f32_3
              %lay_137 = cute.get_layout(%slice_135) : !memref_gmem_f32_3
              %view_138 = cute.make_view(%ptr_133, %append) : !memref_rmem_f32_12
              %grouped_139 = cute.group_modes(%view_138) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
              %append_140 = cute.append_to_rank<2> (%lay_137, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_141 = cute.make_view(%iter_136, %append_140) : !memref_gmem_f32_4
              %grouped_142 = cute.group_modes(%view_141) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_139, %grouped_142) : (!copy_simt, !memref_rmem_f32_13, !memref_gmem_f32_5)
              %int_tuple_143 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
              %ptr_144 = cute.add_offset(%iter_47, %int_tuple_143) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
              %coord_145 = cute.make_coord() : () -> !cute.coord<"(_,2)">
              %slice_146 = cute.slice(%dst_partitioned_120, %coord_145) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
              %iter_147 = cute.get_iter(%slice_146) : !memref_gmem_f32_3
              %lay_148 = cute.get_layout(%slice_146) : !memref_gmem_f32_3
              %view_149 = cute.make_view(%ptr_144, %append) : !memref_rmem_f32_10
              %grouped_150 = cute.group_modes(%view_149) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
              %append_151 = cute.append_to_rank<2> (%lay_148, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_152 = cute.make_view(%iter_147, %append_151) : !memref_gmem_f32_4
              %grouped_153 = cute.group_modes(%view_152) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_150, %grouped_153) : (!copy_simt, !memref_rmem_f32_11, !memref_gmem_f32_5)
              %int_tuple_154 = cute.make_int_tuple() : () -> !cute.int_tuple<"12">
              %ptr_155 = cute.add_offset(%iter_47, %int_tuple_154) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
              %coord_156 = cute.make_coord() : () -> !cute.coord<"(_,3)">
              %slice_157 = cute.slice(%dst_partitioned_120, %coord_156) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
              %iter_158 = cute.get_iter(%slice_157) : !memref_gmem_f32_3
              %lay_159 = cute.get_layout(%slice_157) : !memref_gmem_f32_3
              %view_160 = cute.make_view(%ptr_155, %append) : !memref_rmem_f32_12
              %grouped_161 = cute.group_modes(%view_160) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
              %append_162 = cute.append_to_rank<2> (%lay_159, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_163 = cute.make_view(%iter_158, %append_162) : !memref_gmem_f32_4
              %grouped_164 = cute.group_modes(%view_163) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_161, %grouped_164) : (!copy_simt, !memref_rmem_f32_13, !memref_gmem_f32_5)
            } else {
              %shape_116 = cute.make_shape() : () -> !cute.shape<"4096">
              %lay_117 = cute.make_layout(%shape_116) : !cute.layout<"4096:1">
              %view_118 = cute.make_view(%ptr_105, %lay_117) : !memref_gmem_f32_1
              %atom_119 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
              %74 = cute.make_tiled_copy(%atom_119) : !copy_simt
              %dst_partitioned_120 = cute.tiled.copy.partition_D(%74, %view_118, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
              %shape_121 = cute.make_shape() : () -> !cute.shape<"4">
              %lay_122 = cute.make_layout(%shape_121) : !cute.layout<"4:1">
              %view_123 = cute.make_view(%iter_47, %lay_122) : !memref_rmem_f32_14
              %coord_124 = cute.make_coord() : () -> !cute.coord<"(_,0)">
              %slice_125 = cute.slice(%dst_partitioned_120, %coord_124) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
              %iter_126 = cute.get_iter(%slice_125) : !memref_gmem_f32_3
              %75 = cute.memref.load_vec(%view_123) : (!memref_rmem_f32_14) -> vector<4xf32>
              %76 = builtin.unrealized_conversion_cast %iter_126 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %76, %75 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_127 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
              %ptr_128 = cute.add_offset(%iter_47, %int_tuple_127) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
              %view_129 = cute.make_view(%ptr_128, %lay_122) : !memref_rmem_f32_15
              %coord_130 = cute.make_coord() : () -> !cute.coord<"(_,1)">
              %slice_131 = cute.slice(%dst_partitioned_120, %coord_130) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
              %iter_132 = cute.get_iter(%slice_131) : !memref_gmem_f32_3
              %77 = cute.memref.load_vec(%view_129) : (!memref_rmem_f32_15) -> vector<4xf32>
              %78 = builtin.unrealized_conversion_cast %iter_132 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %78, %77 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_133 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
              %ptr_134 = cute.add_offset(%iter_47, %int_tuple_133) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
              %view_135 = cute.make_view(%ptr_134, %lay_122) : !memref_rmem_f32_14
              %coord_136 = cute.make_coord() : () -> !cute.coord<"(_,2)">
              %slice_137 = cute.slice(%dst_partitioned_120, %coord_136) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
              %iter_138 = cute.get_iter(%slice_137) : !memref_gmem_f32_3
              %79 = cute.memref.load_vec(%view_135) : (!memref_rmem_f32_14) -> vector<4xf32>
              %80 = builtin.unrealized_conversion_cast %iter_138 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %80, %79 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_139 = cute.make_int_tuple() : () -> !cute.int_tuple<"12">
              %ptr_140 = cute.add_offset(%iter_47, %int_tuple_139) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
              %view_141 = cute.make_view(%ptr_140, %lay_122) : !memref_rmem_f32_15
              %coord_142 = cute.make_coord() : () -> !cute.coord<"(_,3)">
              %slice_143 = cute.slice(%dst_partitioned_120, %coord_142) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
              %iter_144 = cute.get_iter(%slice_143) : !memref_gmem_f32_3
              %81 = cute.memref.load_vec(%view_141) : (!memref_rmem_f32_15) -> vector<4xf32>
              %82 = builtin.unrealized_conversion_cast %iter_144 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %82, %81 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
            }
            %coord_106 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,3))">
            %slice_107 = cute.slice(%retiled, %coord_106) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,3))">
            %lay_108 = cute.get_layout(%slice_107) : !memref_rmem_f32_7
            %70 = cute.composition(%lay_108, %rinv) : (!cute.layout<"((8,1),1,2):((1,0),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %coalesce_109 = cute.coalesce(%70) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
            %71 = cute.composition(%rinv, %coalesce_109) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %div_110 = cute.logical_divide(%slice_107, %71) : !memref_rmem_f32_7, !cute.layout<"16:1">
            %div_111 = cute.logical_divide(%rmem_46, %71) : !memref_rmem_f32_1, !cute.layout<"16:1">
            %div_112 = cute.logical_divide(%div_110, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            %div_113 = cute.logical_divide(%div_111, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            cute.copy(%atom_85, %div_112, %div_113) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
            %int_tuple_114 = cute.make_int_tuple() : () -> !cute.int_tuple<"12288">
            %ptr_115 = cute.add_offset(%ptr_68, %int_tuple_114) : (!cute.ptr<f32, gmem, align<16>>, !cute.int_tuple<"12288">) -> !cute.ptr<f32, gmem, align<16>>
            scf.if %65 {
              %shape_116 = cute.make_shape() : () -> !cute.shape<"4096">
              %lay_117 = cute.make_layout(%shape_116) : !cute.layout<"4096:1">
              %view_118 = cute.make_view(%ptr_115, %lay_117) : !memref_gmem_f32_1
              %atom_119 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
              %74 = cute.make_tiled_copy(%atom_119) : !copy_simt
              %dst_partitioned_120 = cute.tiled.copy.partition_D(%74, %view_118, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
              %shape_121 = cute.make_shape() : () -> !cute.shape<"4">
              %lay_122 = cute.make_layout(%shape_121) : !cute.layout<"4:1">
              %coord_123 = cute.make_coord() : () -> !cute.coord<"(_,0)">
              %slice_124 = cute.slice(%dst_partitioned_120, %coord_123) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
              %iter_125 = cute.get_iter(%slice_124) : !memref_gmem_f32_3
              %lay_126 = cute.get_layout(%slice_124) : !memref_gmem_f32_3
              %append = cute.append_to_rank<2> (%lay_122, %lay_42) : !cute.layout<"4:1">, !cute.layout<"1:0">
              %view_127 = cute.make_view(%iter_47, %append) : !memref_rmem_f32_10
              %grouped_128 = cute.group_modes(%view_127) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
              %append_129 = cute.append_to_rank<2> (%lay_126, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_130 = cute.make_view(%iter_125, %append_129) : !memref_gmem_f32_4
              %grouped_131 = cute.group_modes(%view_130) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_128, %grouped_131) : (!copy_simt, !memref_rmem_f32_11, !memref_gmem_f32_5)
              %int_tuple_132 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
              %ptr_133 = cute.add_offset(%iter_47, %int_tuple_132) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
              %coord_134 = cute.make_coord() : () -> !cute.coord<"(_,1)">
              %slice_135 = cute.slice(%dst_partitioned_120, %coord_134) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
              %iter_136 = cute.get_iter(%slice_135) : !memref_gmem_f32_3
              %lay_137 = cute.get_layout(%slice_135) : !memref_gmem_f32_3
              %view_138 = cute.make_view(%ptr_133, %append) : !memref_rmem_f32_12
              %grouped_139 = cute.group_modes(%view_138) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
              %append_140 = cute.append_to_rank<2> (%lay_137, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_141 = cute.make_view(%iter_136, %append_140) : !memref_gmem_f32_4
              %grouped_142 = cute.group_modes(%view_141) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_139, %grouped_142) : (!copy_simt, !memref_rmem_f32_13, !memref_gmem_f32_5)
              %int_tuple_143 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
              %ptr_144 = cute.add_offset(%iter_47, %int_tuple_143) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
              %coord_145 = cute.make_coord() : () -> !cute.coord<"(_,2)">
              %slice_146 = cute.slice(%dst_partitioned_120, %coord_145) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
              %iter_147 = cute.get_iter(%slice_146) : !memref_gmem_f32_3
              %lay_148 = cute.get_layout(%slice_146) : !memref_gmem_f32_3
              %view_149 = cute.make_view(%ptr_144, %append) : !memref_rmem_f32_10
              %grouped_150 = cute.group_modes(%view_149) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
              %append_151 = cute.append_to_rank<2> (%lay_148, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_152 = cute.make_view(%iter_147, %append_151) : !memref_gmem_f32_4
              %grouped_153 = cute.group_modes(%view_152) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_150, %grouped_153) : (!copy_simt, !memref_rmem_f32_11, !memref_gmem_f32_5)
              %int_tuple_154 = cute.make_int_tuple() : () -> !cute.int_tuple<"12">
              %ptr_155 = cute.add_offset(%iter_47, %int_tuple_154) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
              %coord_156 = cute.make_coord() : () -> !cute.coord<"(_,3)">
              %slice_157 = cute.slice(%dst_partitioned_120, %coord_156) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
              %iter_158 = cute.get_iter(%slice_157) : !memref_gmem_f32_3
              %lay_159 = cute.get_layout(%slice_157) : !memref_gmem_f32_3
              %view_160 = cute.make_view(%ptr_155, %append) : !memref_rmem_f32_12
              %grouped_161 = cute.group_modes(%view_160) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
              %append_162 = cute.append_to_rank<2> (%lay_159, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
              %view_163 = cute.make_view(%iter_158, %append_162) : !memref_gmem_f32_4
              %grouped_164 = cute.group_modes(%view_163) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
              cute.copy(%74, %grouped_161, %grouped_164) : (!copy_simt, !memref_rmem_f32_13, !memref_gmem_f32_5)
            } else {
              %shape_116 = cute.make_shape() : () -> !cute.shape<"4096">
              %lay_117 = cute.make_layout(%shape_116) : !cute.layout<"4096:1">
              %view_118 = cute.make_view(%ptr_115, %lay_117) : !memref_gmem_f32_1
              %atom_119 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
              %74 = cute.make_tiled_copy(%atom_119) : !copy_simt
              %dst_partitioned_120 = cute.tiled.copy.partition_D(%74, %view_118, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
              %shape_121 = cute.make_shape() : () -> !cute.shape<"4">
              %lay_122 = cute.make_layout(%shape_121) : !cute.layout<"4:1">
              %view_123 = cute.make_view(%iter_47, %lay_122) : !memref_rmem_f32_14
              %coord_124 = cute.make_coord() : () -> !cute.coord<"(_,0)">
              %slice_125 = cute.slice(%dst_partitioned_120, %coord_124) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
              %iter_126 = cute.get_iter(%slice_125) : !memref_gmem_f32_3
              %75 = cute.memref.load_vec(%view_123) : (!memref_rmem_f32_14) -> vector<4xf32>
              %76 = builtin.unrealized_conversion_cast %iter_126 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %76, %75 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_127 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
              %ptr_128 = cute.add_offset(%iter_47, %int_tuple_127) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
              %view_129 = cute.make_view(%ptr_128, %lay_122) : !memref_rmem_f32_15
              %coord_130 = cute.make_coord() : () -> !cute.coord<"(_,1)">
              %slice_131 = cute.slice(%dst_partitioned_120, %coord_130) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
              %iter_132 = cute.get_iter(%slice_131) : !memref_gmem_f32_3
              %77 = cute.memref.load_vec(%view_129) : (!memref_rmem_f32_15) -> vector<4xf32>
              %78 = builtin.unrealized_conversion_cast %iter_132 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %78, %77 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_133 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
              %ptr_134 = cute.add_offset(%iter_47, %int_tuple_133) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
              %view_135 = cute.make_view(%ptr_134, %lay_122) : !memref_rmem_f32_14
              %coord_136 = cute.make_coord() : () -> !cute.coord<"(_,2)">
              %slice_137 = cute.slice(%dst_partitioned_120, %coord_136) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
              %iter_138 = cute.get_iter(%slice_137) : !memref_gmem_f32_3
              %79 = cute.memref.load_vec(%view_135) : (!memref_rmem_f32_14) -> vector<4xf32>
              %80 = builtin.unrealized_conversion_cast %iter_138 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %80, %79 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
              %int_tuple_139 = cute.make_int_tuple() : () -> !cute.int_tuple<"12">
              %ptr_140 = cute.add_offset(%iter_47, %int_tuple_139) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
              %view_141 = cute.make_view(%ptr_140, %lay_122) : !memref_rmem_f32_15
              %coord_142 = cute.make_coord() : () -> !cute.coord<"(_,3)">
              %slice_143 = cute.slice(%dst_partitioned_120, %coord_142) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
              %iter_144 = cute.get_iter(%slice_143) : !memref_gmem_f32_3
              %81 = cute.memref.load_vec(%view_141) : (!memref_rmem_f32_15) -> vector<4xf32>
              %82 = builtin.unrealized_conversion_cast %iter_144 : !cute.ptr<f32, gmem, align<16>> to !llvm.ptr<1>
              nvvm.red %82, %81 {memOrder = #nvvm.mem_order<relaxed>, memScope = #nvvm.mem_scope<gpu>, op = #nvvm.red_op<add>, type = #nvvm.red_type<f32>} : !llvm.ptr<1>, vector<4xf32>
            }
            %72 = arith.addi %arg36, %c1_i32 : i32
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            %73 = arith.cmpi eq, %0, %c0_i32 : i32
            scf.if %73 {
              %74 = builtin.unrealized_conversion_cast %ptr_62 : !cute.ptr<i32, gmem> to !llvm.ptr<1>
              nvvm.store.ext %72, %74 {order = #nvvm.mem_order<release>, scope = #nvvm.mem_scope<gpu>} : i32, !llvm.ptr<1>
            }
          } else {
            %62 = arith.cmpi eq, %0, %c0_i32 : i32
            scf.if %62 {
              %100 = builtin.unrealized_conversion_cast %ptr_62 : !cute.ptr<i32, gmem> to !llvm.ptr<1>
              %101 = nvvm.load.ext %100 {order = #nvvm.mem_order<acquire>, scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<1> -> i32
              scf.while (%arg42 = %101) : (i32) -> () {
                %102 = arith.cmpi ne, %arg42, %c1_i32 : i32
                scf.condition(%102)
              } do {
                %102 = nvvm.load.ext %100 {order = #nvvm.mem_order<acquire>, scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<1> -> i32
                scf.yield %102 : i32
              }
            }
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            %63 = arith.muli %arg38, %c4_i32 : i32
            %lay_75 = cute.get_layout(%rmem_46) : !memref_rmem_f32_1
            %coord_76 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,0))">
            %slice_77 = cute.slice(%retiled, %coord_76) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,0))">
            %lay_78 = cute.get_layout(%slice_77) : !memref_rmem_f32_7
            %rinv = cute.right_inverse(%lay_75) : (!cute.layout<"(((2,2,2),1),1,2):(((1,2,4),0),0,8)">) -> !cute.layout<"16:1">
            %64 = cute.composition(%lay_78, %rinv) : (!cute.layout<"((8,1),1,2):((1,0),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %coalesce = cute.coalesce(%64) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
            %65 = cute.composition(%rinv, %coalesce) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %div_79 = cute.logical_divide(%slice_77, %65) : !memref_rmem_f32_7, !cute.layout<"16:1">
            %div_80 = cute.logical_divide(%rmem_46, %65) : !memref_rmem_f32_1, !cute.layout<"16:1">
            %shape_81 = cute.make_shape() : () -> !cute.shape<"8">
            %lay_82 = cute.make_layout(%shape_81) : !cute.layout<"8:1">
            %div_83 = cute.logical_divide(%div_79, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            %div_84 = cute.logical_divide(%div_80, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            %atom_85 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 256 b>
            cute.copy(%atom_85, %div_83, %div_84) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
            %rmem_86 = cute.memref.alloca() : !memref_rmem_f32_1
            %iter_87 = cute.get_iter(%rmem_86) : !memref_rmem_f32_1
            %shape_88 = cute.make_shape() : () -> !cute.shape<"4096">
            %lay_89 = cute.make_layout(%shape_88) : !cute.layout<"4096:1">
            %view_90 = cute.make_view(%ptr_68, %lay_89) : !memref_gmem_f32_1
            %atom_91 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
            %66 = cute.make_tiled_copy(%atom_91) : !copy_simt
            %src_partitioned = cute.tiled.copy.partition_S(%66, %view_90, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
            %shape_92 = cute.make_shape() : () -> !cute.shape<"4">
            %lay_93 = cute.make_layout(%shape_92) : !cute.layout<"4:1">
            %coord_94 = cute.make_coord() : () -> !cute.coord<"(_,0)">
            %slice_95 = cute.slice(%src_partitioned, %coord_94) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
            %iter_96 = cute.get_iter(%slice_95) : !memref_gmem_f32_3
            %lay_97 = cute.get_layout(%slice_95) : !memref_gmem_f32_3
            %append = cute.append_to_rank<2> (%lay_97, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_98 = cute.make_view(%iter_96, %append) : !memref_gmem_f32_4
            %grouped_99 = cute.group_modes(%view_98) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %append_100 = cute.append_to_rank<2> (%lay_93, %lay_42) : !cute.layout<"4:1">, !cute.layout<"1:0">
            %view_101 = cute.make_view(%iter_87, %append_100) : !memref_rmem_f32_10
            %grouped_102 = cute.group_modes(%view_101) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
            cute.copy(%66, %grouped_99, %grouped_102) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_11)
            %int_tuple_103 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
            %ptr_104 = cute.add_offset(%iter_87, %int_tuple_103) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
            %coord_105 = cute.make_coord() : () -> !cute.coord<"(_,1)">
            %slice_106 = cute.slice(%src_partitioned, %coord_105) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
            %iter_107 = cute.get_iter(%slice_106) : !memref_gmem_f32_3
            %lay_108 = cute.get_layout(%slice_106) : !memref_gmem_f32_3
            %append_109 = cute.append_to_rank<2> (%lay_108, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_110 = cute.make_view(%iter_107, %append_109) : !memref_gmem_f32_4
            %grouped_111 = cute.group_modes(%view_110) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_112 = cute.make_view(%ptr_104, %append_100) : !memref_rmem_f32_12
            %grouped_113 = cute.group_modes(%view_112) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
            cute.copy(%66, %grouped_111, %grouped_113) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_13)
            %int_tuple_114 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
            %ptr_115 = cute.add_offset(%iter_87, %int_tuple_114) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
            %coord_116 = cute.make_coord() : () -> !cute.coord<"(_,2)">
            %slice_117 = cute.slice(%src_partitioned, %coord_116) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
            %iter_118 = cute.get_iter(%slice_117) : !memref_gmem_f32_3
            %lay_119 = cute.get_layout(%slice_117) : !memref_gmem_f32_3
            %append_120 = cute.append_to_rank<2> (%lay_119, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_121 = cute.make_view(%iter_118, %append_120) : !memref_gmem_f32_4
            %grouped_122 = cute.group_modes(%view_121) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_123 = cute.make_view(%ptr_115, %append_100) : !memref_rmem_f32_10
            %grouped_124 = cute.group_modes(%view_123) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
            cute.copy(%66, %grouped_122, %grouped_124) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_11)
            %int_tuple_125 = cute.make_int_tuple() : () -> !cute.int_tuple<"12">
            %ptr_126 = cute.add_offset(%iter_87, %int_tuple_125) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
            %coord_127 = cute.make_coord() : () -> !cute.coord<"(_,3)">
            %slice_128 = cute.slice(%src_partitioned, %coord_127) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
            %iter_129 = cute.get_iter(%slice_128) : !memref_gmem_f32_3
            %lay_130 = cute.get_layout(%slice_128) : !memref_gmem_f32_3
            %append_131 = cute.append_to_rank<2> (%lay_130, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_132 = cute.make_view(%iter_129, %append_131) : !memref_gmem_f32_4
            %grouped_133 = cute.group_modes(%view_132) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_134 = cute.make_view(%ptr_126, %append_100) : !memref_rmem_f32_12
            %grouped_135 = cute.group_modes(%view_134) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
            cute.copy(%66, %grouped_133, %grouped_135) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_13)
            %67 = cute.memref.load_vec(%rmem_46) : (!memref_rmem_f32_1) -> vector<16xf32>
            %68 = cute.memref.load_vec(%rmem_86) : (!memref_rmem_f32_1) -> vector<16xf32>
            %69 = arith.addf %67, %68 : vector<16xf32>
            cute.memref.store_vec(%69, %rmem_46) : (vector<16xf32>, !memref_rmem_f32_1) -> ()
            iket.range_push("epi_math")
            iket.range_pop
            %rmem_136 = cute.memref.alloca() : !memref_rmem_bf16
            %iter_137 = cute.get_iter(%rmem_136) : !memref_rmem_bf16
            %70 = cute.memref.load_vec(%rmem_46) : (!memref_rmem_f32_1) -> vector<16xf32>
            %71 = arith.truncf %70 : vector<16xf32> to vector<16xbf16>
            %lay_138 = cute.get_layout(%rmem_136) : !memref_rmem_bf16
            cute.memref.store_vec(%71, %rmem_136) : (vector<16xbf16>, !memref_rmem_bf16) -> ()
            iket.range_push("epi_store_acq")
            scf.if %25 {
              nvvm.cp.async.bulk.wait_group 3 {read}
            }
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            iket.range_pop
            %72 = arith.remsi %63, %c4_i32 : i32
            iket.range_push("epi_r2s")
            %coord_139 = cute.make_coord(%72) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_140 = cute.slice(%dst_partitioned, %coord_139) : !memref_smem_bf16_11, !cute.coord<"(_,_,_,?)">
            %iter_141 = cute.get_iter(%slice_140) : !memref_smem_bf16_12
            %lay_142 = cute.get_layout(%slice_140) : !memref_smem_bf16_12
            %append_143 = cute.append_to_rank<2> (%lay_138, %lay_42) : !cute.layout<"(((2,2,2),1),1,2):(((1,2,4),0),0,8)">, !cute.layout<"1:0">
            %view_144 = cute.make_view(%iter_137, %append_143) : !memref_rmem_bf16
            %grouped_145 = cute.group_modes(%view_144) <1, 3> : (!memref_rmem_bf16) -> !memref_rmem_bf16_1
            %append_146 = cute.append_to_rank<2> (%lay_142, %lay_42) : !cute.layout<"(((2,4),1),1,2):(((1,2),0),0,16)">, !cute.layout<"1:0">
            %view_147 = cute.make_view(%iter_141, %append_146) : !memref_smem_bf16_12
            %grouped_148 = cute.group_modes(%view_147) <1, 3> : (!memref_smem_bf16_12) -> !memref_smem_bf16_13
            cute.copy(%44, %grouped_145, %grouped_148) : (!copy_stsm_4, !memref_rmem_bf16_1, !memref_smem_bf16_13)
            iket.range_pop
            iket.range_push("epi_store_issue")
            nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            scf.if %25 {
              %coord_335 = cute.make_coord(%72) : (i32) -> !cute.coord<"(_,?)">
              %slice_336 = cute.slice(%res_smem_tensor, %coord_335) : !memref_smem_bf16_10, !cute.coord<"(_,?)">
              %iter_337 = cute.get_iter(%slice_336) : !memref_smem_bf16_14
              %coord_338 = cute.make_coord() : () -> !cute.coord<"(_,(0,0))">
              %slice_339 = cute.slice(%res_target_tensors, %coord_338) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1,4)):(((1@0,1@1),0),(0,32@0))">, !cute.coord<"(_,(0,0))">
              %iter_340 = cute.get_iter(%slice_339) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
              %tup = cute.deref_arith_tuple_iter(%iter_340) : !cute.arith_tuple_iter<"(?{div=128},?{div=128},?)">
              %e0_341, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=128},?{div=128},?)">
              %lay_342 = cute.get_layout(%slice_336) : !memref_smem_bf16_14
              %lay_343 = cute.get_layout(%slice_339) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
              %append_344 = cute.append_to_rank<2> (%lay_342, %lay_42) : !cute.layout<"((4096,1)):((1,0))">, !cute.layout<"1:0">
              %view_345 = cute.make_view(%iter_337, %append_344) : !memref_smem_bf16_15
              %grouped_346 = cute.group_modes(%view_345) <1, 2> : (!memref_smem_bf16_15) -> !memref_smem_bf16_16
              %append_347 = cute.append_to_rank<2> (%lay_343, %lay_42) : !cute.layout<"(((32,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
              %int_tuple_348 = cute.make_int_tuple(%e0_341, %e1, %e2) : (!cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=128},?{div=128},?)">
              %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_348) : (!cute.int_tuple<"(?{div=128},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=128},?{div=128},?)">
              %view_349 = cute.make_view(%int_tup_iter, %append_347) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">
              %grouped_350 = cute.group_modes(%view_349) <1, 2> : (!cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">
              %100 = cute_nvgpu.atom.make_exec_tma(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 65536, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> !cute_nvgpu.atom.tma_store<bf16, copy_bits = 65536, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>
              cute.copy(%100, %grouped_346, %grouped_350) : (!cute_nvgpu.atom.tma_store<bf16, copy_bits = 65536, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>, !memref_smem_bf16_16, !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">)
              nvvm.cp.async.bulk.commit.group
            }
            iket.range_pop
            %coord_149 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,1))">
            %slice_150 = cute.slice(%retiled, %coord_149) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,1))">
            %lay_151 = cute.get_layout(%slice_150) : !memref_rmem_f32_7
            %73 = cute.composition(%lay_151, %rinv) : (!cute.layout<"((8,1),1,2):((1,0),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %coalesce_152 = cute.coalesce(%73) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
            %74 = cute.composition(%rinv, %coalesce_152) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %div_153 = cute.logical_divide(%slice_150, %74) : !memref_rmem_f32_7, !cute.layout<"16:1">
            %div_154 = cute.logical_divide(%rmem_46, %74) : !memref_rmem_f32_1, !cute.layout<"16:1">
            %div_155 = cute.logical_divide(%div_153, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            %div_156 = cute.logical_divide(%div_154, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            cute.copy(%atom_85, %div_155, %div_156) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
            %int_tuple_157 = cute.make_int_tuple() : () -> !cute.int_tuple<"4096">
            %ptr_158 = cute.add_offset(%ptr_68, %int_tuple_157) : (!cute.ptr<f32, gmem, align<16>>, !cute.int_tuple<"4096">) -> !cute.ptr<f32, gmem, align<16>>
            %rmem_159 = cute.memref.alloca() : !memref_rmem_f32_1
            %iter_160 = cute.get_iter(%rmem_159) : !memref_rmem_f32_1
            %view_161 = cute.make_view(%ptr_158, %lay_89) : !memref_gmem_f32_1
            %src_partitioned_162 = cute.tiled.copy.partition_S(%66, %view_161, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
            %slice_163 = cute.slice(%src_partitioned_162, %coord_94) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
            %iter_164 = cute.get_iter(%slice_163) : !memref_gmem_f32_3
            %lay_165 = cute.get_layout(%slice_163) : !memref_gmem_f32_3
            %append_166 = cute.append_to_rank<2> (%lay_165, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_167 = cute.make_view(%iter_164, %append_166) : !memref_gmem_f32_4
            %grouped_168 = cute.group_modes(%view_167) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_169 = cute.make_view(%iter_160, %append_100) : !memref_rmem_f32_10
            %grouped_170 = cute.group_modes(%view_169) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
            cute.copy(%66, %grouped_168, %grouped_170) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_11)
            %ptr_171 = cute.add_offset(%iter_160, %int_tuple_103) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
            %slice_172 = cute.slice(%src_partitioned_162, %coord_105) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
            %iter_173 = cute.get_iter(%slice_172) : !memref_gmem_f32_3
            %lay_174 = cute.get_layout(%slice_172) : !memref_gmem_f32_3
            %append_175 = cute.append_to_rank<2> (%lay_174, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_176 = cute.make_view(%iter_173, %append_175) : !memref_gmem_f32_4
            %grouped_177 = cute.group_modes(%view_176) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_178 = cute.make_view(%ptr_171, %append_100) : !memref_rmem_f32_12
            %grouped_179 = cute.group_modes(%view_178) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
            cute.copy(%66, %grouped_177, %grouped_179) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_13)
            %ptr_180 = cute.add_offset(%iter_160, %int_tuple_114) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
            %slice_181 = cute.slice(%src_partitioned_162, %coord_116) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
            %iter_182 = cute.get_iter(%slice_181) : !memref_gmem_f32_3
            %lay_183 = cute.get_layout(%slice_181) : !memref_gmem_f32_3
            %append_184 = cute.append_to_rank<2> (%lay_183, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_185 = cute.make_view(%iter_182, %append_184) : !memref_gmem_f32_4
            %grouped_186 = cute.group_modes(%view_185) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_187 = cute.make_view(%ptr_180, %append_100) : !memref_rmem_f32_10
            %grouped_188 = cute.group_modes(%view_187) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
            cute.copy(%66, %grouped_186, %grouped_188) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_11)
            %ptr_189 = cute.add_offset(%iter_160, %int_tuple_125) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
            %slice_190 = cute.slice(%src_partitioned_162, %coord_127) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
            %iter_191 = cute.get_iter(%slice_190) : !memref_gmem_f32_3
            %lay_192 = cute.get_layout(%slice_190) : !memref_gmem_f32_3
            %append_193 = cute.append_to_rank<2> (%lay_192, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_194 = cute.make_view(%iter_191, %append_193) : !memref_gmem_f32_4
            %grouped_195 = cute.group_modes(%view_194) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_196 = cute.make_view(%ptr_189, %append_100) : !memref_rmem_f32_12
            %grouped_197 = cute.group_modes(%view_196) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
            cute.copy(%66, %grouped_195, %grouped_197) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_13)
            %75 = cute.memref.load_vec(%rmem_46) : (!memref_rmem_f32_1) -> vector<16xf32>
            %76 = cute.memref.load_vec(%rmem_159) : (!memref_rmem_f32_1) -> vector<16xf32>
            %77 = arith.addf %75, %76 : vector<16xf32>
            cute.memref.store_vec(%77, %rmem_46) : (vector<16xf32>, !memref_rmem_f32_1) -> ()
            iket.range_push("epi_math")
            iket.range_pop
            %rmem_198 = cute.memref.alloca() : !memref_rmem_bf16
            %iter_199 = cute.get_iter(%rmem_198) : !memref_rmem_bf16
            %78 = cute.memref.load_vec(%rmem_46) : (!memref_rmem_f32_1) -> vector<16xf32>
            %79 = arith.truncf %78 : vector<16xf32> to vector<16xbf16>
            %lay_200 = cute.get_layout(%rmem_198) : !memref_rmem_bf16
            cute.memref.store_vec(%79, %rmem_198) : (vector<16xbf16>, !memref_rmem_bf16) -> ()
            iket.range_push("epi_store_acq")
            scf.if %25 {
              nvvm.cp.async.bulk.wait_group 3 {read}
            }
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            iket.range_pop
            %80 = arith.addi %63, %c1_i32 : i32
            %81 = arith.remsi %80, %c4_i32 : i32
            iket.range_push("epi_r2s")
            %coord_201 = cute.make_coord(%81) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_202 = cute.slice(%dst_partitioned, %coord_201) : !memref_smem_bf16_11, !cute.coord<"(_,_,_,?)">
            %iter_203 = cute.get_iter(%slice_202) : !memref_smem_bf16_12
            %lay_204 = cute.get_layout(%slice_202) : !memref_smem_bf16_12
            %append_205 = cute.append_to_rank<2> (%lay_200, %lay_42) : !cute.layout<"(((2,2,2),1),1,2):(((1,2,4),0),0,8)">, !cute.layout<"1:0">
            %view_206 = cute.make_view(%iter_199, %append_205) : !memref_rmem_bf16
            %grouped_207 = cute.group_modes(%view_206) <1, 3> : (!memref_rmem_bf16) -> !memref_rmem_bf16_1
            %append_208 = cute.append_to_rank<2> (%lay_204, %lay_42) : !cute.layout<"(((2,4),1),1,2):(((1,2),0),0,16)">, !cute.layout<"1:0">
            %view_209 = cute.make_view(%iter_203, %append_208) : !memref_smem_bf16_12
            %grouped_210 = cute.group_modes(%view_209) <1, 3> : (!memref_smem_bf16_12) -> !memref_smem_bf16_13
            cute.copy(%44, %grouped_207, %grouped_210) : (!copy_stsm_4, !memref_rmem_bf16_1, !memref_smem_bf16_13)
            iket.range_pop
            iket.range_push("epi_store_issue")
            nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            scf.if %25 {
              %coord_335 = cute.make_coord(%81) : (i32) -> !cute.coord<"(_,?)">
              %slice_336 = cute.slice(%res_smem_tensor, %coord_335) : !memref_smem_bf16_10, !cute.coord<"(_,?)">
              %iter_337 = cute.get_iter(%slice_336) : !memref_smem_bf16_14
              %coord_338 = cute.make_coord() : () -> !cute.coord<"(_,(0,1))">
              %slice_339 = cute.slice(%res_target_tensors, %coord_338) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1,4)):(((1@0,1@1),0),(0,32@0))">, !cute.coord<"(_,(0,1))">
              %iter_340 = cute.get_iter(%slice_339) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
              %tup = cute.deref_arith_tuple_iter(%iter_340) : !cute.arith_tuple_iter<"(?{div=32},?{div=128},?)">
              %e0_341, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=32},?{div=128},?)">
              %lay_342 = cute.get_layout(%slice_336) : !memref_smem_bf16_14
              %lay_343 = cute.get_layout(%slice_339) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
              %append_344 = cute.append_to_rank<2> (%lay_342, %lay_42) : !cute.layout<"((4096,1)):((1,0))">, !cute.layout<"1:0">
              %view_345 = cute.make_view(%iter_337, %append_344) : !memref_smem_bf16_15
              %grouped_346 = cute.group_modes(%view_345) <1, 2> : (!memref_smem_bf16_15) -> !memref_smem_bf16_16
              %append_347 = cute.append_to_rank<2> (%lay_343, %lay_42) : !cute.layout<"(((32,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
              %int_tuple_348 = cute.make_int_tuple(%e0_341, %e1, %e2) : (!cute.int_tuple<"?{div=32}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=32},?{div=128},?)">
              %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_348) : (!cute.int_tuple<"(?{div=32},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=32},?{div=128},?)">
              %view_349 = cute.make_view(%int_tup_iter, %append_347) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">
              %grouped_350 = cute.group_modes(%view_349) <1, 2> : (!cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">
              %100 = cute_nvgpu.atom.make_exec_tma(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 65536, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> !cute_nvgpu.atom.tma_store<bf16, copy_bits = 65536, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>
              cute.copy(%100, %grouped_346, %grouped_350) : (!cute_nvgpu.atom.tma_store<bf16, copy_bits = 65536, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>, !memref_smem_bf16_16, !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">)
              nvvm.cp.async.bulk.commit.group
            }
            iket.range_pop
            %coord_211 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,2))">
            %slice_212 = cute.slice(%retiled, %coord_211) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,2))">
            %lay_213 = cute.get_layout(%slice_212) : !memref_rmem_f32_7
            %82 = cute.composition(%lay_213, %rinv) : (!cute.layout<"((8,1),1,2):((1,0),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %coalesce_214 = cute.coalesce(%82) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
            %83 = cute.composition(%rinv, %coalesce_214) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %div_215 = cute.logical_divide(%slice_212, %83) : !memref_rmem_f32_7, !cute.layout<"16:1">
            %div_216 = cute.logical_divide(%rmem_46, %83) : !memref_rmem_f32_1, !cute.layout<"16:1">
            %div_217 = cute.logical_divide(%div_215, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            %div_218 = cute.logical_divide(%div_216, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            cute.copy(%atom_85, %div_217, %div_218) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
            %int_tuple_219 = cute.make_int_tuple() : () -> !cute.int_tuple<"8192">
            %ptr_220 = cute.add_offset(%ptr_68, %int_tuple_219) : (!cute.ptr<f32, gmem, align<16>>, !cute.int_tuple<"8192">) -> !cute.ptr<f32, gmem, align<16>>
            %rmem_221 = cute.memref.alloca() : !memref_rmem_f32_1
            %iter_222 = cute.get_iter(%rmem_221) : !memref_rmem_f32_1
            %view_223 = cute.make_view(%ptr_220, %lay_89) : !memref_gmem_f32_1
            %src_partitioned_224 = cute.tiled.copy.partition_S(%66, %view_223, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
            %slice_225 = cute.slice(%src_partitioned_224, %coord_94) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
            %iter_226 = cute.get_iter(%slice_225) : !memref_gmem_f32_3
            %lay_227 = cute.get_layout(%slice_225) : !memref_gmem_f32_3
            %append_228 = cute.append_to_rank<2> (%lay_227, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_229 = cute.make_view(%iter_226, %append_228) : !memref_gmem_f32_4
            %grouped_230 = cute.group_modes(%view_229) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_231 = cute.make_view(%iter_222, %append_100) : !memref_rmem_f32_10
            %grouped_232 = cute.group_modes(%view_231) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
            cute.copy(%66, %grouped_230, %grouped_232) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_11)
            %ptr_233 = cute.add_offset(%iter_222, %int_tuple_103) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
            %slice_234 = cute.slice(%src_partitioned_224, %coord_105) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
            %iter_235 = cute.get_iter(%slice_234) : !memref_gmem_f32_3
            %lay_236 = cute.get_layout(%slice_234) : !memref_gmem_f32_3
            %append_237 = cute.append_to_rank<2> (%lay_236, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_238 = cute.make_view(%iter_235, %append_237) : !memref_gmem_f32_4
            %grouped_239 = cute.group_modes(%view_238) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_240 = cute.make_view(%ptr_233, %append_100) : !memref_rmem_f32_12
            %grouped_241 = cute.group_modes(%view_240) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
            cute.copy(%66, %grouped_239, %grouped_241) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_13)
            %ptr_242 = cute.add_offset(%iter_222, %int_tuple_114) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
            %slice_243 = cute.slice(%src_partitioned_224, %coord_116) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
            %iter_244 = cute.get_iter(%slice_243) : !memref_gmem_f32_3
            %lay_245 = cute.get_layout(%slice_243) : !memref_gmem_f32_3
            %append_246 = cute.append_to_rank<2> (%lay_245, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_247 = cute.make_view(%iter_244, %append_246) : !memref_gmem_f32_4
            %grouped_248 = cute.group_modes(%view_247) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_249 = cute.make_view(%ptr_242, %append_100) : !memref_rmem_f32_10
            %grouped_250 = cute.group_modes(%view_249) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
            cute.copy(%66, %grouped_248, %grouped_250) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_11)
            %ptr_251 = cute.add_offset(%iter_222, %int_tuple_125) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
            %slice_252 = cute.slice(%src_partitioned_224, %coord_127) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
            %iter_253 = cute.get_iter(%slice_252) : !memref_gmem_f32_3
            %lay_254 = cute.get_layout(%slice_252) : !memref_gmem_f32_3
            %append_255 = cute.append_to_rank<2> (%lay_254, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_256 = cute.make_view(%iter_253, %append_255) : !memref_gmem_f32_4
            %grouped_257 = cute.group_modes(%view_256) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_258 = cute.make_view(%ptr_251, %append_100) : !memref_rmem_f32_12
            %grouped_259 = cute.group_modes(%view_258) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
            cute.copy(%66, %grouped_257, %grouped_259) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_13)
            %84 = cute.memref.load_vec(%rmem_46) : (!memref_rmem_f32_1) -> vector<16xf32>
            %85 = cute.memref.load_vec(%rmem_221) : (!memref_rmem_f32_1) -> vector<16xf32>
            %86 = arith.addf %84, %85 : vector<16xf32>
            cute.memref.store_vec(%86, %rmem_46) : (vector<16xf32>, !memref_rmem_f32_1) -> ()
            iket.range_push("epi_math")
            iket.range_pop
            %rmem_260 = cute.memref.alloca() : !memref_rmem_bf16
            %iter_261 = cute.get_iter(%rmem_260) : !memref_rmem_bf16
            %87 = cute.memref.load_vec(%rmem_46) : (!memref_rmem_f32_1) -> vector<16xf32>
            %88 = arith.truncf %87 : vector<16xf32> to vector<16xbf16>
            %lay_262 = cute.get_layout(%rmem_260) : !memref_rmem_bf16
            cute.memref.store_vec(%88, %rmem_260) : (vector<16xbf16>, !memref_rmem_bf16) -> ()
            iket.range_push("epi_store_acq")
            scf.if %25 {
              nvvm.cp.async.bulk.wait_group 3 {read}
            }
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            iket.range_pop
            %89 = arith.addi %63, %c2_i32 : i32
            %90 = arith.remsi %89, %c4_i32 : i32
            iket.range_push("epi_r2s")
            %coord_263 = cute.make_coord(%90) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_264 = cute.slice(%dst_partitioned, %coord_263) : !memref_smem_bf16_11, !cute.coord<"(_,_,_,?)">
            %iter_265 = cute.get_iter(%slice_264) : !memref_smem_bf16_12
            %lay_266 = cute.get_layout(%slice_264) : !memref_smem_bf16_12
            %append_267 = cute.append_to_rank<2> (%lay_262, %lay_42) : !cute.layout<"(((2,2,2),1),1,2):(((1,2,4),0),0,8)">, !cute.layout<"1:0">
            %view_268 = cute.make_view(%iter_261, %append_267) : !memref_rmem_bf16
            %grouped_269 = cute.group_modes(%view_268) <1, 3> : (!memref_rmem_bf16) -> !memref_rmem_bf16_1
            %append_270 = cute.append_to_rank<2> (%lay_266, %lay_42) : !cute.layout<"(((2,4),1),1,2):(((1,2),0),0,16)">, !cute.layout<"1:0">
            %view_271 = cute.make_view(%iter_265, %append_270) : !memref_smem_bf16_12
            %grouped_272 = cute.group_modes(%view_271) <1, 3> : (!memref_smem_bf16_12) -> !memref_smem_bf16_13
            cute.copy(%44, %grouped_269, %grouped_272) : (!copy_stsm_4, !memref_rmem_bf16_1, !memref_smem_bf16_13)
            iket.range_pop
            iket.range_push("epi_store_issue")
            nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            scf.if %25 {
              %coord_335 = cute.make_coord(%90) : (i32) -> !cute.coord<"(_,?)">
              %slice_336 = cute.slice(%res_smem_tensor, %coord_335) : !memref_smem_bf16_10, !cute.coord<"(_,?)">
              %iter_337 = cute.get_iter(%slice_336) : !memref_smem_bf16_14
              %coord_338 = cute.make_coord() : () -> !cute.coord<"(_,(0,2))">
              %slice_339 = cute.slice(%res_target_tensors, %coord_338) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1,4)):(((1@0,1@1),0),(0,32@0))">, !cute.coord<"(_,(0,2))">
              %iter_340 = cute.get_iter(%slice_339) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
              %tup = cute.deref_arith_tuple_iter(%iter_340) : !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
              %e0_341, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=64},?{div=128},?)">
              %lay_342 = cute.get_layout(%slice_336) : !memref_smem_bf16_14
              %lay_343 = cute.get_layout(%slice_339) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
              %append_344 = cute.append_to_rank<2> (%lay_342, %lay_42) : !cute.layout<"((4096,1)):((1,0))">, !cute.layout<"1:0">
              %view_345 = cute.make_view(%iter_337, %append_344) : !memref_smem_bf16_15
              %grouped_346 = cute.group_modes(%view_345) <1, 2> : (!memref_smem_bf16_15) -> !memref_smem_bf16_16
              %append_347 = cute.append_to_rank<2> (%lay_343, %lay_42) : !cute.layout<"(((32,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
              %int_tuple_348 = cute.make_int_tuple(%e0_341, %e1, %e2) : (!cute.int_tuple<"?{div=64}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=64},?{div=128},?)">
              %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_348) : (!cute.int_tuple<"(?{div=64},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=64},?{div=128},?)">
              %view_349 = cute.make_view(%int_tup_iter, %append_347) : !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">
              %grouped_350 = cute.group_modes(%view_349) <1, 2> : (!cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">
              %100 = cute_nvgpu.atom.make_exec_tma(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 65536, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> !cute_nvgpu.atom.tma_store<bf16, copy_bits = 65536, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>
              cute.copy(%100, %grouped_346, %grouped_350) : (!cute_nvgpu.atom.tma_store<bf16, copy_bits = 65536, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>, !memref_smem_bf16_16, !cute.coord_tensor<"(?{div=64},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">)
              nvvm.cp.async.bulk.commit.group
            }
            iket.range_pop
            %coord_273 = cute.make_coord() : () -> !cute.coord<"(_,_,_,(0,3))">
            %slice_274 = cute.slice(%retiled, %coord_273) : !memref_rmem_f32_6, !cute.coord<"(_,_,_,(0,3))">
            %lay_275 = cute.get_layout(%slice_274) : !memref_rmem_f32_7
            %91 = cute.composition(%lay_275, %rinv) : (!cute.layout<"((8,1),1,2):((1,0),0,8)">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %coalesce_276 = cute.coalesce(%91) : (!cute.layout<"16:1">) -> !cute.layout<"16:1">
            %92 = cute.composition(%rinv, %coalesce_276) : (!cute.layout<"16:1">, !cute.layout<"16:1">) -> !cute.layout<"16:1">
            %div_277 = cute.logical_divide(%slice_274, %92) : !memref_rmem_f32_7, !cute.layout<"16:1">
            %div_278 = cute.logical_divide(%rmem_46, %92) : !memref_rmem_f32_1, !cute.layout<"16:1">
            %div_279 = cute.logical_divide(%div_277, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            %div_280 = cute.logical_divide(%div_278, %lay_82) : !memref_rmem_f32_8, !cute.layout<"8:1">
            cute.copy(%atom_85, %div_279, %div_280) : (!cute_nvgpu.atom.universal_copy<f32, 256 b>, !memref_rmem_f32_9, !memref_rmem_f32_9)
            %int_tuple_281 = cute.make_int_tuple() : () -> !cute.int_tuple<"12288">
            %ptr_282 = cute.add_offset(%ptr_68, %int_tuple_281) : (!cute.ptr<f32, gmem, align<16>>, !cute.int_tuple<"12288">) -> !cute.ptr<f32, gmem, align<16>>
            %rmem_283 = cute.memref.alloca() : !memref_rmem_f32_1
            %iter_284 = cute.get_iter(%rmem_283) : !memref_rmem_f32_1
            %view_285 = cute.make_view(%ptr_282, %lay_89) : !memref_gmem_f32_1
            %src_partitioned_286 = cute.tiled.copy.partition_S(%66, %view_285, %coord_45) : (!copy_simt, !memref_gmem_f32_1, !cute.coord<"?">) -> !memref_gmem_f32_2
            %slice_287 = cute.slice(%src_partitioned_286, %coord_94) : !memref_gmem_f32_2, !cute.coord<"(_,0)">
            %iter_288 = cute.get_iter(%slice_287) : !memref_gmem_f32_3
            %lay_289 = cute.get_layout(%slice_287) : !memref_gmem_f32_3
            %append_290 = cute.append_to_rank<2> (%lay_289, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_291 = cute.make_view(%iter_288, %append_290) : !memref_gmem_f32_4
            %grouped_292 = cute.group_modes(%view_291) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_293 = cute.make_view(%iter_284, %append_100) : !memref_rmem_f32_10
            %grouped_294 = cute.group_modes(%view_293) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
            cute.copy(%66, %grouped_292, %grouped_294) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_11)
            %ptr_295 = cute.add_offset(%iter_284, %int_tuple_103) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"4">) -> !cute.ptr<f32, rmem, align<16>>
            %slice_296 = cute.slice(%src_partitioned_286, %coord_105) : !memref_gmem_f32_2, !cute.coord<"(_,1)">
            %iter_297 = cute.get_iter(%slice_296) : !memref_gmem_f32_3
            %lay_298 = cute.get_layout(%slice_296) : !memref_gmem_f32_3
            %append_299 = cute.append_to_rank<2> (%lay_298, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_300 = cute.make_view(%iter_297, %append_299) : !memref_gmem_f32_4
            %grouped_301 = cute.group_modes(%view_300) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_302 = cute.make_view(%ptr_295, %append_100) : !memref_rmem_f32_12
            %grouped_303 = cute.group_modes(%view_302) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
            cute.copy(%66, %grouped_301, %grouped_303) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_13)
            %ptr_304 = cute.add_offset(%iter_284, %int_tuple_114) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"8">) -> !cute.ptr<f32, rmem, align<32>>
            %slice_305 = cute.slice(%src_partitioned_286, %coord_116) : !memref_gmem_f32_2, !cute.coord<"(_,2)">
            %iter_306 = cute.get_iter(%slice_305) : !memref_gmem_f32_3
            %lay_307 = cute.get_layout(%slice_305) : !memref_gmem_f32_3
            %append_308 = cute.append_to_rank<2> (%lay_307, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_309 = cute.make_view(%iter_306, %append_308) : !memref_gmem_f32_4
            %grouped_310 = cute.group_modes(%view_309) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_311 = cute.make_view(%ptr_304, %append_100) : !memref_rmem_f32_10
            %grouped_312 = cute.group_modes(%view_311) <1, 2> : (!memref_rmem_f32_10) -> !memref_rmem_f32_11
            cute.copy(%66, %grouped_310, %grouped_312) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_11)
            %ptr_313 = cute.add_offset(%iter_284, %int_tuple_125) : (!cute.ptr<f32, rmem, align<32>>, !cute.int_tuple<"12">) -> !cute.ptr<f32, rmem, align<16>>
            %slice_314 = cute.slice(%src_partitioned_286, %coord_127) : !memref_gmem_f32_2, !cute.coord<"(_,3)">
            %iter_315 = cute.get_iter(%slice_314) : !memref_gmem_f32_3
            %lay_316 = cute.get_layout(%slice_314) : !memref_gmem_f32_3
            %append_317 = cute.append_to_rank<2> (%lay_316, %lay_42) : !cute.layout<"((4,1)):((1,0))">, !cute.layout<"1:0">
            %view_318 = cute.make_view(%iter_315, %append_317) : !memref_gmem_f32_4
            %grouped_319 = cute.group_modes(%view_318) <1, 2> : (!memref_gmem_f32_4) -> !memref_gmem_f32_5
            %view_320 = cute.make_view(%ptr_313, %append_100) : !memref_rmem_f32_12
            %grouped_321 = cute.group_modes(%view_320) <1, 2> : (!memref_rmem_f32_12) -> !memref_rmem_f32_13
            cute.copy(%66, %grouped_319, %grouped_321) : (!copy_simt, !memref_gmem_f32_5, !memref_rmem_f32_13)
            %93 = cute.memref.load_vec(%rmem_46) : (!memref_rmem_f32_1) -> vector<16xf32>
            %94 = cute.memref.load_vec(%rmem_283) : (!memref_rmem_f32_1) -> vector<16xf32>
            %95 = arith.addf %93, %94 : vector<16xf32>
            cute.memref.store_vec(%95, %rmem_46) : (vector<16xf32>, !memref_rmem_f32_1) -> ()
            iket.range_push("epi_math")
            iket.range_pop
            %rmem_322 = cute.memref.alloca() : !memref_rmem_bf16
            %iter_323 = cute.get_iter(%rmem_322) : !memref_rmem_bf16
            %96 = cute.memref.load_vec(%rmem_46) : (!memref_rmem_f32_1) -> vector<16xf32>
            %97 = arith.truncf %96 : vector<16xf32> to vector<16xbf16>
            %lay_324 = cute.get_layout(%rmem_322) : !memref_rmem_bf16
            cute.memref.store_vec(%97, %rmem_322) : (vector<16xbf16>, !memref_rmem_bf16) -> ()
            iket.range_push("epi_store_acq")
            scf.if %25 {
              nvvm.cp.async.bulk.wait_group 3 {read}
            }
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            iket.range_pop
            %98 = arith.addi %63, %c3_i32 : i32
            %99 = arith.remsi %98, %c4_i32 : i32
            iket.range_push("epi_r2s")
            %coord_325 = cute.make_coord(%99) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_326 = cute.slice(%dst_partitioned, %coord_325) : !memref_smem_bf16_11, !cute.coord<"(_,_,_,?)">
            %iter_327 = cute.get_iter(%slice_326) : !memref_smem_bf16_12
            %lay_328 = cute.get_layout(%slice_326) : !memref_smem_bf16_12
            %append_329 = cute.append_to_rank<2> (%lay_324, %lay_42) : !cute.layout<"(((2,2,2),1),1,2):(((1,2,4),0),0,8)">, !cute.layout<"1:0">
            %view_330 = cute.make_view(%iter_323, %append_329) : !memref_rmem_bf16
            %grouped_331 = cute.group_modes(%view_330) <1, 3> : (!memref_rmem_bf16) -> !memref_rmem_bf16_1
            %append_332 = cute.append_to_rank<2> (%lay_328, %lay_42) : !cute.layout<"(((2,4),1),1,2):(((1,2),0),0,16)">, !cute.layout<"1:0">
            %view_333 = cute.make_view(%iter_327, %append_332) : !memref_smem_bf16_12
            %grouped_334 = cute.group_modes(%view_333) <1, 3> : (!memref_smem_bf16_12) -> !memref_smem_bf16_13
            cute.copy(%44, %grouped_331, %grouped_334) : (!copy_stsm_4, !memref_rmem_bf16_1, !memref_smem_bf16_13)
            iket.range_pop
            iket.range_push("epi_store_issue")
            nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            scf.if %25 {
              %coord_335 = cute.make_coord(%99) : (i32) -> !cute.coord<"(_,?)">
              %slice_336 = cute.slice(%res_smem_tensor, %coord_335) : !memref_smem_bf16_10, !cute.coord<"(_,?)">
              %iter_337 = cute.get_iter(%slice_336) : !memref_smem_bf16_14
              %coord_338 = cute.make_coord() : () -> !cute.coord<"(_,(0,3))">
              %slice_339 = cute.slice(%res_target_tensors, %coord_338) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((32,128),1),(1,4)):(((1@0,1@1),0),(0,32@0))">, !cute.coord<"(_,(0,3))">
              %iter_340 = cute.get_iter(%slice_339) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
              %tup = cute.deref_arith_tuple_iter(%iter_340) : !cute.arith_tuple_iter<"(?{div=32},?{div=128},?)">
              %e0_341, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=32},?{div=128},?)">
              %lay_342 = cute.get_layout(%slice_336) : !memref_smem_bf16_14
              %lay_343 = cute.get_layout(%slice_339) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1)):(((1@0,1@1),0))">
              %append_344 = cute.append_to_rank<2> (%lay_342, %lay_42) : !cute.layout<"((4096,1)):((1,0))">, !cute.layout<"1:0">
              %view_345 = cute.make_view(%iter_337, %append_344) : !memref_smem_bf16_15
              %grouped_346 = cute.group_modes(%view_345) <1, 2> : (!memref_smem_bf16_15) -> !memref_smem_bf16_16
              %append_347 = cute.append_to_rank<2> (%lay_343, %lay_42) : !cute.layout<"(((32,128),1)):(((1@0,1@1),0))">, !cute.layout<"1:0">
              %int_tuple_348 = cute.make_int_tuple(%e0_341, %e1, %e2) : (!cute.int_tuple<"?{div=32}">, !cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=32},?{div=128},?)">
              %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_348) : (!cute.int_tuple<"(?{div=32},?{div=128},?)">) -> !cute.arith_tuple_iter<"(?{div=32},?{div=128},?)">
              %view_349 = cute.make_view(%int_tup_iter, %append_347) : !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">
              %grouped_350 = cute.group_modes(%view_349) <1, 2> : (!cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),1):(((1@0,1@1),0),0)">) -> !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">
              %100 = cute_nvgpu.atom.make_exec_tma(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 65536, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>) -> !cute_nvgpu.atom.tma_store<bf16, copy_bits = 65536, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>
              cute.copy(%100, %grouped_346, %grouped_350) : (!cute_nvgpu.atom.tma_store<bf16, copy_bits = 65536, mode = tiled, g_stride = <"()"> tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">>, !memref_smem_bf16_16, !cute.coord_tensor<"(?{div=32},?{div=128},?)", "(((32,128),1),(1)):(((1@0,1@1),0),(0))">)
              nvvm.cp.async.bulk.commit.group
            }
            iket.range_pop
            nvvm.barrier id = %c1_i32 number_of_threads = %c256_i32
            scf.if %62 {
              %100 = builtin.unrealized_conversion_cast %ptr_62 : !cute.ptr<i32, gmem> to !llvm.ptr<1>
              nvvm.store.ext %c0_i32, %100 {order = #nvvm.mem_order<release>, scope = #nvvm.mem_scope<gpu>} : i32, !llvm.ptr<1>
            }
            scf.if %25 {
              nvvm.cp.async.bulk.wait_group 0 {read}
            }
          }
          iket.range_pop
          %46 = arith.addi %arg38, %c1_i32 : i32
          iket.range_push("fetch_wait")
          %int_tuple_69 = cute.make_int_tuple(%arg40) : (i32) -> !cute.int_tuple<"?">
          %ptr_70 = cute.add_offset(%iter_11, %int_tuple_69) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %47 = builtin.unrealized_conversion_cast %ptr_70 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity %47, %arg41, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          iket.range_pop
          iket.range_push("fetch_decode")
          %coord_71 = cute.make_coord(%arg40) : (i32) -> !cute.coord<"(0,?)">
          %48 = cute.memref.load(%view, %coord_71) : (!memref_smem_i32, !cute.coord<"(0,?)">) -> i32
          %coord_72 = cute.make_coord(%arg40) : (i32) -> !cute.coord<"(1,?)">
          %49 = cute.memref.load(%view, %coord_72) : (!memref_smem_i32, !cute.coord<"(1,?)">) -> i32
          %coord_73 = cute.make_coord(%arg40) : (i32) -> !cute.coord<"(2,?)">
          %50 = cute.memref.load(%view, %coord_73) : (!memref_smem_i32, !cute.coord<"(2,?)">) -> i32
          %coord_74 = cute.make_coord(%arg40) : (i32) -> !cute.coord<"(3,?)">
          %51 = cute.memref.load(%view, %coord_74) : (!memref_smem_i32, !cute.coord<"(3,?)">) -> i32
          nvvm.bar.warp.sync %c-1_i32 : i32
          %52 = nvvm.elect.sync -> i1
          scf.if %52 {
            %ptr_75 = cute.add_offset(%ptr_13, %int_tuple_69) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %62 = builtin.unrealized_conversion_cast %ptr_75 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            %63 = nvvm.mbarrier.arrive %62, %c1_i32 : !llvm.ptr<3> -> i64
          }
          %53 = arith.addi %arg40, %c1_i32 : i32
          %54 = arith.addi %arg39, %c1_i32 : i32
          %55 = arith.cmpi eq, %53, %c1_i32 : i32
          %56 = arith.select %55, %c0_i32, %53 : i32
          %57 = scf.if %55 -> (i32) {
            %62 = arith.xori %arg41, %c1_i32 : i32
            scf.yield %62 : i32
          } else {
            scf.yield %arg41 : i32
          }
          %58 = arith.cmpi ne, %51, %c0_i32 : i32
          iket.range_pop
          %59 = arith.floordivsi %50, %c2_i32 : i32
          %60 = arith.muli %59, %c2_i32 : i32
          %61 = arith.subi %50, %60 : i32
          scf.yield %41#2, %41#3, %41#4, %48, %49, %61, %59, %58, %46, %54, %56, %57 : i32, i32, i32, i32, i32, i32, i32, i1, i32, i32, i32, i32
        }
        nvvm.griddepcontrol launch_dependents
        scf.if %25 {
          nvvm.cp.async.bulk.wait_group 0 {read}
        }
      }
      return
    }
  }
  func.func @cutlass___call___quackgemm_default_epiGemmDefaultSm90_object_at__FakeTensorBFloat16_div1_div1_div1_i64_div8_i64_div8_1_FakeTensorBFloat16div1div1div1i64div8i64div81_FakeTensorBFloa(%arg0: !memref_gmem_bf16, %arg1: !memref_gmem_bf16, %arg2: !memref_gmem_bf16, %arg3: f32, %arg4: !memref_gmem_i32, %arg5: !memref_gmem_f32, %arg6: i32, %arg7: i32, %arg8: !cuda.stream) -> i32 attributes {llvm.emit_c_interface} {
    %c31_i32 = arith.constant 31 : i32
    %c384_i32 = arith.constant 384 : i32
    %c232448_i64 = arith.constant 232448 : i64
    %c4294967295_i64 = arith.constant 4294967295 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = cute.static : !cute.swizzle<"S<2,4,3>">
    %1 = cute.static : !cute.swizzle<"S<3,4,3>">
    %false = arith.constant false
    %iter = cute.get_iter(%arg0) : !memref_gmem_bf16
    %iter_0 = cute.get_iter(%arg1) : !memref_gmem_bf16
    %iter_1 = cute.get_iter(%arg2) : !memref_gmem_bf16
    %lay = cute.get_layout(%arg0) : !memref_gmem_bf16
    %2 = cute.select<[1, 2, 0]> (%lay) : (!cute.layout<"(?,?,?):(?{i64 div=8},?{i64 div=8},1)">) -> !cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">
    %view = cute.make_view(%iter, %2) : !memref_gmem_bf16_1
    %lay_2 = cute.get_layout(%arg1) : !memref_gmem_bf16
    %3 = cute.select<[1, 2, 0]> (%lay_2) : (!cute.layout<"(?,?,?):(?{i64 div=8},?{i64 div=8},1)">) -> !cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">
    %view_3 = cute.make_view(%iter_0, %3) : !memref_gmem_bf16_1
    %lay_4 = cute.get_layout(%arg2) : !memref_gmem_bf16
    %4 = cute.select<[1, 2, 0]> (%lay_4) : (!cute.layout<"(?,?,?):(?{i64 div=8},?{i64 div=8},1)">) -> !cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">
    %view_5 = cute.make_view(%iter_1, %4) : !memref_gmem_bf16_1
    %5 = cute.get_shape(%2) : (!cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">) -> !cute.shape<"(?,?,?)">
    %e0, %e1, %e2 = cute.get_leaves(%5) : !cute.shape<"(?,?,?)">
    %itup = cute.to_int_tuple(%e0) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_6 = cute.to_int_tuple(%e1) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_7 = cute.to_int_tuple(%e2) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %6 = cute.get_shape(%3) : (!cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">) -> !cute.shape<"(?,?,?)">
    %e0_8, %e1_9, %e2_10 = cute.get_leaves(%6) : !cute.shape<"(?,?,?)">
    %itup_11 = cute.to_int_tuple(%e0_8) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_12 = cute.to_int_tuple(%e1_9) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_13 = cute.to_int_tuple(%e2_10) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %7 = cute.get_shape(%4) : (!cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">) -> !cute.shape<"(?,?,?)">
    %e0_14, %e1_15, %e2_16 = cute.get_leaves(%7) : !cute.shape<"(?,?,?)">
    %itup_17 = cute.to_int_tuple(%e0_14) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_18 = cute.to_int_tuple(%e1_15) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_19 = cute.to_int_tuple(%e2_16) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %8 = cute.get_scalars(%itup_19) : !cute.int_tuple<"?">
    %atom = cute.make_atom(%false) : (i1) -> !cute_nvgpu.sm90.mma<64x128x16, ab_major = (k, k), elem_type = (bf16, bf16, f32), frag_kind = ss>
    %9 = cute.make_tiled_mma(%atom) : !mma_bf16_bf16_f32_64x128x16
    %shape = cute.make_shape() : () -> !cute.shape<"(1,1,1)">
    %lay_20 = cute.make_layout(%shape) : !cute.layout<"(1,1,1):(0,0,0)">
    %shape_21 = cute.make_shape() : () -> !cute.shape<"(8,64)">
    %stride = cute.make_stride() : () -> !cute.stride<"(64,1)">
    %lay_22 = cute.make_layout(%shape_21, %stride) : !cute.layout<"(8,64):(64,1)">
    %int_tuple = cute.make_int_tuple() : () -> !cute.int_tuple<"0">
    %lay_23 = cute.make_composed_layout(%1, %int_tuple, %lay_22) : !cute.composed_layout<"S<3,4,3> o 0 o (8,64):(64,1)">
    %shape_24 = cute.make_shape() : () -> !cute.shape<"(128,64,6)">
    %int_tuple_25 = cute.make_int_tuple() : () -> !cute.int_tuple<"(0,1,2)">
    %tile_to_shape = cute.tile_to_shape(%lay_23, %shape_24, %int_tuple_25) : (!cute.composed_layout<"S<3,4,3> o 0 o (8,64):(64,1)">, !cute.shape<"(128,64,6)">, !cute.int_tuple<"(0,1,2)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">
    %shape_26 = cute.make_shape() : () -> !cute.shape<"(8,32)">
    %stride_27 = cute.make_stride() : () -> !cute.stride<"(32,1)">
    %lay_28 = cute.make_layout(%shape_26, %stride_27) : !cute.layout<"(8,32):(32,1)">
    %lay_29 = cute.make_composed_layout(%0, %int_tuple, %lay_28) : !cute.composed_layout<"S<2,4,3> o 0 o (8,32):(32,1)">
    %shape_30 = cute.make_shape() : () -> !cute.shape<"(128,32,4)">
    %tile_to_shape_31 = cute.tile_to_shape(%lay_29, %shape_30, %int_tuple_25) : (!cute.composed_layout<"S<2,4,3> o 0 o (8,32):(32,1)">, !cute.shape<"(128,32,4)">, !cute.int_tuple<"(0,1,2)">) -> !cute.composed_layout<"S<2,4,3> o 0 o ((8,16),(32,1),(1,4)):((32,256),(1,0),(0,4096))">
    %coord = cute.make_coord() : () -> !cute.coord<"(_,_,0)">
    %slice = cute.slice(%tile_to_shape, %coord) : !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">, !cute.coord<"(_,_,0)">
    %shape_32 = cute.make_shape(%itup, %itup_6, %itup_7) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?,?)">
    %10 = cute.make_identity_layout(%shape_32) : !cute.layout<"(?,?,?):(1@0,1@1,1@2)">
    %tile = cute.make_tile() : () -> !cute.tile<"[128:1;64:1]">
    %11 = cute.composition(%10, %tile) : (!cute.layout<"(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;64:1]">) -> !cute.layout<"(128,64):(1@0,1@1)">
    %non_exec_atom, %tma_tensor = cute_nvgpu.atom.make_non_exec_tiled_tma_load(%view, %slice, %11) <{kind = <sm_90> num_multicast = 1}> : (!memref_gmem_bf16_1, !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1)):((64,512),(1,0))">, !cute.layout<"(128,64):(1@0,1@1)">) -> (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">)
    %shape_33 = cute.make_shape(%itup_11, %itup_12, %itup_13) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?,?)">
    %12 = cute.make_identity_layout(%shape_33) : !cute.layout<"(?,?,?):(1@0,1@1,1@2)">
    %13 = cute.composition(%12, %tile) : (!cute.layout<"(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;64:1]">) -> !cute.layout<"(128,64):(1@0,1@1)">
    %non_exec_atom_34, %tma_tensor_35 = cute_nvgpu.atom.make_non_exec_tiled_tma_load(%view_3, %slice, %13) <{kind = <sm_90> num_multicast = 1}> : (!memref_gmem_bf16_1, !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1)):((64,512),(1,0))">, !cute.layout<"(128,64):(1@0,1@1)">) -> (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">)
    %slice_36 = cute.slice(%tile_to_shape_31, %coord) : !cute.composed_layout<"S<2,4,3> o 0 o ((8,16),(32,1),(1,4)):((32,256),(1,0),(0,4096))">, !cute.coord<"(_,_,0)">
    %shape_37 = cute.make_shape(%itup_17, %itup_18, %itup_19) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?,?)">
    %14 = cute.make_identity_layout(%shape_37) : !cute.layout<"(?,?,?):(1@0,1@1,1@2)">
    %tile_38 = cute.make_tile() : () -> !cute.tile<"[128:1;32:1]">
    %15 = cute.composition(%14, %tile_38) : (!cute.layout<"(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;32:1]">) -> !cute.layout<"(128,32):(1@0,1@1)">
    %16 = cute.composition(%14, %15) : (!cute.layout<"(?,?,?):(1@0,1@1,1@2)">, !cute.layout<"(128,32):(1@0,1@1)">) -> !cute.layout<"(128,32):(1@0,1@1)">
    %non_exec_atom_39, %tma_tensor_40 = cute_nvgpu.atom.make_non_exec_tiled_tma_store(%view_5, %slice_36, %16) : (!memref_gmem_bf16_1, !cute.composed_layout<"S<2,4,3> o 0 o ((8,16),(32,1)):((32,256),(1,0))">, !cute.layout<"(128,32):(1@0,1@1)">) -> (!cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 65536, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">)
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
    %17 = cute.get_scalars(%e0_53) : !cute.int_tuple<"?">
    %int_tuple_54 = cute.make_int_tuple(%e0_49) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %shp_55 = cute.ceil_div(%int_tuple_54, %tile_51) : !cute.int_tuple<"?">, !cute.tile<"1:0">
    %e0_56 = cute.get_leaves(%shp_55) : !cute.int_tuple<"?">
    %18 = cute.get_scalars(%e0_56) : !cute.int_tuple<"?">
    %int_tuple_57 = cute.make_int_tuple(%e0_53, %e0_56) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?,?)">
    %sz_58 = cute.size(%int_tuple_57) : (!cute.int_tuple<"(?,?)">) -> !cute.int_tuple<"?">
    %e0_59 = cute.get_leaves(%sz_58) : !cute.int_tuple<"?">
    %19 = cute.get_scalars(%e0_59) : !cute.int_tuple<"?">
    %int_tuple_60 = cute.make_int_tuple(%arg7) : (i32) -> !cute.int_tuple<"?">
    %add = cute.tuple_add(%e0_53, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %int_tuple_61 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
    %sub = cute.tuple_sub(%add, %int_tuple_61) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %div = cute.tuple_div(%sub, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %mul = cute.tuple_mul(%div, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %20 = cute.get_scalars(%mul) : !cute.int_tuple<"?">
    %add_62 = cute.tuple_add(%e0_56, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %sub_63 = cute.tuple_sub(%add_62, %int_tuple_61) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %div_64 = cute.tuple_div(%sub_63, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %mul_65 = cute.tuple_mul(%div_64, %int_tuple_60) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %21 = cute.get_scalars(%mul_65) : !cute.int_tuple<"?">
    %22 = arith.cmpi sle, %21, %20 : i32
    %23 = arith.extui %22 : i1 to i32
    %24 = arith.cmpi eq, %23, %c0_i32 : i32
    %25 = arith.select %24, %17, %18 : i32
    %26 = arith.select %24, %18, %17 : i32
    %27 = arith.minsi %arg7, %25 : i32
    %28 = arith.remsi %25, %27 : i32
    %29 = arith.floordivsi %25, %27 : i32
    %30 = arith.muli %27, %26 : i32
    %sub_66 = cute.tuple_sub(%e0_59, %int_tuple_61) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %31 = cute.get_scalars(%sub_66) : !cute.int_tuple<"?">
    %32 = "llvm.intr.ctlz"(%31) <{is_zero_poison = false}> : (i32) -> i32
    %33 = arith.subi %c31_i32, %32 : i32
    %34 = arith.maxsi %33, %c0_i32 : i32
    %35 = arith.shli %c1_i32, %34 : i32
    %36 = arith.extui %35 : i32 to i64
    %37 = arith.muli %36, %c4294967296_i64 : i64
    %38 = arith.extui %19 : i32 to i64
    %39 = arith.addi %37, %38 : i64
    %40 = arith.subi %39, %c1_i64 : i64
    %41 = arith.divui %40, %38 : i64
    %42 = arith.andi %41, %c4294967295_i64 : i64
    %43 = arith.trunci %42 : i64 to i32
    %44 = arith.subi %27, %c1_i32 : i32
    %45 = "llvm.intr.ctlz"(%44) <{is_zero_poison = false}> : (i32) -> i32
    %46 = arith.subi %c31_i32, %45 : i32
    %47 = arith.maxsi %46, %c0_i32 : i32
    %48 = arith.shli %c1_i32, %47 : i32
    %49 = arith.extui %48 : i32 to i64
    %50 = arith.muli %49, %c4294967296_i64 : i64
    %51 = arith.extui %27 : i32 to i64
    %52 = arith.addi %50, %51 : i64
    %53 = arith.subi %52, %c1_i64 : i64
    %54 = arith.divui %53, %51 : i64
    %55 = arith.andi %54, %c4294967295_i64 : i64
    %56 = arith.trunci %55 : i64 to i32
    %57 = arith.cmpi sgt, %28, %c0_i32 : i32
    %58 = arith.select %57, %28, %c1_i32 : i32
    %59 = arith.subi %58, %c1_i32 : i32
    %60 = "llvm.intr.ctlz"(%59) <{is_zero_poison = false}> : (i32) -> i32
    %61 = arith.subi %c31_i32, %60 : i32
    %62 = arith.maxsi %61, %c0_i32 : i32
    %63 = arith.shli %c1_i32, %62 : i32
    %64 = arith.extui %63 : i32 to i64
    %65 = arith.muli %64, %c4294967296_i64 : i64
    %66 = arith.extui %58 : i32 to i64
    %67 = arith.addi %65, %66 : i64
    %68 = arith.subi %67, %c1_i64 : i64
    %69 = arith.divui %68, %66 : i64
    %70 = arith.andi %69, %c4294967295_i64 : i64
    %71 = arith.trunci %70 : i64 to i32
    %72 = arith.subi %30, %c1_i32 : i32
    %73 = "llvm.intr.ctlz"(%72) <{is_zero_poison = false}> : (i32) -> i32
    %74 = arith.subi %c31_i32, %73 : i32
    %75 = arith.maxsi %74, %c0_i32 : i32
    %76 = arith.shli %c1_i32, %75 : i32
    %77 = arith.extui %76 : i32 to i64
    %78 = arith.muli %77, %c4294967296_i64 : i64
    %79 = arith.extui %30 : i32 to i64
    %80 = arith.addi %78, %79 : i64
    %81 = arith.subi %80, %c1_i64 : i64
    %82 = arith.divui %81, %79 : i64
    %83 = arith.andi %82, %c4294967295_i64 : i64
    %84 = arith.trunci %83 : i64 to i32
    %int_tuple_67 = cute.make_int_tuple(%e0_53, %e0_56, %itup_19) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?,?,?)">
    %sz_68 = cute.size(%int_tuple_67) : (!cute.int_tuple<"(?,?,?)">) -> !cute.int_tuple<"?">
    %e0_69 = cute.get_leaves(%sz_68) : !cute.int_tuple<"?">
    %mul_70 = cute.tuple_mul(%e0_69, %int_tuple_61) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %int_tuple_71 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
    %mul_72 = cute.tuple_mul(%mul_70, %int_tuple_71) : (!cute.int_tuple<"?">, !cute.int_tuple<"2">) -> !cute.int_tuple<"?{div=2}">
    %85 = cute.get_scalars(%mul_72) : !cute.int_tuple<"?{div=2}">
    %86 = arith.minsi %85, %arg6 : i32
    %87 = cute.kernel_smem_size @kernels::@kernel_cutlass_kernel_quackgemm_default_epiGemmDefaultSm90_object_at__TiledMMA_ThrLayoutVMNK128211112800_PermutationMNK____MMAAtom_ThrID1281_ShapeMNK6412816_TVLayoutA12864160164_TVLayoutB_0 : i64
    %88 = arith.cmpi sgt, %87, %c232448_i64 : i64
    scf.if %88 {
      cute.print("\0AError: kernel '@kernels::@kernel_cutlass_kernel_quackgemm_default_epiGemmDefaultSm90_object_at__TiledMMA_ThrLayoutVMNK128211112800_PermutationMNK____MMAAtom_ThrID1281_ShapeMNK6412816_TVLayoutA12864160164_TVLayoutB_0' launch shared memory exceeds current GPU arch sm_90a allowed. Allocated: {} bytes. Max: 232448 bytes.\0A\0A", %87) : i64
    }
    %89 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c384_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %87, gridDim = (%c1_i32, %c1_i32, %86), stream = %arg8) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17>
    cuda.launch_cfg.programmatic_stream_serialization_allowed[%89] %c1_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    cuda.launch_cfg.cluster_dim[%89] (%c1_i32, %c1_i32, %c1_i32) : !cuda.launch_cfg<max_attrs = 17>, i32, i32, i32
    cuda.launch_cfg.cooperative[%89] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    %90 = cuda.launch_ex @kernels::@kernel_cutlass_kernel_quackgemm_default_epiGemmDefaultSm90_object_at__TiledMMA_ThrLayoutVMNK128211112800_PermutationMNK____MMAAtom_ThrID1281_ShapeMNK6412816_TVLayoutA12864160164_TVLayoutB_0<%89> (%9, %non_exec_atom, %tma_tensor, %non_exec_atom_34, %tma_tensor_35, %non_exec_atom_39, %tma_tensor_40, %arg3, %arg4, %arg5, %lay_20, %tile_to_shape, %tile_to_shape, %tile_to_shape_31, %17, %18, %8, %23, %43, %34, %19, %29, %56, %47, %27, %71, %62, %58, %84, %75, %30) {assume_kernel_attr = #cuda.assume_kernel_attr<true>} : !cuda.launch_cfg<max_attrs = 17>, (!mma_bf16_bf16_f32_64x128x16, !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 131072, tma_gbasis = <"(64,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, !cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 65536, tma_gbasis = <"(32,128,1):(1@1,1@0,1@2)">, tma_format = BF16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@1,1@0,1@2)">, f32, !memref_gmem_i32, !memref_gmem_f32, !cute.layout<"(1,1,1):(0,0,0)">, !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">, !cute.composed_layout<"S<3,4,3> o 0 o ((8,16),(64,1),(1,6)):((64,512),(1,0),(0,8192))">, !cute.composed_layout<"S<2,4,3> o 0 o ((8,16),(32,1),(1,4)):((32,256),(1,0),(0,4096))">, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !cuda.result
    %91 = cuda.cast %90 : !cuda.result -> i32
    cuda.return_if_error %91 : i32
    return %c0_i32 : i32
  }
}

