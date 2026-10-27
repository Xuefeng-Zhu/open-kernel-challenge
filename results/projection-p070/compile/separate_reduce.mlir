!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 128 b>, layout_copy_tv = <"((16,8),4):((32,1),8)">, tiler_mn = <"[8:1;64:1]">>
!copy_simt1 = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<bf16, 64 b>, layout_copy_tv = <"((16,8),4):((32,1),8)">, tiler_mn = <"[8:1;64:1]">>
!memref_gmem_bf16 = !cute.memref<bf16, gmem, align<16>, "(?,?,?):(?{i64 div=8},1,?{i64 div=8})">
!memref_gmem_bf16_1 = !cute.memref<bf16, gmem, align<16>, "(?,?):(?{i64 div=8},1)">
!memref_gmem_bf16_2 = !cute.memref<bf16, gmem, align<16>, "(8,64):(?{i64 div=8},1)">
!memref_gmem_bf16_3 = !cute.memref<bf16, gmem, align<8>, "((4,1),1,1):((1,0),0,0)">
!memref_gmem_bf16_4 = !cute.memref<bf16, gmem, align<8>, "((4,1),(1,1)):((1,0),(0,0))">
!memref_gmem_f32 = !cute.memref<f32, gmem, align<16>, "(?,?,?):(?{i64 div=4},1,?{i64 div=4})">
!memref_gmem_f32_1 = !cute.memref<f32, gmem, align<16>, "(?,?):(?{i64 div=4},1)">
!memref_gmem_f32_2 = !cute.memref<f32, gmem, align<16>, "(8,64):(?{i64 div=4},1)">
!memref_gmem_f32_3 = !cute.memref<f32, gmem, align<16>, "((4,1),1,1):((1,0),0,0)">
!memref_gmem_f32_4 = !cute.memref<f32, gmem, align<16>, "((4,1),(1,1)):((1,0),(0,0))">
!memref_rmem_bf16 = !cute.memref<bf16, rmem, align<32>, "((4,1),1,1):((1,0),0,0)">
!memref_rmem_bf16_1 = !cute.memref<bf16, rmem, align<32>, "((4,1),(1,1)):((1,0),(0,0))">
!memref_rmem_f32 = !cute.memref<f32, rmem, align<32>, "((4,1),1,1):((1,0),0,0)">
!memref_rmem_f32_1 = !cute.memref<f32, rmem, align<32>, "((4,1),(1,1)):((1,0),(0,0))">
!memref_rmem_i8 = !cute.memref<i8, rmem, align<32>, "(1,1,1):(1,0,1)">
!memref_rmem_i8_1 = !cute.memref<i8, rmem, align<32>, "(1,(1,1)):(1,(0,1))">
module attributes {gpu.container_module} {
  gpu.module @kernels {
    cuda.kernel @kernel_cutlass_kernel_quacksplit_k_reduceSplitKReduce_object_at__tensorptrf32gmemalign16oi64div41i64div4_tensorptrbf16gmemalign16oi64div81i64div8_None_None_None_None__TiledCopy_TilerMN816_0(%arg0: !memref_gmem_f32, %arg1: !memref_gmem_bf16, %arg2: f32, %arg3: !copy_simt, %arg4: !copy_simt1) attributes {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, cute.kernel, cutlass_launch_facts = {cluster_launch = false, cooperative_launch = false, exact_block_dim = array<i64: 128, 1, 1>, schema_version = 1 : i64}, gpu.kernel, nvvm.reqntid = array<i32: 128, 1, 1>, smem.hint_smem_base_uniform, smem.partition_num = 2 : i32} {
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %lay = cute.get_layout(%arg1) : !memref_gmem_bf16
      %0 = nvvm.read.ptx.sreg.tid.x : i32
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = nvvm.read.ptx.sreg.ctaid.y : i32
      %3 = nvvm.read.ptx.sreg.ctaid.z : i32
      %4 = cute.get_shape(%lay) : (!cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">) -> !cute.shape<"(?,?,?)">
      %e0, %e1, %e2 = cute.get_leaves(%4) : !cute.shape<"(?,?,?)">
      %itup = cute.to_int_tuple(%e0) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %5 = cute.get_scalars(%itup) : !cute.int_tuple<"?">
      %itup_0 = cute.to_int_tuple(%e1) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %shape = cute.make_shape(%itup, %itup_0) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?)">
      %6 = cute.make_identity_tensor(%shape) : !cute.coord_tensor<"(0,0)", "(?,?):(1@0,1@1)">
      %tile = cute.make_tile() : () -> !cute.tile<"[8:1;64:1]">
      %coord = cute.make_coord(%1, %2) : (i32, i32) -> !cute.coord<"(?,?)">
      %tiled_view = cute.local_tile(%6, %tile, %coord) : (!cute.coord_tensor<"(0,0)", "(?,?):(1@0,1@1)">, !cute.tile<"[8:1;64:1]">, !cute.coord<"(?,?)">) -> !cute.coord_tensor<"(?{div=8},?{div=64})", "(8,64):(1@0,1@1)">
      %coord_1 = cute.make_coord(%0) : (i32) -> !cute.coord<"?">
      %src_partitioned = cute.tiled.copy.partition_S(%arg3, %tiled_view, %coord_1) : (!copy_simt, !cute.coord_tensor<"(?{div=8},?{div=64})", "(8,64):(1@0,1@1)">, !cute.coord<"?">) -> !cute.coord_tensor<"(?,?{div=4})", "((4,1),1,1):((1@1,0),0,0)">
      %coord_2 = cute.make_coord() : () -> !cute.coord<"((0,_),_,_)">
      %slice = cute.slice(%src_partitioned, %coord_2) : !cute.coord_tensor<"(?,?{div=4})", "((4,1),1,1):((1@1,0),0,0)">, !cute.coord<"((0,_),_,_)">
      %coord_3 = cute.make_coord() : () -> !cute.coord<"0">
      %slice_4 = cute.slice(%slice, %coord_3) : !cute.coord_tensor<"(?,?{div=4})", "(1,1,1):(0,0,0)">, !cute.coord<"0">
      %iter = cute.get_iter(%slice_4) : !cute.coord_tensor<"(?,?{div=4})", "():()">
      %tup = cute.deref_arith_tuple_iter(%iter) : !cute.arith_tuple_iter<"(?,?{div=4})">
      %e0_5, %e1_6 = cute.get_leaves(%tup) : !cute.int_tuple<"(?,?{div=4})">
      %7 = cute.get_scalars(%e0_5) : !cute.int_tuple<"?">
      %rmem = cute.memref.alloca() : !memref_rmem_i8
      %iter_7 = cute.get_iter(%rmem) : !memref_rmem_i8
      %coord_8 = cute.make_coord() : () -> !cute.coord<"((0,0),0,0)">
      %slice_9 = cute.slice(%src_partitioned, %coord_8) : !cute.coord_tensor<"(?,?{div=4})", "((4,1),1,1):((1@1,0),0,0)">, !cute.coord<"((0,0),0,0)">
      %iter_10 = cute.get_iter(%slice_9) : !cute.coord_tensor<"(?,?{div=4})", "():()">
      %tup_11 = cute.deref_arith_tuple_iter(%iter_10) : !cute.arith_tuple_iter<"(?,?{div=4})">
      %e0_12, %e1_13 = cute.get_leaves(%tup_11) : !cute.int_tuple<"(?,?{div=4})">
      %coord_14 = cute.make_coord(%e1_13) : (!cute.int_tuple<"?{div=4}">) -> !cute.coord<"?{div=4}">
      %coord_15 = cute.make_coord(%itup_0) : (!cute.int_tuple<"?">) -> !cute.coord<"?">
      %8 = cute.elem_less(%coord_14, %coord_15) : !cute.coord<"?{div=4}">, !cute.coord<"?">
      %9 = arith.extui %8 : i1 to i8
      %coord_16 = cute.make_coord() : () -> !cute.coord<"(0,0,0)">
      cute.memref.store(%rmem, %coord_16, %9) : (!memref_rmem_i8, !cute.coord<"(0,0,0)">, i8) -> ()
      %10 = arith.cmpi slt, %7, %5 : i32
      %coord_17 = cute.make_coord(%3) : (i32) -> !cute.coord<"(_,_,?)">
      %slice_18 = cute.slice(%arg1, %coord_17) : !memref_gmem_bf16, !cute.coord<"(_,_,?)">
      %tiled_view_19 = cute.local_tile(%slice_18, %tile, %coord) : (!memref_gmem_bf16_1, !cute.tile<"[8:1;64:1]">, !cute.coord<"(?,?)">) -> !memref_gmem_bf16_2
      %src_partitioned_20 = cute.tiled.copy.partition_S(%arg4, %tiled_view_19, %coord_1) : (!copy_simt1, !memref_gmem_bf16_2, !cute.coord<"?">) -> !memref_gmem_bf16_3
      %iter_21 = cute.get_iter(%src_partitioned_20) : !memref_gmem_bf16_3
      %11 = arith.muli %3, %c2_i32 : i32
      %coord_22 = cute.make_coord(%11) : (i32) -> !cute.coord<"(_,_,?)">
      %slice_23 = cute.slice(%arg0, %coord_22) : !memref_gmem_f32, !cute.coord<"(_,_,?)">
      %tiled_view_24 = cute.local_tile(%slice_23, %tile, %coord) : (!memref_gmem_f32_1, !cute.tile<"[8:1;64:1]">, !cute.coord<"(?,?)">) -> !memref_gmem_f32_2
      %src_partitioned_25 = cute.tiled.copy.partition_S(%arg3, %tiled_view_24, %coord_1) : (!copy_simt, !memref_gmem_f32_2, !cute.coord<"?">) -> !memref_gmem_f32_3
      %iter_26 = cute.get_iter(%src_partitioned_25) : !memref_gmem_f32_3
      %12 = arith.addi %11, %c1_i32 : i32
      %coord_27 = cute.make_coord(%12) : (i32) -> !cute.coord<"(_,_,?)">
      %slice_28 = cute.slice(%arg0, %coord_27) : !memref_gmem_f32, !cute.coord<"(_,_,?)">
      %tiled_view_29 = cute.local_tile(%slice_28, %tile, %coord) : (!memref_gmem_f32_1, !cute.tile<"[8:1;64:1]">, !cute.coord<"(?,?)">) -> !memref_gmem_f32_2
      %src_partitioned_30 = cute.tiled.copy.partition_S(%arg3, %tiled_view_29, %coord_1) : (!copy_simt, !memref_gmem_f32_2, !cute.coord<"?">) -> !memref_gmem_f32_3
      %iter_31 = cute.get_iter(%src_partitioned_30) : !memref_gmem_f32_3
      %rmem_32 = cute.memref.alloca() : !memref_rmem_f32
      %iter_33 = cute.get_iter(%rmem_32) : !memref_rmem_f32
      %rmem_34 = cute.memref.alloca() : !memref_rmem_f32
      %iter_35 = cute.get_iter(%rmem_34) : !memref_rmem_f32
      scf.if %10 {
        %lay_39 = cute.get_layout(%src_partitioned_25) : !memref_gmem_f32_3
        %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
        %lay_40 = cute.get_layout(%rmem_32) : !memref_rmem_f32
        %shape_41 = cute.make_shape() : () -> !cute.shape<"1">
        %lay_42 = cute.make_layout(%shape_41) : !cute.layout<"1:0">
        %append = cute.append_to_rank<2> (%lay_39, %lay_42) : !cute.layout<"((4,1),1,1):((1,0),0,0)">, !cute.layout<"1:0">
        %view = cute.make_view(%iter_26, %append) : !memref_gmem_f32_3
        %grouped = cute.group_modes(%view) <1, 3> : (!memref_gmem_f32_3) -> !memref_gmem_f32_4
        %append_43 = cute.append_to_rank<2> (%lay_40, %lay_42) : !cute.layout<"((4,1),1,1):((1,0),0,0)">, !cute.layout<"1:0">
        %view_44 = cute.make_view(%iter_33, %append_43) : !memref_rmem_f32
        %grouped_45 = cute.group_modes(%view_44) <1, 3> : (!memref_rmem_f32) -> !memref_rmem_f32_1
        %lay_46 = cute.get_layout(%rmem) : !memref_rmem_i8
        %append_47 = cute.append_to_rank<2> (%lay_46, %lay_42) : !cute.layout<"(1,1,1):(1,0,1)">, !cute.layout<"1:0">
        %view_48 = cute.make_view(%iter_7, %append_47) : !memref_rmem_i8
        %grouped_49 = cute.group_modes(%view_48) <1, 3> : (!memref_rmem_i8) -> !memref_rmem_i8_1
        cute.copy(%atom, %grouped, %grouped_45, %grouped_49) : (!cute_nvgpu.atom.universal_copy<f32, 128 b>, !memref_gmem_f32_4, !memref_rmem_f32_1, !memref_rmem_i8_1)
        %lay_50 = cute.get_layout(%src_partitioned_30) : !memref_gmem_f32_3
        %lay_51 = cute.get_layout(%rmem_34) : !memref_rmem_f32
        %append_52 = cute.append_to_rank<2> (%lay_50, %lay_42) : !cute.layout<"((4,1),1,1):((1,0),0,0)">, !cute.layout<"1:0">
        %view_53 = cute.make_view(%iter_31, %append_52) : !memref_gmem_f32_3
        %grouped_54 = cute.group_modes(%view_53) <1, 3> : (!memref_gmem_f32_3) -> !memref_gmem_f32_4
        %append_55 = cute.append_to_rank<2> (%lay_51, %lay_42) : !cute.layout<"((4,1),1,1):((1,0),0,0)">, !cute.layout<"1:0">
        %view_56 = cute.make_view(%iter_35, %append_55) : !memref_rmem_f32
        %grouped_57 = cute.group_modes(%view_56) <1, 3> : (!memref_rmem_f32) -> !memref_rmem_f32_1
        cute.copy(%atom, %grouped_54, %grouped_57, %grouped_49) : (!cute_nvgpu.atom.universal_copy<f32, 128 b>, !memref_gmem_f32_4, !memref_rmem_f32_1, !memref_rmem_i8_1)
      }
      %13 = cute.memref.load_vec(%rmem_32) : (!memref_rmem_f32) -> vector<4xf32>
      %14 = cute.memref.load_vec(%rmem_34) : (!memref_rmem_f32) -> vector<4xf32>
      %15 = arith.addf %13, %14 : vector<4xf32>
      %rmem_36 = cute.memref.alloca() : !memref_rmem_f32
      cute.memref.store_vec(%15, %rmem_36) : (vector<4xf32>, !memref_rmem_f32) -> ()
      %16 = cute.memref.load_vec(%rmem_36) : (!memref_rmem_f32) -> vector<4xf32>
      %rmem_37 = cute.memref.alloca() : !memref_rmem_bf16
      %iter_38 = cute.get_iter(%rmem_37) : !memref_rmem_bf16
      %17 = arith.truncf %16 : vector<4xf32> to vector<4xbf16>
      cute.memref.store_vec(%17, %rmem_37) : (vector<4xbf16>, !memref_rmem_bf16) -> ()
      scf.if %10 {
        %lay_39 = cute.get_layout(%rmem_37) : !memref_rmem_bf16
        %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<bf16, 64 b>
        %lay_40 = cute.get_layout(%src_partitioned_20) : !memref_gmem_bf16_3
        %shape_41 = cute.make_shape() : () -> !cute.shape<"1">
        %lay_42 = cute.make_layout(%shape_41) : !cute.layout<"1:0">
        %append = cute.append_to_rank<2> (%lay_39, %lay_42) : !cute.layout<"((4,1),1,1):((1,0),0,0)">, !cute.layout<"1:0">
        %view = cute.make_view(%iter_38, %append) : !memref_rmem_bf16
        %grouped = cute.group_modes(%view) <1, 3> : (!memref_rmem_bf16) -> !memref_rmem_bf16_1
        %append_43 = cute.append_to_rank<2> (%lay_40, %lay_42) : !cute.layout<"((4,1),1,1):((1,0),0,0)">, !cute.layout<"1:0">
        %view_44 = cute.make_view(%iter_21, %append_43) : !memref_gmem_bf16_3
        %grouped_45 = cute.group_modes(%view_44) <1, 3> : (!memref_gmem_bf16_3) -> !memref_gmem_bf16_4
        %lay_46 = cute.get_layout(%rmem) : !memref_rmem_i8
        %append_47 = cute.append_to_rank<2> (%lay_46, %lay_42) : !cute.layout<"(1,1,1):(1,0,1)">, !cute.layout<"1:0">
        %view_48 = cute.make_view(%iter_7, %append_47) : !memref_rmem_i8
        %grouped_49 = cute.group_modes(%view_48) <1, 3> : (!memref_rmem_i8) -> !memref_rmem_i8_1
        cute.copy(%atom, %grouped, %grouped_45, %grouped_49) : (!cute_nvgpu.atom.universal_copy<bf16, 64 b>, !memref_rmem_bf16_1, !memref_gmem_bf16_4, !memref_rmem_i8_1)
      }
      return
    }
  }
  func.func @cutlass___call___quacksplit_k_reduceSplitKReduce_object_at__FakeTensorFloat32div1div1div1i64div41i64div4_FakeTensorBFloat16div1div1div1i64div81i64div8_None_None_None_None__FakeStre(%arg0: !memref_gmem_f32, %arg1: !memref_gmem_bf16, %arg2: f32, %arg3: !cuda.stream) -> i32 attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c232448_i64 = arith.constant 232448 : i64
    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 128 b>
    %0 = cute.make_tiled_copy(%atom) : !copy_simt
    %atom_0 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<bf16, 64 b>
    %1 = cute.make_tiled_copy(%atom_0) : !copy_simt1
    %lay = cute.get_layout(%arg1) : !memref_gmem_bf16
    %2 = cute.get_shape(%lay) : (!cute.layout<"(?,?,?):(?{i64 div=8},1,?{i64 div=8})">) -> !cute.shape<"(?,?,?)">
    %e0, %e1, %e2 = cute.get_leaves(%2) : !cute.shape<"(?,?,?)">
    %itup = cute.to_int_tuple(%e0) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_1 = cute.to_int_tuple(%e1) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_2 = cute.to_int_tuple(%e2) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %3 = cute.get_scalars(%itup_2) : !cute.int_tuple<"?">
    %int_tuple = cute.make_int_tuple(%itup) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %tile = cute.make_tile() : () -> !cute.tile<"8:1">
    %shp = cute.ceil_div(%int_tuple, %tile) : !cute.int_tuple<"?">, !cute.tile<"8:1">
    %e0_3 = cute.get_leaves(%shp) : !cute.int_tuple<"?">
    %4 = cute.get_scalars(%e0_3) : !cute.int_tuple<"?">
    %int_tuple_4 = cute.make_int_tuple(%itup_1) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %tile_5 = cute.make_tile() : () -> !cute.tile<"64:1">
    %shp_6 = cute.ceil_div(%int_tuple_4, %tile_5) : !cute.int_tuple<"?">, !cute.tile<"64:1">
    %e0_7 = cute.get_leaves(%shp_6) : !cute.int_tuple<"?">
    %5 = cute.get_scalars(%e0_7) : !cute.int_tuple<"?">
    %6 = cute.kernel_smem_size @kernels::@kernel_cutlass_kernel_quacksplit_k_reduceSplitKReduce_object_at__tensorptrf32gmemalign16oi64div41i64div4_tensorptrbf16gmemalign16oi64div81i64div8_None_None_None_None__TiledCopy_TilerMN816_0 : i64
    %7 = arith.cmpi sgt, %6, %c232448_i64 : i64
    scf.if %7 {
      cute.print("\0AError: kernel '@kernels::@kernel_cutlass_kernel_quacksplit_k_reduceSplitKReduce_object_at__tensorptrf32gmemalign16oi64div41i64div4_tensorptrbf16gmemalign16oi64div81i64div8_None_None_None_None__TiledCopy_TilerMN816_0' launch shared memory exceeds current GPU arch sm_90a allowed. Allocated: {} bytes. Max: 232448 bytes.\0A\0A", %6) : i64
    }
    %8 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c128_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %6, gridDim = (%4, %5, %3), stream = %arg3) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17>
    cuda.launch_cfg.programmatic_stream_serialization_allowed[%8] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    cuda.launch_cfg.cooperative[%8] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    %9 = cuda.launch_ex @kernels::@kernel_cutlass_kernel_quacksplit_k_reduceSplitKReduce_object_at__tensorptrf32gmemalign16oi64div41i64div4_tensorptrbf16gmemalign16oi64div81i64div8_None_None_None_None__TiledCopy_TilerMN816_0<%8> (%arg0, %arg1, %arg2, %0, %1) {assume_kernel_attr = #cuda.assume_kernel_attr<true>} : !cuda.launch_cfg<max_attrs = 17>, (!memref_gmem_f32, !memref_gmem_bf16, f32, !copy_simt, !copy_simt1) -> !cuda.result
    %10 = cuda.cast %9 : !cuda.result -> i32
    cuda.return_if_error %10 : i32
    return %c0_i32 : i32
  }
}

