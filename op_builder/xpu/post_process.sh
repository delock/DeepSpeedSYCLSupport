
# fix cg::thread_block_tile<threadsPerHead> to auto
find ./deepspeed/third-party/ -type f -exec sed -Ei "s/cg::\S*/auto/g" {} +

# migrate thread_rank() to get_local_linear_id()
find ./deepspeed/third-party/ -type f -exec sed -i "s/thread_rank()/get_local_linear_id()/g" {} +

# migrate shfl to shuffle
find ./deepspeed/third-party/ -type f -exec sed -Ei "s/\.shfl/\.shuffle/g" {} +

# fix __half to sycl::half
find ./deepspeed/third-party/ -type f -exec sed -Ei "s/__half/sycl::half/g" {} +

# fix half2_raw to half2
find ./deepspeed/third-party/ -type f -exec sed -Ei "s/half2_raw/half2/g" {} +

# migrate meta_group_size to get_group_range().size()
find ./deepspeed/third-party/ -type f -exec sed -Ei "s/meta_group_size[(][)]/get_group_range().size()/g" {} +

# add #include <ipex.h>
find ./deepspeed/third-party/ -type f -exec sed -Ei "s:#include <c10/cuda/CUDAStream.h>:&\n#include <ipex.h>:g" {} +

# fix _free_memory_size is 0 error, give it 20G.
find ./deepspeed/third-party -type f -exec sed -i "s/if (\!_free_memory_size/_free_memory_size = 21474836480\;\n&/g" {} +

# change group_local_memory to group_local_memory_for_overwrite
find ./deepspeed/third-party -type f -exec sed -i "s/group_local_memory</group_local_memory_for_overwrite</g" {} +

# fix attn_softmax_v2 lacking of iterations
find ./deepspeed/third-party/ -type f -exec sed -i "s/attn_softmax_v2<T>/attn_softmax_v2<T, iterations>/g" {} +

# fix device at::kCUDA to at::kXPU
find ./deepspeed/third-party/ -type f -exec sed -i "s/at::kCUDA/at::kXPU/g" {} +

# fix pt_binding.cpp torch::from_blob 4 inputs pattern
patch ./deepspeed/third-party/csrc/transformer/inference/csrc/pt_binding.cpp << 'DIFF___'
@@ -549,32 +549,24 @@
     if (layer_id == num_layers - 1) InferenceContext::Instance().advance_tokens();
     auto prev_key = at::from_blob(workspace + offset,
                                   {bsz, heads, all_tokens, k},
-                                  c10::TensorType::contiguousStridesOf({bsz, heads, all_tokens, k}),
-                                  nullptr,
                                   {hidden_dim * InferenceContext::Instance().GetMaxTokenLength(),
                                    k * InferenceContext::Instance().GetMaxTokenLength(),
                                    k,
                                    1},
-                                  {hidden_dim * InferenceContext::Instance().GetMaxTokenLength(),
-                                   k * InferenceContext::Instance().GetMaxTokenLength(),
-                                   k,
-                                   1}
-                                      .device());
+                                   nullptr,
+                                   options,
+                                   options.device());

     auto prev_value =
         at::from_blob(workspace + offset + value_offset,
                       {bsz, heads, all_tokens, k},
-                      c10::TensorType::contiguousStridesOf({bsz, heads, all_tokens, k}),
-                      nullptr,
                       {hidden_dim * InferenceContext::Instance().GetMaxTokenLength(),
                        k * InferenceContext::Instance().GetMaxTokenLength(),
                        k,
                        1},
-                      {hidden_dim * InferenceContext::Instance().GetMaxTokenLength(),
-                       k * InferenceContext::Instance().GetMaxTokenLength(),
-                       k,
-                       1}
-                          .device());
+                       nullptr,
+                       options,
+                       options.device());

     return {output, prev_key, prev_value};
 }
DIFF___

# fix pt_binding.cpp at::from_blob device error
patch ./deepspeed/third-party/csrc/transformer/inference/csrc/pt_binding.cpp << 'DIFF___'
@@ -157,7 +157,7 @@
                            c10::TensorType::contiguousStridesOf({Q.size(1), Q.size(2), W.size(1)}),
                            nullptr,
                            options,
-                           options.device());
+                           Q.device());
     unsigned m = W.size(1);
     unsigned n = Q.size(1) * Q.size(2);
     unsigned k = Q.size(0);
@@ -479,7 +479,7 @@
                                 c10::TensorType::contiguousStridesOf({bsz, seq_len, hidden_dim}),
                                 nullptr,
                                 options,
-                                options.device());
+                                query_key_value.device());

     auto query_cont = workspace + 5 * buf_size;
     size_t offset =
@@ -555,7 +555,7 @@
                                    1},
                                    nullptr,
                                    options,
-                                   options.device());
+                                   query_key_value.device());

     auto prev_value =
         at::from_blob(workspace + offset + value_offset,
@@ -566,7 +566,7 @@
                        1},
                        nullptr,
                        options,
-                       options.device());
+                       query_key_value.device());

     return {output, prev_key, prev_value};
 }
@@ -991,14 +991,14 @@
                                   c10::TensorType::contiguousStridesOf(input.sizes()),
                                   nullptr,
                                   options,
-                                  options.device());
+                                  input.device());
     auto output = at::from_blob(
         workspace,
         {input.size(0), input.size(1), out_size},
         c10::TensorType::contiguousStridesOf({input.size(0), input.size(1), out_size}),
         nullptr,
         options,
-        options.device());
+        input.device());

     launch_rms_norm((T*)rms_norm.data_ptr(),
                     (T*)nullptr,
@@ -1067,14 +1067,14 @@
                        .layout(at::kStrided)
                        .device(at::kXPU)
                        .requires_grad(false);
-
+
     auto output = at::from_blob(
         workspace,
         {input.size(0), input.size(1), out_size},
         c10::TensorType::contiguousStridesOf({input.size(0), input.size(1), out_size}),
         nullptr,
         options,
-        options.device());
+        input.device());
     auto inp_norm = qkv_unfused_cublas<T>(output,
                                           input,
                                           weight,
@@ -1161,7 +1161,7 @@
         c10::TensorType::contiguousStridesOf({input.size(0), input.size(1), out_size}),
         nullptr,
         options,
-        options.device());
+        input.device());

     float alpha = (T)1.0;
     float gemm_beta = (T)0.0;
@@ -1231,7 +1231,7 @@
                     {3, input.size(0), num_heads, input.size(1), padded_head_size}),
                 nullptr,
                 options,
-                options.device());
+                input.device());
             // return at::from_blob(padded_output, {input.size(0) * input.size(1), 3, num_heads,
             // padded_head_size}, options);
         } else {
@@ -1261,7 +1261,7 @@
                                      {3, input.size(0), num_heads, input.size(1), head_size}),
                                  nullptr,
                                  options,
-                                 options.device());
+                                 input.device());
             // return at::from_blob(workspace, {input.size(0) * input.size(1), 3, num_heads,
             // head_size}, options);
         }
@@ -1412,7 +1412,7 @@
         c10::TensorType::contiguousStridesOf({input.size(0), input.size(1), out_size}),
         nullptr,
         options,
-        options.device());
+        input.device());
     if (q_int8) {
         quantized_gemm<T>(output.data_ptr(),
                           (T*)input.data_ptr(),
@@ -1616,7 +1616,7 @@
         c10::TensorType::contiguousStridesOf({input.size(0), input.size(1), out_size}),
         nullptr,
         options,
-        options.device());
+        input.device());
     int bsz = input.size(0) * input.size(1);

     auto act_func_type = static_cast<ActivationFuncType>(activation_type);
@@ -1675,20 +1675,20 @@
                                 c10::TensorType::contiguousStridesOf(input.sizes()),
                                 nullptr,
                                 options,
-                                options.device());
+                                input.device());
     auto inp_norm = at::from_blob(inp_norm_ptr,
                                   input.sizes(),
                                   c10::TensorType::contiguousStridesOf(input.sizes()),
                                   nullptr,
                                   options,
-                                  options.device());
+                                  input.device());
     auto intermediate_gemm = at::from_blob(
         intermediate_ptr,
         {input.size(0), input.size(1), mlp_1_out_neurons},
         c10::TensorType::contiguousStridesOf({input.size(0), input.size(1), mlp_1_out_neurons}),
         nullptr,
         options,
-        options.device());
+        input.device());

     auto act_func_type = static_cast<ActivationFuncType>(activation_type);

DIFF___
