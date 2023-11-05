# add at::cuda::getCurrentCUDAStream and at::cuda::getStreamFromPool
patch ./build/csrc/includes/context.h << 'DIFF___'
@@ -16,6 +16,23 @@
 #include <dpct/rng_utils.hpp>

 #include "gemm_test.h"
+#include <ipex.h>
+namespace at {
+  namespace cuda {
+    dpct::queue_ptr getCurrentCUDAStream() {
+      auto device_type = c10::DeviceType::XPU;
+      c10::impl::VirtualGuardImpl impl(device_type);
+      c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
+      auto& queue = xpu::get_queue_from_stream(c10_stream);
+      return &queue;
+    }
+
+    dpct::queue_ptr getStreamFromPool() {
+      // not implemented
+      return nullptr;
+    }
+  }
+}

 #define WARP_SIZE 32

DIFF___
