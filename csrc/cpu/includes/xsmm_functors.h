/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifndef _XSMM_FUNCTORS_H_
#define _XSMM_FUNCTORS_H_

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include "libxsmm.h"
#include "libxsmm_intrinsics_x86.h"
#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#else
#include <pytorch_extension_wrapper.h>
#endif
//#include <bfloat8.h>
#include <string>
#include <unordered_map>

#define TPP_ASSERT(cond, x...) \
  do {                         \
    if (!(cond)) {             \
      printf(x);               \
      fflush(stdout);          \
      exit(1);                 \
    }                          \
  } while (0)
#define DECL_VLA_PTR(type, name, dims, ptr) type(*name) dims = (type(*) dims)ptr
#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
//extern long long hsh_key, hsh_ret;
namespace tpp {
typedef at::BFloat16 bfloat16;
typedef at::Half half;
//typedef at::BFloat8 bfloat8;
inline float upconvert_to_float(float val) {
  return val;
}
inline float upconvert_to_float(bfloat16 val) {
  return (float)val;
}
inline float upconvert_to_float(half val) {
  return (float)val;
}
template <typename T>
inline libxsmm_datatype XsmmDtype();
template <>
inline libxsmm_datatype XsmmDtype<int64_t>() {
  return LIBXSMM_DATATYPE_I64;
}
template <>
inline libxsmm_datatype XsmmDtype<int32_t>() {
  return LIBXSMM_DATATYPE_I32;
}
template <>
inline libxsmm_datatype XsmmDtype<float>() {
  return LIBXSMM_DATATYPE_F32;
}
template <>
inline libxsmm_datatype XsmmDtype<bfloat16>() {
  return LIBXSMM_DATATYPE_BF16;
}
template <>
inline libxsmm_datatype XsmmDtype<half>() {
  return LIBXSMM_DATATYPE_F16;
}

#ifdef __AVX512F__
inline __m512 _mm512_loadu_ps_auto(float const* mem_addr) {
  return _mm512_loadu_ps(mem_addr);
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, float const* mem_addr) {
  return _mm512_maskz_loadu_ps(k, mem_addr);
}
inline void _mm512_storeu_ps_auto(float* mem_addr, __m512 a) {
  _mm512_storeu_ps(mem_addr, a);
}
inline void _mm512_mask_storeu_ps_auto(float* mem_addr, __mmask16 k, __m512 a) {
  _mm512_mask_storeu_ps(mem_addr, k, a);
}

inline __m512 _mm512_loadu_ps_auto(half const* mem_addr) {
  return _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, half const* mem_addr) {
  return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));
}
inline void _mm512_storeu_ps_auto(half* mem_addr, __m512 a) {
  _mm256_storeu_si256(
      (__m256i*)mem_addr,
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
inline void _mm512_mask_storeu_ps_auto(half* mem_addr, __mmask16 k, __m512 a) {
  _mm256_mask_storeu_epi16(
      (__m256i*)mem_addr,
      k,
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline __m512 _mm512_convert_bf_ps(__m256i a) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(a), 16));
}
inline __m256i _mm256_convert_ps_bf(__m512 a) {
  return _mm512_cvtepi32_epi16(
      _mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a), 16));
}

inline __m512 _mm512_loadu_ps_auto(bfloat16 const* mem_addr) {
  return _mm512_convert_bf_ps(_mm256_loadu_si256((__m256i*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(
    __mmask16 k,
    bfloat16 const* mem_addr) {
  return _mm512_convert_bf_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));
}
inline void _mm512_storeu_ps_auto(bfloat16* mem_addr, __m512 a) {
  _mm256_storeu_si256((__m256i*)mem_addr, _mm256_convert_ps_bf(a));
}
inline void _mm512_mask_storeu_ps_auto(
    bfloat16* mem_addr,
    __mmask16 k,
    __m512 a) {
  _mm256_mask_storeu_epi16((__m256i*)mem_addr, k, _mm256_convert_ps_bf(a));
}

inline __m512 _mm512_split_loadu_ps(bfloat16 const* hi, bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline __m512 _mm512_maskz_split_loadu_ps(
    __mmask16 k,
    bfloat16 const* hi,
    bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline void _mm512_split_storeu_ps(bfloat16* hi, bfloat16* lo, __m512 a) {
  //_mm512_storeu_ps_auto(hi, a);
  _mm256_storeu_si256(
      (__m256i*)hi,
      _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_storeu_si256(
      (__m256i*)lo, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline void _mm512_mask_split_storeu_ps(
    bfloat16* hi,
    bfloat16* lo,
    __mmask16 k,
    __m512 a) {
  //_mm512_mask_storeu_ps_auto(hi, k, a);
  _mm256_mask_storeu_epi16(
      (__m256i*)hi,
      k,
      _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_mask_storeu_epi16(
      (__m256i*)lo, k, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline __m512 _mm512_convert_bf8_ps(__m128i a) {
  return _mm512_cvtph_ps(_mm256_slli_epi16(_mm256_cvtepi8_epi16(a), 8));
}
inline __m128i _mm_convert_ps_bf8(__m512 a) {
  return _mm256_cvtepi16_epi8(_mm256_srai_epi16(
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), 8));
}

#endif

inline libxsmm_datatype convert_dtype_pt2xsmm(at::ScalarType dtype) {
  static const std::map<at::ScalarType, libxsmm_datatype> pt2xsmmDtypes = {
      {at::kDouble, LIBXSMM_DATATYPE_F64},
      {at::kFloat, LIBXSMM_DATATYPE_F32},
      {at::kHalf, LIBXSMM_DATATYPE_F16},
      {at::kBFloat16, LIBXSMM_DATATYPE_BF16},
      {at::kByte, LIBXSMM_DATATYPE_I8},
      {at::kChar, LIBXSMM_DATATYPE_I8},
      {at::kShort, LIBXSMM_DATATYPE_I16},
      {at::kInt, LIBXSMM_DATATYPE_I32},
      {at::kLong, LIBXSMM_DATATYPE_I64}};

  return pt2xsmmDtypes.at(dtype);
}

inline int xsmm_get_vnni_block_size(libxsmm_datatype dtype) {
  int bs = libxsmm_cpuid_dot_pack_factor(dtype);
  if (bs <= 0) {
    throw std::invalid_argument("Unsupported datatype");
  }
  return bs;
}

inline int get_vnni_block_size(at::ScalarType dtype) {
  auto xsmm_dtype = convert_dtype_pt2xsmm(dtype);
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

inline int get_vnni_block_size(caffe2::TypeMeta dtype_) {
  at::ScalarType dtype = dtype_.toScalarType();
  auto xsmm_dtype = convert_dtype_pt2xsmm(dtype);
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

template <typename T>
inline int get_vnni_block_size() {
  auto xsmm_dtype = XsmmDtype<T>();
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

inline void debug_print_eqn_tree(libxsmm_blasint eqn_no) {
  if (false) {
    //libxsmm_matrix_eqn_tree_print(eqn_no);
    //libxsmm_matrix_eqn_rpn_print(eqn_no);
  }
}

class BaseTPP {
 public:
  void* get_kernel() {
    auto t0 = __rdtsc();
    auto& kernel_cache = get_kernel_cache();
    void* kernel = NULL;
    if (hash == "")
      hash = hash_str();
    auto t1 = __rdtsc();
    auto search = kernel_cache.find(hash);
    if (search != kernel_cache.end())
      kernel = search->second;
    if (kernel == NULL) {
      kernel = build_kernel();
      if (kernel == NULL) {
        //fprintf(stderr, "Unable to get JIT kernel for %s\n", hash.c_str());
        //exit(1);
      }
      // printf("TPP: %s @ %p\n", hash.c_str(), kernel);
      kernel_cache[hash] = kernel;
      // printf("Hash size = %ld\n", (long)kernel_cache.size());
    }
    auto t2 = __rdtsc();
    //hsh_key += t1 - t0;
    //hsh_ret += t2 - t1;
    // printf("%6lld  %6lld %6lld  get_kernel[%s]\n", t2-t0, (t1-t0), (t2-t1),
    // hash.c_str());
    return kernel;
  }
  // We should make hash_str() public
  std::string get_hash_str() {
    return hash_str();
  }

 protected:
#if 0
  std::unordered_map<std::string, void*>& get_kernel_cache() {
    static std::unordered_map<std::string, void*> kernel_cache;
    return kernel_cache;
  }
#else
  ska::flat_hash_map<std::string, void*>& get_kernel_cache() {
    static ska::flat_hash_map<std::string, void*> kernel_cache;
    return kernel_cache;
  }
#endif
  virtual std::string hash_str() = 0;
  virtual void* build_kernel() = 0;
  std::string hash = "";
  bool initialized = false;
};

class UnaryTPP : public BaseTPP {
 public:
  UnaryTPP() {}
  UnaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_unary_type type)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        dt_in(dt_in),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_unary)get_kernel();
    if (kernel)
      initialized = true;
  }

  void operator()(void* in, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.out.primary = out;
    kernel(&unary_param);
  }
  void operator()(void* in, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }
  void operator()(void* in, void* in2, void* in3, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }

  void operator()(
      void* in,
      void* in2,
      void* in3,
      void* op,
      void* op2,
      void* op3,
      void* out,
      void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.op.primary = op;
    unary_param.op.secondary = op2;
    unary_param.op.tertiary = op3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "unary_r%d_c%d_i%d_o%d_di%d_do%d_dc%d_f%d_t%d",
        rows,
        cols,
        ldi,
        ldo,
        dt_in,
        dt_out,
        dt_compute,
        flags,
        type);
    return std::string(hash);
  }
  void* build_kernel() override {
    //libxsmm_meltw_unary_shape shape = libxsmm_create_meltw_unary_shape(
        //cols, rows, ldi, ldo, dt_in, dt_out, dt_compute);
    //return (void*)libxsmm_dispatch_meltw_unary(type, shape, flags);
    return NULL;
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi = 0;
  libxsmm_blasint ldo = 0;
  libxsmm_datatype dt_in = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype dt_out = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype dt_compute = LIBXSMM_DATATYPE_F32;
  libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  libxsmm_meltwfunction_unary kernel = NULL;
};
class BinaryTPP : public BaseTPP {
 public:
  BinaryTPP() {}
  BinaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_binary_type type)
      : BinaryTPP(
            rows,
            cols,
            ldi,
            ldi,
            ldo,
            dt_in,
            dt_in,
            dt_out,
            dt_compute,
            flags,
            type) {}
  BinaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi0,
      libxsmm_blasint ldi1,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in0,
      libxsmm_datatype dt_in1,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_binary_type type)
      : rows(rows),
        cols(cols),
        ldi0(ldi0),
        ldi1(ldi1),
        ldo(ldo),
        dt_in0(dt_in0),
        dt_in1(dt_in1),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_binary)get_kernel();
    if (kernel)
      initialized = true;
  }

  void operator()(void* in0, void* in1, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_binary_param binary_param;
    binary_param.in0.primary = in0;
    binary_param.in1.primary = in1;
    binary_param.out.primary = out;
    kernel(&binary_param);
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "binary_r%d_c%d_i0%d_i1%d_o%d_di0%d_di1%d_do%d_dc%d_f%d_t%d",
        rows,
        cols,
        ldi0,
        ldi1,
        ldo,
        dt_in0,
        dt_in1,
        dt_out,
        dt_compute,
        flags,
        type);
    return std::string(hash);
  }
  void* build_kernel() override {
    //libxsmm_meltw_binary_shape shape = libxsmm_create_meltw_binary_shape(
        //cols, rows, ldi0, ldi1, ldo, dt_in0, dt_in1, dt_out, dt_compute);
    //return (void*)libxsmm_dispatch_meltw_binary(type, shape, flags);
    return NULL;
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi0;
  libxsmm_blasint ldi1;
  libxsmm_blasint ldo;
  libxsmm_datatype dt_in0;
  libxsmm_datatype dt_in1;
  libxsmm_datatype dt_out;
  libxsmm_datatype dt_compute;
  libxsmm_bitfield flags;
  libxsmm_meltw_binary_type type;
  libxsmm_meltwfunction_binary kernel = NULL;
};

template <typename Tin, typename Tout>
class ConvertTPP {
 public:
  ConvertTPP() {}
  ConvertTPP(int N) : ConvertTPP(1, N) {}
  ConvertTPP(int rows, int cols) : ConvertTPP(rows, cols, cols, cols) {}
  ConvertTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY),
        init_done(true) {}
  void operator()(Tin* in, Tout* out) {
    if (!(XsmmDtype<Tin>() == LIBXSMM_DATATYPE_F32 &&
          XsmmDtype<Tout>() == LIBXSMM_DATATYPE_F32) ||
        ((void*)in != (void*)out))
      kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[i * ldi + j];
      }
    }
  }
  bool initialized() {
    return init_done;
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
  bool init_done = false;
};

template <typename T>
class CpyTPP {
 public:
  CpyTPP() {}
  CpyTPP(int N) : CpyTPP(1, N) {}
  CpyTPP(int rows, int cols) : CpyTPP(rows, cols, cols, cols) {}
  CpyTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = in[i * ldi + j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class CpyBiasTPP {
 public:
  CpyBiasTPP() {}
  CpyBiasTPP(int rows, int cols) : CpyBiasTPP(rows, cols, cols) {}
  CpyBiasTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            cols,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class CpyBcastTPP {
 public:
  CpyBcastTPP() {}
  CpyBcastTPP(int rows, int cols) : CpyBcastTPP(rows, cols, cols) {}
  CpyBcastTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[i];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};
template <typename T>
class AddBiasTPP {
 public:
  AddBiasTPP() {}
  AddBiasTPP(int rows, int cols) : AddBiasTPP(rows, cols, cols) {}
  AddBiasTPP(int rows, int cols, int ld)
      : rows(rows),
        cols(cols),
        ld(ld),
        kernel(
            rows,
            cols,
            cols,
            ld,
            ld,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_ADD) {}
  void operator()(T* in, float* out) {
    kernel((void*)in, (void*)out, (void*)out);
  }
  void ref(T* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ld + c] += (float)in[c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ld;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout = Tin, typename Tin2 = Tin>
class AddTPP {
 public:
  AddTPP() {}
  AddTPP(int N) : AddTPP(1, N) {}
  AddTPP(int rows, int cols) : AddTPP(rows, cols, cols, cols) {}
  AddTPP(int rows, int cols, int ldi, int ldo)
      : AddTPP(rows, cols, ldi, ldi, ldo) {}
  AddTPP(int rows, int cols, int ldi0, int ldi1, int ldo)
      : rows(rows),
        cols(cols),
        ldi0(ldi0),
        ldi1(ldi1),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi0,
            ldi1,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin2>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_ADD) {}
  void operator()(Tin* in0, Tin2* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin2* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi0 + c] + (float)in1[r * ldi1 + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi0;
  int ldi1;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class MulTPP {
 public:
  MulTPP() {}
  MulTPP(int N) : MulTPP(1, N) {}
  MulTPP(int rows, int cols) : MulTPP(rows, cols, cols, cols) {}
  MulTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] * (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin>
class GradBiasTPP {
 public:
  GradBiasTPP() {}
  GradBiasTPP(int rows, int cols) : GradBiasTPP(rows, cols, cols) {}
  GradBiasTPP(int rows, int cols, int ldi)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        reduce(
            rows,
            cols,
            ldi,
            cols,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        add(cols) {}
  void operator()(Tin* in, float* out) {
    float tmp[cols];
    reduce((void*)in, (void*)tmp);
    add(tmp, out, out);
  }
  void ref(Tin* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;

  UnaryTPP reduce;
  AddTPP<float, float> add;
};

// ############################# Mul & Reduction TPP
// #####################################

template <typename Tin, typename Tout = Tin>
class ReduceAddColTPP {
 public:
  ReduceAddColTPP() {}
  ReduceAddColTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        reduce(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) {}
  void operator()(Tin* in, float* out) {
    reduce(in, out);
  }
  void ref(Tin* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        if (r == 0)
          out[c] = 0;
        out[c] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi, ldo;

  UnaryTPP reduce;
};

template <typename Tin, typename Tout = Tin>
class ReduceAddRowTPP {
 public:
  ReduceAddRowTPP() {}
  ReduceAddRowTPP(int rows, int cols, bool acc)
      : ReduceAddRowTPP(rows, cols, cols, acc) {}
  ReduceAddRowTPP(int rows, int cols, int ldi, bool acc)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        acc(acc),
        reduce(
            rows,
            cols,
            ldi,
            cols,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        add(rows) {}
  void operator()(Tin* in, Tout* out) {
    if (acc) {
      Tout tmp[rows];
      reduce((void*)in, (void*)tmp);
      add(tmp, out, out);
    } else {
      reduce((void*)in, (void*)out);
    }
  }
  void ref(Tin* in, Tout* out) {
    for (int r = 0; r < rows; r++) {
      if (!acc) {
        out[r] = 0;
      }
      for (int c = 0; c < cols; c++) {
        out[r] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  bool acc;
  UnaryTPP reduce;
  AddTPP<Tout, Tout> add;
};

// ############################# Broadcast & Multiplication TPP
// #####################################
template <typename Tin, typename Tout = Tin>
class BCastMulTPP {
 public:
  BCastMulTPP() {}
  BCastMulTPP(int rows, int cols) : BCastMulTPP(rows, cols, cols, cols) {}
  BCastMulTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0, // Broadcast in Row
                                                      // Dimension
            LIBXSMM_MELTW_TYPE_BINARY_MUL) // Multiplication
  {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c * ldo + r] = (Tin)in0[r] * in1[c * ldi + r];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

// ############################# Broadcast & Multiplication Addition TPP
// #####################################
template <typename Tin, typename Tout = Tin>
class BCastMulAddTPP {
 public:
  BCastMulAddTPP() {}
  BCastMulAddTPP(int rows, int cols) : BCastMulAddTPP(rows, cols, cols, cols) {}
  BCastMulAddTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0, // Broadcast in Row
                                                      // Dimension
            LIBXSMM_MELTW_TYPE_BINARY_MULADD) // Multiplication
  {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }

  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c * ldo + r] += (Tin)in0[r] * (Tin)in1[c * ldi + r];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout>
class ScaleTPP {
 public:
  ScaleTPP() {}
  ScaleTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in, Tout* out, float scale) {
    Tin alpha = scale;
    kernel((void*)&alpha, (void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out, float scale) {
    Tin alpha = scale;
    for (int i = 0; i < N; i++) {
      out[i] = (float)in[i] * (float)alpha;
    }
  }

 private:
  int N = 0;
  BinaryTPP kernel;
};

template <typename T, typename TN = float>
class Norm2TPP {
 public:
  Norm2TPP() {}
  Norm2TPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) {}
  void operator()(T* in, TN* sum) {
    float lsum = 0.0f;
    kernel((void*)in, (void*)&lsum);
    *sum += (TN)lsum;
  }
  void ref(T* in, TN* sum) {
    float lsum = 0.0f;
    for (int i = 0; i < N; i++) {
      lsum += (float)in[i] * (float)in[i];
    }
    *sum += (TN)lsum;
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename T>
class RecpTPP {
 public:
  RecpTPP() {}
  RecpTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < N; i++)
      out[i] = 1.0 / in[i];
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename T>
class RecpSqrtTPP {
 public:
  RecpSqrtTPP() {}
  RecpSqrtTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < N; i++)
      out[i] = 1.0 / sqrt(in[i]);
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class MulNormTPP {
 public:
  MulNormTPP() {}
  MulNormTPP(int rows, int cols) : MulNormTPP(rows, cols, cols, cols) {}
  MulNormTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1, // ldi0
            ldi, // ldi1
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in, Tin* in2, Tout* out) {
    kernel((void*)in, (void*)in2, (void*)out);
  }
  void ref(Tin* in, Tin* in2, Tout* out) {
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        out[r * ldo + c] = in[r] * in2[r * ldi + c];
  }

 private:
  int rows, cols;
  int ldi, ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout>
class ScaleAddTPP {
 public:
  ScaleAddTPP() {}
  ScaleAddTPP(int N) : ScaleAddTPP(1, N) {}
  ScaleAddTPP(int rows, int cols) : ScaleAddTPP(rows, cols, cols, cols) {}
  ScaleAddTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldi,
            ldo,
            XsmmDtype<float>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MULADD) {}
  void operator()(Tin* in, Tout* out, float scale) {
    float alpha = scale;
    kernel((void*)&alpha, (void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out, float scale) {
    float alpha = scale;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] += (float)in[i * ldi + j] * (float)alpha;
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tind, typename Tout>
class EmbeddingFwdTPP {
 public:
  EmbeddingFwdTPP() {}
  EmbeddingFwdTPP(int rows, int cols, int ldi)
      : EmbeddingFwdTPP(rows, cols, ldi, ldi) {}
  EmbeddingFwdTPP(int rows, int cols)
      : EmbeddingFwdTPP(rows, cols, cols, cols) {}
  EmbeddingFwdTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            (LIBXSMM_MELTW_FLAG_UNARY_GS_COLS |
             (sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES
                                : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES)),
            LIBXSMM_MELTW_TYPE_UNARY_GATHER) {}
  void operator()(Tin* in0, Tind* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, NULL, (void*)out, NULL);
  }
  void ref(Tin* in0, Tind* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      auto ind = in1[r];
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = in0[ind * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tind, typename Tout>
class EmbeddingBwdTPP {
 public:
  EmbeddingBwdTPP() {}
  EmbeddingBwdTPP(int E)
      : E(E),
        kernel(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            (libxsmm_meltw_unary_flags)(
                sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES
                                  : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES),
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD) {}
  void operator()(Tin* in0, Tind* in1, Tout* out, int N) {
    unsigned long long _N = N;
    kernel((void*)in0, (void*)in1, (void*)&_N, (void*)out, NULL);
  }
  void ref(Tin* in0, Tind* in1, Tout* out, int N) {
    for (long v = 0; v < E; v++)
      out[v] = 0;
    for (long s = 0; s < N; s++) {
      auto ind = in1[s];
      for (long v = 0; v < E; v++)
        out[v] += in0[ind * E + v];
    }
  }

 private:
  int E = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tind, typename Tout>
class ScatterTPP {
 public:
  ScatterTPP() {}
  ScatterTPP(int rows, int cols, int ldi) : ScatterTPP(rows, cols, ldi, ldi) {}
  ScatterTPP(int rows, int cols) : ScatterTPP(rows, cols, cols, cols) {}
  ScatterTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            (LIBXSMM_MELTW_FLAG_UNARY_GS_COLS |
             (sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES
                                : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES)),
            LIBXSMM_MELTW_TYPE_UNARY_SCATTER) {}
  void operator()(Tin* in, Tind* out1, Tout* out) {
    kernel((void*)in, NULL, NULL, (void*)out, (void*)out1);
  }
  void ref(Tin* in, Tind* out1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      auto ind = out1[r];
      for (int c = 0; c < cols; c++) {
        out[ind * ldo + c] = in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
};

class XformTPP {
 public:
  XformTPP() {}
  XformTPP(
      libxsmm_blasint rows_i,
      libxsmm_blasint cols_i,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dtype,
      libxsmm_meltw_unary_type type)
      : rows(rows_i),
        cols(cols_i),
        ldi(ldi),
        ldo(ldo),
        dtype(dtype),
        type(type),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            dtype,
            dtype,
            dtype,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            type) {}
  void operator()(void* in, void* out) {
    kernel(in, out);
  }
  typedef enum XFORM_TYPE {
    XFORM_NONE_TPP = 0,
    XFORM_XPOSE_TPP = 1,
    XFORM_N2V_TPP = 2,
    XFORM_XPOSE_N2V_TPP = 3,
    XFORM_XPOSE_V2V_TPP = 4
  } XFORM_TYPE;

 private:
  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_datatype dtype;
  libxsmm_meltw_unary_type type;
  UnaryTPP kernel;
};

}; // namespace tpp

#endif // _XSMM_FUNCTORS_H_
