# ------------------------------------------------------------
# SSE2 backend for x86_64 (baseline SIMD)
# Processes 4 floats (128-bit) or 4 ints at a time
# ------------------------------------------------------------
class SIMD::SSE2 < SIMD::Base
  VECTOR_WIDTH = 4 # 128-bit / 32-bit = 4 floats

  private def check_len(*slices)
    n = slices[0].as(Slice).size
    slices.each do |slice|
      raise ArgumentError.new("length mismatch") unless slice.as(Slice).size == n
    end
    n
  end

  def add(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         movups ($2), %xmm1
         addps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    # Scalar tail
    while i < n
      dst[i] = a[i] + b[i]
      i += 1
    end
  end

  def sub(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         movups ($2), %xmm1
         subps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] - b[i]
      i += 1
    end
  end

  def mul(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         movups ($2), %xmm1
         mulps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] * b[i]
      i += 1
    end
  end

  def fma(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32), c : Slice(Float32)) : Nil
    # SSE2 doesn't have FMA, so we use mul + add
    n = check_len(dst, a, b, c)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    c_ptr = c.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         movups ($2), %xmm1
         movups ($3), %xmm2
         mulps %xmm1, %xmm0
         addps %xmm2, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(c_ptr + i)
              : "xmm0", "xmm1", "xmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] * b[i] + c[i]
      i += 1
    end
  end

  def clamp(dst : Slice(Float32), a : Slice(Float32), lo : Float32, hi : Float32) : Nil
    # SSE2 clamp is slower than scalar due to setup overhead
    Log.trace { "clamp forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  def axpby(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32), alpha : Float32, beta : Float32) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    alpha_arr = StaticArray[alpha, alpha, alpha, alpha]
    beta_arr = StaticArray[beta, beta, beta, beta]
    alpha_ptr = alpha_arr.to_unsafe
    beta_ptr = beta_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         movups ($2), %xmm1
         movups ($3), %xmm2
         movups ($4), %xmm3
         mulps %xmm2, %xmm0
         mulps %xmm3, %xmm1
         addps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(alpha_ptr), "r"(beta_ptr)
              : "xmm0", "xmm1", "xmm2", "xmm3", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = alpha * a[i] + beta * b[i]
      i += 1
    end
  end

  def sum(a : Slice(Float32)) : Float32
    n = a.size
    a_ptr = a.to_unsafe

    acc = StaticArray[0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32]
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($0), %xmm0
         movups ($1), %xmm1
         addps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    # Horizontal sum of accumulator
    result = acc[0] + acc[1] + acc[2] + acc[3]

    # Add tail elements
    while i < n
      result += a[i]
      i += 1
    end

    result
  end

  def dot(a : Slice(Float32), b : Slice(Float32)) : Float32
    raise ArgumentError.new("length mismatch") unless a.size == b.size
    n = a.size
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    acc = StaticArray[0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32]
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($0), %xmm0
         movups ($1), %xmm1
         movups ($2), %xmm2
         mulps %xmm2, %xmm1
         addps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "xmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = acc[0] + acc[1] + acc[2] + acc[3]

    while i < n
      result += a[i] * b[i]
      i += 1
    end

    result
  end

  def max(a : Slice(Float32)) : Float32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    # Initialize with first vector
    max_vec = StaticArray(Float32, 4).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($0), %xmm0
         movups ($1), %xmm1
         maxps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    # Reduce vector to scalar max
    result = max_vec[0]
    result = max_vec[1] if max_vec[1] > result
    result = max_vec[2] if max_vec[2] > result
    result = max_vec[3] if max_vec[3] > result

    # Handle tail
    while i < n
      result = a[i] if a[i] > result
      i += 1
    end

    result
  end

  def bitwise_and(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pand %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] & b[i]
      i += 1
    end
  end

  def bitwise_or(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         por %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] | b[i]
      i += 1
    end
  end

  def xor(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pxor %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] ^ b[i]
      i += 1
    end
  end

  def bswap(dst : Slice(UInt32), a : Slice(UInt32)) : Nil
    n = check_len(dst, a)

    # SSE2 doesn't have pshufb, use scalar bswap
    i = 0
    while i < n
      v = a[i]
      dst[i] = ((v & 0x000000FF_u32) << 24) |
               ((v & 0x0000FF00_u32) << 8) |
               ((v & 0x00FF0000_u32) >> 8) |
               ((v & 0xFF000000_u32) >> 24)
      i += 1
    end
  end

  def popcount(a : Slice(UInt8)) : UInt64
    # SSE2 lacks popcount instruction; scalar is equally fast or faster
    SIMD::SCALAR_FALLBACK.popcount(a)
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    # SSE2 cmpps produces 32-bit masks, we need to extract byte masks
    i = 0
    while i + VECTOR_WIDTH <= n
      mask_tmp = uninitialized StaticArray(UInt32, 4)
      asm(
        "movups ($1), %xmm0
         movups ($2), %xmm1
         cmpltps %xmm0, %xmm1
         movdqu %xmm1, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      # cmpltps sets all 1s (0xFFFFFFFF) if b < a (i.e., a > b), else 0
      dst_mask[i] = mask_tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = mask_tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 2] = mask_tmp[2] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 3] = mask_tmp[3] != 0 ? 0xFF_u8 : 0x00_u8
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def blend(dst : Slice(Float32), t : Slice(Float32), f : Slice(Float32), mask : Slice(UInt8)) : Nil
    # SSE2 lacks blendvps; use scalar fallback
    n = check_len(dst, t, f, mask)
    i = 0
    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  def compress(dst : Slice(Float32), src : Slice(Float32), mask : Slice(UInt8)) : Int32
    # SSE2 has no compress instruction; use scalar
    raise ArgumentError.new("length mismatch") unless src.size == mask.size
    outp = 0
    i = 0
    n = src.size
    while i < n
      if mask[i] != 0
        raise ArgumentError.new("dst too small") if outp >= dst.size
        dst[outp] = src[i]
        outp += 1
      end
      i += 1
    end
    outp
  end

  def copy(dst : Slice(UInt8), src : Slice(UInt8)) : Nil
    # SSE2 copy is slower than scalar/LLVM memcpy optimization
    Log.trace { "copy forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.copy(dst, src)
  end

  def fill(dst : Slice(UInt8), value : UInt8) : Nil
    # Scalar fill is faster (LLVM optimizes to memset)
    Log.trace { "fill forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.fill(dst, value)
  end

  def convert(dst : Slice(Float32), src : Slice(Int16)) : Nil
    # SSE2 i16_to_f32 is slower due to manual unpacking overhead
    Log.trace { "convert forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.convert(dst, src)
  end

  def convert(dst : Slice(Float32), src : Slice(UInt8), scale : Float32) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe

    scale_arr = StaticArray[scale, scale, scale, scale]
    scale_ptr = scale_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # Unpack u8 to i32, then convert to f32 and scale
      tmp = uninitialized StaticArray(Int32, 4)
      tmp[0] = src[i].to_i32
      tmp[1] = src[i + 1].to_i32
      tmp[2] = src[i + 2].to_i32
      tmp[3] = src[i + 3].to_i32

      asm(
        "movdqu ($1), %xmm0
         cvtdq2ps %xmm0, %xmm0
         movups ($2), %xmm1
         mulps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(tmp.to_unsafe), "r"(scale_ptr)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = src[i].to_f32 * scale
      i += 1
    end
  end

  def fir(dst : Slice(Float32), src : Slice(Float32), coeff : Slice(Float32)) : Nil
    # SSE2 FIR is slower than scalar due to setup overhead
    Log.trace { "fir forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.fir(dst, src, coeff)
  end

  def xor_block16(dst : Slice(UInt8), src : Slice(UInt8), key16 : StaticArray(UInt8, 16)) : Nil
    raise ArgumentError.new("length mismatch") unless dst.size == src.size
    n = src.size
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe
    key_ptr = key16.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pxor %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i), "r"(key_ptr)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      dst[i] = src[i] ^ key16[i & 15]
      i += 1
    end
  end

  # ============================================================
  # FLOAT32 - min (new)
  # ============================================================

  def min(a : Slice(Float32)) : Float32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] < m }
      return m
    end

    min_vec = StaticArray(Float32, 4).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($0), %xmm0
         movups ($1), %xmm1
         minps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = min_vec[0]
    result = min_vec[1] if min_vec[1] < result
    result = min_vec[2] if min_vec[2] < result
    result = min_vec[3] if min_vec[3] < result

    while i < n
      result = a[i] if a[i] < result
      i += 1
    end

    result
  end

  # ============================================================
  # FLOAT64 OPERATIONS
  # ============================================================

  VECTOR_WIDTH_F64 = 2 # 128-bit / 64-bit = 2 doubles

  def add(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($1), %xmm0
         movupd ($2), %xmm1
         addpd %xmm1, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = a[i] + b[i]
      i += 1
    end
  end

  def sub(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($1), %xmm0
         movupd ($2), %xmm1
         subpd %xmm1, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = a[i] - b[i]
      i += 1
    end
  end

  def mul(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($1), %xmm0
         movupd ($2), %xmm1
         mulpd %xmm1, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = a[i] * b[i]
      i += 1
    end
  end

  def fma(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64), c : Slice(Float64)) : Nil
    n = check_len(dst, a, b, c)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    c_ptr = c.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($1), %xmm0
         movupd ($2), %xmm1
         movupd ($3), %xmm2
         mulpd %xmm1, %xmm0
         addpd %xmm2, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(c_ptr + i)
              : "xmm0", "xmm1", "xmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = a[i] * b[i] + c[i]
      i += 1
    end
  end

  def clamp(dst : Slice(Float64), a : Slice(Float64), lo : Float64, hi : Float64) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  def axpby(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64), alpha : Float64, beta : Float64) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    alpha_arr = StaticArray[alpha, alpha]
    beta_arr = StaticArray[beta, beta]
    alpha_ptr = alpha_arr.to_unsafe
    beta_ptr = beta_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($1), %xmm0
         movupd ($2), %xmm1
         movupd ($3), %xmm2
         movupd ($4), %xmm3
         mulpd %xmm2, %xmm0
         mulpd %xmm3, %xmm1
         addpd %xmm1, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(alpha_ptr), "r"(beta_ptr)
              : "xmm0", "xmm1", "xmm2", "xmm3", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = alpha * a[i] + beta * b[i]
      i += 1
    end
  end

  def sum(a : Slice(Float64)) : Float64
    n = a.size
    a_ptr = a.to_unsafe

    acc = StaticArray[0.0_f64, 0.0_f64]
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($0), %xmm0
         movupd ($1), %xmm1
         addpd %xmm1, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = acc[0] + acc[1]

    while i < n
      result += a[i]
      i += 1
    end

    result
  end

  def dot(a : Slice(Float64), b : Slice(Float64)) : Float64
    raise ArgumentError.new("length mismatch") unless a.size == b.size
    n = a.size
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    acc = StaticArray[0.0_f64, 0.0_f64]
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($0), %xmm0
         movupd ($1), %xmm1
         movupd ($2), %xmm2
         mulpd %xmm2, %xmm1
         addpd %xmm1, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "xmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = acc[0] + acc[1]

    while i < n
      result += a[i] * b[i]
      i += 1
    end

    result
  end

  def max(a : Slice(Float64)) : Float64
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_F64
      return a[0]
    end

    max_vec = StaticArray(Float64, 2).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_F64
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($0), %xmm0
         movupd ($1), %xmm1
         maxpd %xmm1, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = max_vec[0]
    result = max_vec[1] if max_vec[1] > result

    while i < n
      result = a[i] if a[i] > result
      i += 1
    end

    result
  end

  def min(a : Slice(Float64)) : Float64
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_F64
      return a[0]
    end

    min_vec = StaticArray(Float64, 2).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_F64
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($0), %xmm0
         movupd ($1), %xmm1
         minpd %xmm1, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = min_vec[0]
    result = min_vec[1] if min_vec[1] < result

    while i < n
      result = a[i] if a[i] < result
      i += 1
    end

    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      mask_tmp = uninitialized StaticArray(UInt64, 2)
      asm(
        "movupd ($1), %xmm0
         movupd ($2), %xmm1
         cmpltpd %xmm0, %xmm1
         movupd %xmm1, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      dst_mask[i] = mask_tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = mask_tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def blend(dst : Slice(Float64), t : Slice(Float64), f : Slice(Float64), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    i = 0
    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  def compress(dst : Slice(Float64), src : Slice(Float64), mask : Slice(UInt8)) : Int32
    raise ArgumentError.new("length mismatch") unless src.size == mask.size
    outp = 0
    i = 0
    n = src.size
    while i < n
      if mask[i] != 0
        raise ArgumentError.new("dst too small") if outp >= dst.size
        dst[outp] = src[i]
        outp += 1
      end
      i += 1
    end
    outp
  end

  def fir(dst : Slice(Float64), src : Slice(Float64), coeff : Slice(Float64)) : Nil
    SIMD::SCALAR_FALLBACK.fir(dst, src, coeff)
  end

  # ============================================================
  # UINT64 OPERATIONS
  # ============================================================

  VECTOR_WIDTH_64 = 2 # 128-bit / 64-bit = 2

  def add(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         paddq %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst[i] = a[i] &+ b[i]
      i += 1
    end
  end

  def sub(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         psubq %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst[i] = a[i] &- b[i]
      i += 1
    end
  end

  def mul(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    # SSE2 lacks 64-bit multiply; use scalar
    SIMD::SCALAR_FALLBACK.mul(dst, a, b)
  end

  def clamp(dst : Slice(UInt64), a : Slice(UInt64), lo : UInt64, hi : UInt64) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  def sum(a : Slice(UInt64)) : UInt64
    SIMD::SCALAR_FALLBACK.sum(a)
  end

  def max(a : Slice(UInt64)) : UInt64
    SIMD::SCALAR_FALLBACK.max(a)
  end

  def min(a : Slice(UInt64)) : UInt64
    SIMD::SCALAR_FALLBACK.min(a)
  end

  def bitwise_and(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pand %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst[i] = a[i] & b[i]
      i += 1
    end
  end

  def bitwise_or(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         por %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst[i] = a[i] | b[i]
      i += 1
    end
  end

  def xor(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pxor %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst[i] = a[i] ^ b[i]
      i += 1
    end
  end

  def bswap(dst : Slice(UInt64), a : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.bswap(dst, a)
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  def blend(dst : Slice(UInt64), t : Slice(UInt64), f : Slice(UInt64), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    i = 0
    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  # ============================================================
  # INT64 OPERATIONS (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int64), a : Slice(Int64), lo : Int64, hi : Int64) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  def max(a : Slice(Int64)) : Int64
    SIMD::SCALAR_FALLBACK.max(a)
  end

  def min(a : Slice(Int64)) : Int64
    SIMD::SCALAR_FALLBACK.min(a)
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int64), b : Slice(Int64)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  # ============================================================
  # UINT32 OPERATIONS (new)
  # ============================================================

  def add(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         paddd %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] &+ b[i]
      i += 1
    end
  end

  def sub(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         psubd %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] &- b[i]
      i += 1
    end
  end

  def mul(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    # SSE2 pmuludq only produces 64-bit results; scalar is simpler
    SIMD::SCALAR_FALLBACK.mul(dst, a, b)
  end

  def clamp(dst : Slice(UInt32), a : Slice(UInt32), lo : UInt32, hi : UInt32) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  def sum(a : Slice(UInt32)) : UInt32
    SIMD::SCALAR_FALLBACK.sum(a)
  end

  def max(a : Slice(UInt32)) : UInt32
    SIMD::SCALAR_FALLBACK.max(a)
  end

  def min(a : Slice(UInt32)) : UInt32
    SIMD::SCALAR_FALLBACK.min(a)
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  def blend(dst : Slice(UInt32), t : Slice(UInt32), f : Slice(UInt32), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    i = 0
    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  # ============================================================
  # INT32 OPERATIONS (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int32), a : Slice(Int32), lo : Int32, hi : Int32) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  def max(a : Slice(Int32)) : Int32
    SIMD::SCALAR_FALLBACK.max(a)
  end

  def min(a : Slice(Int32)) : Int32
    SIMD::SCALAR_FALLBACK.min(a)
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      mask_tmp = uninitialized StaticArray(Int32, 4)
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pcmpgtd %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      dst_mask[i] = mask_tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = mask_tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 2] = mask_tmp[2] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 3] = mask_tmp[3] != 0 ? 0xFF_u8 : 0x00_u8
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  # ============================================================
  # UINT16 OPERATIONS
  # ============================================================

  VECTOR_WIDTH_16 = 8 # 128-bit / 16-bit = 8

  def add(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         paddw %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst[i] = a[i] &+ b[i]
      i += 1
    end
  end

  def sub(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         psubw %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst[i] = a[i] &- b[i]
      i += 1
    end
  end

  def mul(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pmullw %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst[i] = a[i] &* b[i]
      i += 1
    end
  end

  def clamp(dst : Slice(UInt16), a : Slice(UInt16), lo : UInt16, hi : UInt16) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  def sum(a : Slice(UInt16)) : UInt16
    SIMD::SCALAR_FALLBACK.sum(a)
  end

  def max(a : Slice(UInt16)) : UInt16
    SIMD::SCALAR_FALLBACK.max(a)
  end

  def min(a : Slice(UInt16)) : UInt16
    SIMD::SCALAR_FALLBACK.min(a)
  end

  def bitwise_and(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pand %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst[i] = a[i] & b[i]
      i += 1
    end
  end

  def bitwise_or(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         por %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst[i] = a[i] | b[i]
      i += 1
    end
  end

  def xor(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pxor %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst[i] = a[i] ^ b[i]
      i += 1
    end
  end

  def bswap(dst : Slice(UInt16), a : Slice(UInt16)) : Nil
    SIMD::SCALAR_FALLBACK.bswap(dst, a)
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  def blend(dst : Slice(UInt16), t : Slice(UInt16), f : Slice(UInt16), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    i = 0
    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  # ============================================================
  # INT16 OPERATIONS (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int16), a : Slice(Int16), lo : Int16, hi : Int16) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  def max(a : Slice(Int16)) : Int16
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_16
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    max_vec = StaticArray(Int16, 8).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_16
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pmaxsw %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    result = max_vec[0]
    (1...8).each { |j| result = max_vec[j] if max_vec[j] > result }

    while i < n
      result = a[i] if a[i] > result
      i += 1
    end

    result
  end

  def min(a : Slice(Int16)) : Int16
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_16
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] < m }
      return m
    end

    min_vec = StaticArray(Int16, 8).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_16
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pminsw %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    result = min_vec[0]
    (1...8).each { |j| result = min_vec[j] if min_vec[j] < result }

    while i < n
      result = a[i] if a[i] < result
      i += 1
    end

    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int16), b : Slice(Int16)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      mask_tmp = uninitialized StaticArray(Int16, 8)
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pcmpgtw %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      (0...8).each { |j| dst_mask[i + j] = mask_tmp[j] != 0 ? 0xFF_u8 : 0x00_u8 }
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  # ============================================================
  # UINT8 OPERATIONS
  # ============================================================

  VECTOR_WIDTH_8 = 16 # 128-bit / 8-bit = 16

  def add(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         paddb %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst[i] = a[i] &+ b[i]
      i += 1
    end
  end

  def sub(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         psubb %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst[i] = a[i] &- b[i]
      i += 1
    end
  end

  def mul(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    # SSE2 lacks 8-bit multiply; use scalar
    SIMD::SCALAR_FALLBACK.mul(dst, a, b)
  end

  def clamp(dst : Slice(UInt8), a : Slice(UInt8), lo : UInt8, hi : UInt8) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  def sum(a : Slice(UInt8)) : UInt8
    SIMD::SCALAR_FALLBACK.sum(a)
  end

  def max(a : Slice(UInt8)) : UInt8
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_8
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    max_vec = StaticArray(UInt8, 16).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pmaxub %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    result = max_vec[0]
    (1...16).each { |j| result = max_vec[j] if max_vec[j] > result }

    while i < n
      result = a[i] if a[i] > result
      i += 1
    end

    result
  end

  def min(a : Slice(UInt8)) : UInt8
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_8
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] < m }
      return m
    end

    min_vec = StaticArray(UInt8, 16).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pminub %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    result = min_vec[0]
    (1...16).each { |j| result = min_vec[j] if min_vec[j] < result }

    while i < n
      result = a[i] if a[i] < result
      i += 1
    end

    result
  end

  def bitwise_and(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pand %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst[i] = a[i] & b[i]
      i += 1
    end
  end

  def bitwise_or(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         por %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst[i] = a[i] | b[i]
      i += 1
    end
  end

  def xor(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pxor %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst[i] = a[i] ^ b[i]
      i += 1
    end
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  def blend(dst : Slice(UInt8), t : Slice(UInt8), f : Slice(UInt8), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    i = 0
    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  # ============================================================
  # INT8 OPERATIONS (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int8), a : Slice(Int8), lo : Int8, hi : Int8) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  def max(a : Slice(Int8)) : Int8
    SIMD::SCALAR_FALLBACK.max(a)
  end

  def min(a : Slice(Int8)) : Int8
    SIMD::SCALAR_FALLBACK.min(a)
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      mask_tmp = uninitialized StaticArray(Int8, 16)
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pcmpgtb %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      (0...16).each { |j| dst_mask[i + j] = mask_tmp[j] != 0 ? 0xFF_u8 : 0x00_u8 }
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  # ============================================================
  # CONVERSION OPERATIONS (new)
  # ============================================================

  def convert(dst : Slice(Float64), src : Slice(Int32)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movq ($1), %xmm0
         cvtdq2pd %xmm0, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = src[i].to_f64
      i += 1
    end
  end

  def convert(dst : Slice(Float64), src : Slice(Int16)) : Nil
    SIMD::SCALAR_FALLBACK.convert(dst, src)
  end

  def convert(dst : Slice(Float64), src : Slice(Float32)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movq ($1), %xmm0
         cvtps2pd %xmm0, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = src[i].to_f64
      i += 1
    end
  end

  def convert(dst : Slice(Float32), src : Slice(Float64)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($1), %xmm0
         cvtpd2ps %xmm0, %xmm0
         movq %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = src[i].to_f32
      i += 1
    end
  end
end
