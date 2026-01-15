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

  # ============================================================
  # Additional operations (non-float / generic)
  # ============================================================

  @[AlwaysInline]
  def abs(dst : Slice(UInt8), a : Slice(UInt8)) : Nil
    # SSE2 abs is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  @[AlwaysInline]
  def abs(dst : Slice(Int8), a : Slice(Int8)) : Nil
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  @[AlwaysInline]
  def abs(dst : Slice(UInt16), a : Slice(UInt16)) : Nil
    # SSE2 abs is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  @[AlwaysInline]
  def abs(dst : Slice(Int16), a : Slice(Int16)) : Nil
    # SSE2 abs is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  @[AlwaysInline]
  def abs(dst : Slice(UInt32), a : Slice(UInt32)) : Nil
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  @[AlwaysInline]
  def abs(dst : Slice(Int32), a : Slice(Int32)) : Nil
    # SSE2 abs is slower than scalar for Int32
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  @[AlwaysInline]
  def abs(dst : Slice(UInt64), a : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  @[AlwaysInline]
  def abs(dst : Slice(Int64), a : Slice(Int64)) : Nil
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  def neg(dst : Slice(Int8), a : Slice(Int8)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "pxor %xmm0, %xmm0
         movdqu ($1), %xmm1
         psubb %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      dst[i] = 0_i8 &- a[i]
      i += 1
    end
  end

  @[AlwaysInline]
  def neg(dst : Slice(Int16), a : Slice(Int16)) : Nil
    # SSE2 neg is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.neg(dst, a)
  end

  @[AlwaysInline]
  def neg(dst : Slice(Int32), a : Slice(Int32)) : Nil
    # SSE2 neg is slower than scalar for Int32
    SIMD::SCALAR_FALLBACK.neg(dst, a)
  end

  @[AlwaysInline]
  def neg(dst : Slice(Int64), a : Slice(Int64)) : Nil
    # SSE2 neg is slower than scalar for Int32
    SIMD::SCALAR_FALLBACK.neg(dst, a)
  end

  @[AlwaysInline]
  def argmax(a : Slice(T)) : Int32 forall T
    SIMD::SCALAR_FALLBACK.argmax(a)
  end

  @[AlwaysInline]
  def argmin(a : Slice(T)) : Int32 forall T
    SIMD::SCALAR_FALLBACK.argmin(a)
  end

  @[AlwaysInline]
  def any_nonzero(a : Slice(T)) : Bool forall T
    SIMD::SCALAR_FALLBACK.any_nonzero(a)
  end

  @[AlwaysInline]
  def all_nonzero(a : Slice(T)) : Bool forall T
    SIMD::SCALAR_FALLBACK.all_nonzero(a)
  end

  @[AlwaysInline]
  def cmp_eq(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    SIMD::SCALAR_FALLBACK.cmp_eq(dst_mask, a, b)
  end

  @[AlwaysInline]
  def min(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    SIMD::SCALAR_FALLBACK.min(dst, a, b)
  end

  @[AlwaysInline]
  def max(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    SIMD::SCALAR_FALLBACK.max(dst, a, b)
  end

  @[AlwaysInline]
  def min(dst : Slice(Int16), a : Slice(Int16), b : Slice(Int16)) : Nil
    # SSE2 min is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.min(dst, a, b)
  end

  @[AlwaysInline]
  def max(dst : Slice(Int16), a : Slice(Int16), b : Slice(Int16)) : Nil
    # SSE2 max is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.max(dst, a, b)
  end

  @[AlwaysInline]
  def min(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    # SSE2 min is slower than scalar for Int32
    SIMD::SCALAR_FALLBACK.min(dst, a, b)
  end

  @[AlwaysInline]
  def max(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    # SSE2 max is slower than scalar for Int32
    SIMD::SCALAR_FALLBACK.max(dst, a, b)
  end

  def cmp_eq(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst_mask, a, b)
    dst_ptr = dst_mask.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pcmpeqb %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      dst_mask[i] = a[i] == b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ne(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst_mask, a, b)
    dst_ptr = dst_mask.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pcmpeqb %xmm1, %xmm0
         pcmpeqb %xmm2, %xmm2
         pxor %xmm2, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "xmm2", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      dst_mask[i] = a[i] != b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_lt(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst_mask, a, b)
    dst_ptr = dst_mask.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pcmpgtb %xmm0, %xmm1
         movdqu %xmm1, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      dst_mask[i] = a[i] < b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_le(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst_mask, a, b)
    dst_ptr = dst_mask.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pcmpgtb %xmm1, %xmm0
         pcmpeqb %xmm2, %xmm2
         pxor %xmm2, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "xmm2", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      dst_mask[i] = a[i] <= b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    cmp_le(dst_mask, b, a)
  end

  @[AlwaysInline]
  def cmp_eq(dst_mask : Slice(UInt8), a : Slice(Int16), b : Slice(Int16)) : Nil
    # SSE2 cmp_eq is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.cmp_eq(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_ne(dst_mask : Slice(UInt8), a : Slice(Int16), b : Slice(Int16)) : Nil
    # SSE2 cmp_ne is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.cmp_ne(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_lt(dst_mask : Slice(UInt8), a : Slice(Int16), b : Slice(Int16)) : Nil
    # SSE2 cmp_lt is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.cmp_lt(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_le(dst_mask : Slice(UInt8), a : Slice(Int16), b : Slice(Int16)) : Nil
    # SSE2 cmp_le is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.cmp_le(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(Int16), b : Slice(Int16)) : Nil
    # SSE2 cmp_ge is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.cmp_ge(dst_mask, a, b)
  end

  def cmp_ne(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 4 <= n
      tmp = uninitialized StaticArray(Int32, 4)
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pcmpeqd %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      (0...4).each { |j| dst_mask[i + j] = tmp[j] == 0 ? 0xFF_u8 : 0x00_u8 }
      i += 4
    end

    while i < n
      dst_mask[i] = a[i] != b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  @[AlwaysInline]
  def cmp_lt(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_lt(dst_mask, a, b)
  end

  def cmp_le(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 4 <= n
      tmp = uninitialized StaticArray(Int32, 4)
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pcmpgtd %xmm1, %xmm0
         pcmpeqd %xmm2, %xmm2
         pxor %xmm2, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "xmm2", "memory"
              : "volatile"
      )
      (0...4).each { |j| dst_mask[i + j] = tmp[j] != 0 ? 0xFF_u8 : 0x00_u8 }
      i += 4
    end

    while i < n
      dst_mask[i] = a[i] <= b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    cmp_le(dst_mask, b, a)
  end

  @[AlwaysInline]
  def cmp_ne(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    SIMD::SCALAR_FALLBACK.cmp_ne(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_lt(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    SIMD::SCALAR_FALLBACK.cmp_lt(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_le(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    SIMD::SCALAR_FALLBACK.cmp_le(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    SIMD::SCALAR_FALLBACK.cmp_ge(dst_mask, a, b)
  end

  @[AlwaysInline]
  def min(dst : Slice(T), a : Slice(T), b : Slice(T)) : Nil forall T
    SIMD::SCALAR_FALLBACK.min(dst, a, b)
  end

  @[AlwaysInline]
  def max(dst : Slice(T), a : Slice(T), b : Slice(T)) : Nil forall T
    SIMD::SCALAR_FALLBACK.max(dst, a, b)
  end

  @[AlwaysInline]
  def gather(dst : Slice(T), src : Slice(T), indices : Slice(Int32)) : Nil forall T
    # SSE2 lacks gather instruction; scalar is faster
    SIMD::SCALAR_FALLBACK.gather(dst, src, indices)
  end

  @[AlwaysInline]
  def scatter(dst : Slice(T), src : Slice(T), indices : Slice(Int32)) : Nil forall T
    # SSE2 lacks scatter instruction; scalar is faster
    SIMD::SCALAR_FALLBACK.scatter(dst, src, indices)
  end

  @[AlwaysInline]
  def transpose(dst : Slice(T), src : Slice(T), rows : Int32, cols : Int32) : Nil forall T
    SIMD::SCALAR_FALLBACK.transpose(dst, src, rows, cols)
  end

  @[AlwaysInline]
  def gemv(dst : Slice(Float32), matrix : Slice(Float32), vector : Slice(Float32), alpha : Float32, beta : Float32) : Nil
    SIMD::SCALAR_FALLBACK.gemv(dst, matrix, vector, alpha, beta)
  end

  @[AlwaysInline]
  def gemv(dst : Slice(Float64), matrix : Slice(Float64), vector : Slice(Float64), alpha : Float64, beta : Float64) : Nil
    # SSE2 gemv is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.gemv(dst, matrix, vector, alpha, beta)
  end

  @[AlwaysInline]
  def gemm(c : Slice(Float32), a : Slice(Float32), b : Slice(Float32), alpha : Float32, beta : Float32) : Nil
    # SSE2 gemv is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.gemm(c, a, b, alpha, beta)
  end

  @[AlwaysInline]
  def gemm(c : Slice(Float64), a : Slice(Float64), b : Slice(Float64), alpha : Float64, beta : Float64) : Nil
    SIMD::SCALAR_FALLBACK.gemm(c, a, b, alpha, beta)
  end

  @[AlwaysInline]
  def add(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    # SSE2 add is slower than scalar due to setup overhead
    SIMD::SCALAR_FALLBACK.add(dst, a, b)
  end

  @[AlwaysInline]
  def sub(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    # SSE2 sub is slower than scalar due to setup overhead
    SIMD::SCALAR_FALLBACK.sub(dst, a, b)
  end

  @[AlwaysInline]
  def mul(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    # SSE2 mul is slower than scalar due to setup overhead
    SIMD::SCALAR_FALLBACK.mul(dst, a, b)
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

  @[AlwaysInline]
  def clamp(dst : Slice(Float32), a : Slice(Float32), lo : Float32, hi : Float32) : Nil
    # SSE2 clamp is slower than scalar due to setup overhead
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  @[AlwaysInline]
  def axpby(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32), alpha : Float32, beta : Float32) : Nil
    # SSE2 axpby is slower than scalar due to setup overhead
    SIMD::SCALAR_FALLBACK.axpby(dst, a, b, alpha, beta)
  end

  def div(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         movups ($2), %xmm1
         divps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] / b[i]
      i += 1
    end
  end

  def sqrt(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         sqrtps %xmm0, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = Math.sqrt(a[i])
      i += 1
    end
  end

  @[AlwaysInline]
  def rsqrt(dst : Slice(Float32), a : Slice(Float32)) : Nil
    # SSE2 rsqrt is slower than scalar due to setup overhead
    SIMD::SCALAR_FALLBACK.rsqrt(dst, a)
  end

  @[AlwaysInline]
  def abs(dst : Slice(Float32), a : Slice(Float32)) : Nil
    # SSE2 abs is slower than scalar due to setup overhead
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  @[AlwaysInline]
  def neg(dst : Slice(Float32), a : Slice(Float32)) : Nil
    # SSE2 neg is slower than scalar due to setup overhead
    SIMD::SCALAR_FALLBACK.neg(dst, a)
  end

  def min(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         movups ($2), %xmm1
         minps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av < bv ? av : bv
      i += 1
    end
  end

  def max(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         movups ($2), %xmm1
         maxps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av > bv ? av : bv
      i += 1
    end
  end

  # SSE2 doesn't provide vector floor/ceil/round for Float32.
  # (SSE4.1 overrides these in SIMD::SSE41.)
  @[AlwaysInline]
  def floor(dst : Slice(Float32), a : Slice(Float32)) : Nil
    # SSE2 lacks roundps; scalar is faster
    SIMD::SCALAR_FALLBACK.floor(dst, a)
  end

  @[AlwaysInline]
  def ceil(dst : Slice(Float32), a : Slice(Float32)) : Nil
    # SSE2 lacks roundps; scalar is faster
    SIMD::SCALAR_FALLBACK.ceil(dst, a)
  end

  @[AlwaysInline]
  def round(dst : Slice(Float32), a : Slice(Float32)) : Nil
    # SSE2 lacks roundps; scalar is faster
    SIMD::SCALAR_FALLBACK.round(dst, a)
  end

  @[AlwaysInline]
  def cmp_eq(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    # SSE2 cmp_eq is slower than scalar due to mask extraction overhead
    SIMD::SCALAR_FALLBACK.cmp_eq(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_ne(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    # SSE2 cmp_ne is slower than scalar due to mask extraction overhead
    SIMD::SCALAR_FALLBACK.cmp_ne(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_lt(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    # SSE2 cmp_lt is slower than scalar due to mask extraction overhead
    SIMD::SCALAR_FALLBACK.cmp_lt(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_le(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    # SSE2 cmp_le is slower than scalar due to mask extraction overhead
    SIMD::SCALAR_FALLBACK.cmp_le(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    # SSE2 cmp_ge is slower than scalar due to mask extraction overhead
    SIMD::SCALAR_FALLBACK.cmp_ge(dst_mask, a, b)
  end

  @[AlwaysInline]
  def sum(a : Slice(Float32)) : Float32
    # SSE2 sum is slower than scalar
    SIMD::SCALAR_FALLBACK.sum(a)
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

  @[AlwaysInline]
  def bitwise_and(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    SIMD::SCALAR_FALLBACK.bitwise_and(dst, a, b)
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

  @[AlwaysInline]
  def bswap(dst : Slice(UInt32), a : Slice(UInt32)) : Nil
    SIMD::SCALAR_FALLBACK.bswap(dst, a)
  end

  @[AlwaysInline]
  def popcount(a : Slice(UInt8)) : UInt64
    # SSE2 lacks popcount instruction; scalar is equally fast or faster
    SIMD::SCALAR_FALLBACK.popcount(a)
  end

  @[AlwaysInline]
  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  @[AlwaysInline]
  def compress(dst : Slice(Float32), src : Slice(Float32), mask : Slice(UInt8)) : Int32
    # SSE2 has no compress instruction; use scalar
    SIMD::SCALAR_FALLBACK.compress(dst, src, mask)
  end

  @[AlwaysInline]
  def copy(dst : Slice(UInt8), src : Slice(UInt8)) : Nil
    # SSE2 copy is slower than scalar/LLVM memcpy optimization
    SIMD::SCALAR_FALLBACK.copy(dst, src)
  end

  @[AlwaysInline]
  def fill(dst : Slice(UInt8), value : UInt8) : Nil
    # Scalar fill is faster (LLVM optimizes to memset)
    SIMD::SCALAR_FALLBACK.fill(dst, value)
  end

  @[AlwaysInline]
  def convert(dst : Slice(Float32), src : Slice(Int16)) : Nil
    # SSE2 i16_to_f32 is slower due to manual unpacking overhead
    SIMD::SCALAR_FALLBACK.convert(dst, src)
  end

  @[AlwaysInline]
  def convert(dst : Slice(Float32), src : Slice(UInt8), scale : Float32) : Nil
    # SSE2 convert with scale is slower than scalar
    SIMD::SCALAR_FALLBACK.convert(dst, src, scale)
  end

  @[AlwaysInline]
  def fir(dst : Slice(Float32), src : Slice(Float32), coeff : Slice(Float32)) : Nil
    # SSE2 FIR is slower than scalar due to setup overhead
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

  @[AlwaysInline]
  def add(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    # SSE2 add is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.add(dst, a, b)
  end

  @[AlwaysInline]
  def sub(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    # SSE2 sub is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.sub(dst, a, b)
  end

  @[AlwaysInline]
  def mul(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    # SSE2 mul is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.mul(dst, a, b)
  end

  @[AlwaysInline]
  def fma(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64), c : Slice(Float64)) : Nil
    # SSE2 fma is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.fma(dst, a, b, c)
  end

  @[AlwaysInline]
  def clamp(dst : Slice(Float64), a : Slice(Float64), lo : Float64, hi : Float64) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  @[AlwaysInline]
  def axpby(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64), alpha : Float64, beta : Float64) : Nil
    # SSE2 axpby is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.axpby(dst, a, b, alpha, beta)
  end

  @[AlwaysInline]
  def div(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    # SSE2 div is marginally slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.div(dst, a, b)
  end

  @[AlwaysInline]
  def sqrt(dst : Slice(Float64), a : Slice(Float64)) : Nil
    # SSE2 sqrt is marginally slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.sqrt(dst, a)
  end

  @[AlwaysInline]
  def rsqrt(dst : Slice(Float64), a : Slice(Float64)) : Nil
    # SSE2 rsqrt is marginally slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.rsqrt(dst, a)
  end

  @[AlwaysInline]
  def abs(dst : Slice(Float64), a : Slice(Float64)) : Nil
    # SSE2 abs is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  @[AlwaysInline]
  def neg(dst : Slice(Float64), a : Slice(Float64)) : Nil
    # SSE2 neg is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.neg(dst, a)
  end

  @[AlwaysInline]
  def min(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    # SSE2 min is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.min(dst, a, b)
  end

  @[AlwaysInline]
  def max(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    # SSE2 max is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.max(dst, a, b)
  end

  @[AlwaysInline]
  def floor(dst : Slice(Float64), a : Slice(Float64)) : Nil
    SIMD::SCALAR_FALLBACK.floor(dst, a)
  end

  @[AlwaysInline]
  def ceil(dst : Slice(Float64), a : Slice(Float64)) : Nil
    SIMD::SCALAR_FALLBACK.ceil(dst, a)
  end

  @[AlwaysInline]
  def round(dst : Slice(Float64), a : Slice(Float64)) : Nil
    SIMD::SCALAR_FALLBACK.round(dst, a)
  end

  def cmp_eq(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      tmp = uninitialized StaticArray(UInt64, 2)
      asm(
        "movupd ($1), %xmm0
         movupd ($2), %xmm1
         cmppd $$0, %xmm1, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      dst_mask[i] = tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst_mask[i] = a[i] == b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ne(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      tmp = uninitialized StaticArray(UInt64, 2)
      asm(
        "movupd ($1), %xmm0
         movupd ($2), %xmm1
         cmppd $$4, %xmm1, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      dst_mask[i] = tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst_mask[i] = a[i] != b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  @[AlwaysInline]
  def cmp_lt(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    # SSE2 cmp_lt is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.cmp_lt(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_le(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    # SSE2 cmp_le is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.cmp_le(dst_mask, a, b)
  end

  @[AlwaysInline]
  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    cmp_le(dst_mask, b, a)
  end

  @[AlwaysInline]
  def sum(a : Slice(Float64)) : Float64
    # SSE2 sum is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.sum(a)
  end

  @[AlwaysInline]
  def dot(a : Slice(Float64), b : Slice(Float64)) : Float64
    # SSE2 dot is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.dot(a, b)
  end

  @[AlwaysInline]
  def max(a : Slice(Float64)) : Float64
    # SSE2 max_reduce is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.max(a)
  end

  @[AlwaysInline]
  def min(a : Slice(Float64)) : Float64
    # SSE2 min_reduce is slower than scalar for Float64
    SIMD::SCALAR_FALLBACK.min(a)
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

  @[AlwaysInline]
  def compress(dst : Slice(Float64), src : Slice(Float64), mask : Slice(UInt8)) : Int32
    SIMD::SCALAR_FALLBACK.compress(dst, src, mask)
  end

  @[AlwaysInline]
  def fir(dst : Slice(Float64), src : Slice(Float64), coeff : Slice(Float64)) : Nil
    SIMD::SCALAR_FALLBACK.fir(dst, src, coeff)
  end

  # ============================================================
  # UINT64 OPERATIONS
  # ============================================================

  VECTOR_WIDTH_64 = 2 # 128-bit / 64-bit = 2

  @[AlwaysInline]
  def add(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.add(dst, a, b)
  end

  @[AlwaysInline]
  def sub(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.sub(dst, a, b)
  end

  @[AlwaysInline]
  def mul(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    # SSE2 lacks 64-bit multiply; use scalar
    SIMD::SCALAR_FALLBACK.mul(dst, a, b)
  end

  @[AlwaysInline]
  def clamp(dst : Slice(UInt64), a : Slice(UInt64), lo : UInt64, hi : UInt64) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  @[AlwaysInline]
  def sum(a : Slice(UInt64)) : UInt64
    SIMD::SCALAR_FALLBACK.sum(a)
  end

  @[AlwaysInline]
  def max(a : Slice(UInt64)) : UInt64
    SIMD::SCALAR_FALLBACK.max(a)
  end

  @[AlwaysInline]
  def min(a : Slice(UInt64)) : UInt64
    SIMD::SCALAR_FALLBACK.min(a)
  end

  @[AlwaysInline]
  def bitwise_and(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.bitwise_and(dst, a, b)
  end

  @[AlwaysInline]
  def bitwise_or(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.bitwise_or(dst, a, b)
  end

  @[AlwaysInline]
  def xor(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.xor(dst, a, b)
  end

  @[AlwaysInline]
  def bswap(dst : Slice(UInt64), a : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.bswap(dst, a)
  end

  @[AlwaysInline]
  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  # ============================================================
  # INT64 OPERATIONS (signed-specific)
  # ============================================================

  @[AlwaysInline]
  def clamp(dst : Slice(Int64), a : Slice(Int64), lo : Int64, hi : Int64) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  @[AlwaysInline]
  def max(a : Slice(Int64)) : Int64
    SIMD::SCALAR_FALLBACK.max(a)
  end

  @[AlwaysInline]
  def min(a : Slice(Int64)) : Int64
    SIMD::SCALAR_FALLBACK.min(a)
  end

  @[AlwaysInline]
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

  @[AlwaysInline]
  def mul(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    # SSE2 pmuludq only produces 64-bit results; scalar is simpler
    SIMD::SCALAR_FALLBACK.mul(dst, a, b)
  end

  @[AlwaysInline]
  def clamp(dst : Slice(UInt32), a : Slice(UInt32), lo : UInt32, hi : UInt32) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  @[AlwaysInline]
  def sum(a : Slice(UInt32)) : UInt32
    SIMD::SCALAR_FALLBACK.sum(a)
  end

  @[AlwaysInline]
  def max(a : Slice(UInt32)) : UInt32
    SIMD::SCALAR_FALLBACK.max(a)
  end

  @[AlwaysInline]
  def min(a : Slice(UInt32)) : UInt32
    SIMD::SCALAR_FALLBACK.min(a)
  end

  @[AlwaysInline]
  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  # ============================================================
  # INT32 OPERATIONS (signed-specific)
  # ============================================================

  @[AlwaysInline]
  def clamp(dst : Slice(Int32), a : Slice(Int32), lo : Int32, hi : Int32) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  @[AlwaysInline]
  def max(a : Slice(Int32)) : Int32
    SIMD::SCALAR_FALLBACK.max(a)
  end

  @[AlwaysInline]
  def min(a : Slice(Int32)) : Int32
    SIMD::SCALAR_FALLBACK.min(a)
  end

  @[AlwaysInline]
  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    # SSE2 cmp_gt_mask is slower than scalar for Int32
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
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

  @[AlwaysInline]
  def clamp(dst : Slice(UInt16), a : Slice(UInt16), lo : UInt16, hi : UInt16) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  @[AlwaysInline]
  def sum(a : Slice(UInt16)) : UInt16
    SIMD::SCALAR_FALLBACK.sum(a)
  end

  @[AlwaysInline]
  def max(a : Slice(UInt16)) : UInt16
    SIMD::SCALAR_FALLBACK.max(a)
  end

  @[AlwaysInline]
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

  @[AlwaysInline]
  def bswap(dst : Slice(UInt16), a : Slice(UInt16)) : Nil
    SIMD::SCALAR_FALLBACK.bswap(dst, a)
  end

  @[AlwaysInline]
  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  # ============================================================
  # INT16 OPERATIONS (signed-specific)
  # ============================================================

  @[AlwaysInline]
  def clamp(dst : Slice(Int16), a : Slice(Int16), lo : Int16, hi : Int16) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  @[AlwaysInline]
  def max(a : Slice(Int16)) : Int16
    # SSE2 max_reduce is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.max(a)
  end

  @[AlwaysInline]
  def min(a : Slice(Int16)) : Int16
    # SSE2 min_reduce is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.min(a)
  end

  @[AlwaysInline]
  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int16), b : Slice(Int16)) : Nil
    # SSE2 cmp_gt_mask is slower than scalar for Int16
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  # ============================================================
  # UINT8 OPERATIONS
  # ============================================================

  VECTOR_WIDTH_8 = 16 # 128-bit / 8-bit = 16

  @[AlwaysInline]
  def add(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    SIMD::SCALAR_FALLBACK.add(dst, a, b)
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

  @[AlwaysInline]
  def mul(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    # SSE2 lacks 8-bit multiply; use scalar
    SIMD::SCALAR_FALLBACK.mul(dst, a, b)
  end

  @[AlwaysInline]
  def clamp(dst : Slice(UInt8), a : Slice(UInt8), lo : UInt8, hi : UInt8) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  @[AlwaysInline]
  def sum(a : Slice(UInt8)) : UInt8
    SIMD::SCALAR_FALLBACK.sum(a)
  end

  @[AlwaysInline]
  def max(a : Slice(UInt8)) : UInt8
    # benchmarks show scalar is faster
    SIMD::SCALAR_FALLBACK.max(a)
  end

  @[AlwaysInline]
  def min(a : Slice(UInt8)) : UInt8
    # benchmarks show scalar is faster
    SIMD::SCALAR_FALLBACK.min(a)
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

  @[AlwaysInline]
  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  @[AlwaysInline]
  def blend(dst : Slice(UInt8), t : Slice(UInt8), f : Slice(UInt8), mask : Slice(UInt8)) : Nil
    SIMD::SCALAR_FALLBACK.blend(dst, t, f, mask)
  end

  # ============================================================
  # INT8 OPERATIONS (signed-specific)
  # ============================================================

  @[AlwaysInline]
  def clamp(dst : Slice(Int8), a : Slice(Int8), lo : Int8, hi : Int8) : Nil
    SIMD::SCALAR_FALLBACK.clamp(dst, a, lo, hi)
  end

  @[AlwaysInline]
  def max(a : Slice(Int8)) : Int8
    SIMD::SCALAR_FALLBACK.max(a)
  end

  @[AlwaysInline]
  def min(a : Slice(Int8)) : Int8
    SIMD::SCALAR_FALLBACK.min(a)
  end

  @[AlwaysInline]
  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  # ============================================================
  # CONVERSION OPERATIONS (new)
  # ============================================================

  @[AlwaysInline]
  def convert(dst : Slice(Float64), src : Slice(Int32)) : Nil
    # SSE2 convert Int32->Float64 is slower than scalar
    SIMD::SCALAR_FALLBACK.convert(dst, src)
  end

  @[AlwaysInline]
  def convert(dst : Slice(Float64), src : Slice(Int16)) : Nil
    SIMD::SCALAR_FALLBACK.convert(dst, src)
  end

  @[AlwaysInline]
  def convert(dst : Slice(Float64), src : Slice(Float32)) : Nil
    # SSE2 convert Float32->Float64 is slower than scalar
    SIMD::SCALAR_FALLBACK.convert(dst, src)
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
