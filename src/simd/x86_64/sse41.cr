require "./sse2"

# ------------------------------------------------------------
# SSE4.1 backend for x86_64
# Extends SSE2 with blendvps, dpps, pshufb, and other ops
# ------------------------------------------------------------
class SIMD::SSE41 < SIMD::SSE2
  # Inherit most operations from SSE2, override where SSE4.1 has better instructions

  def min(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pminsb %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av < bv ? av : bv
      i += 1
    end
  end

  def max(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pmaxsb %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av > bv ? av : bv
      i += 1
    end
  end

  def min(dst : Slice(Int16), a : Slice(Int16), b : Slice(Int16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 8 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pminsw %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += 8
    end

    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av < bv ? av : bv
      i += 1
    end
  end

  def max(dst : Slice(Int16), a : Slice(Int16), b : Slice(Int16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 8 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pmaxsw %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += 8
    end

    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av > bv ? av : bv
      i += 1
    end
  end

  def min(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 4 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pminsd %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += 4
    end

    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av < bv ? av : bv
      i += 1
    end
  end

  def max(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + 4 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pmaxsd %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += 4
    end

    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av > bv ? av : bv
      i += 1
    end
  end

  def floor(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         roundps $$1, %xmm0, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i].floor
      i += 1
    end
  end

  def ceil(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         roundps $$2, %xmm0, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i].ceil
      i += 1
    end
  end

  def round(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movups ($1), %xmm0
         roundps $$0, %xmm0, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i].round
      i += 1
    end
  end

  def floor(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($1), %xmm0
         roundpd $$1, %xmm0, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = a[i].floor
      i += 1
    end
  end

  def ceil(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($1), %xmm0
         roundpd $$2, %xmm0, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = a[i].ceil
      i += 1
    end
  end

  def round(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "movupd ($1), %xmm0
         roundpd $$0, %xmm0, %xmm0
         movupd %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = a[i].round
      i += 1
    end
  end

  def bswap(dst : Slice(UInt32), a : Slice(UInt32)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    # pshufb shuffle mask for byte swap within each 32-bit element
    shuffle_mask = StaticArray[
      3_u8, 2_u8, 1_u8, 0_u8,     # element 0
      7_u8, 6_u8, 5_u8, 4_u8,     # element 1
      11_u8, 10_u8, 9_u8, 8_u8,   # element 2
      15_u8, 14_u8, 13_u8, 12_u8, # element 3
    ]
    shuffle_ptr = shuffle_mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pshufb %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(shuffle_ptr)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      v = a[i]
      dst[i] = ((v & 0x000000FF_u32) << 24) |
               ((v & 0x0000FF00_u32) << 8) |
               ((v & 0x00FF0000_u32) >> 8) |
               ((v & 0xFF000000_u32) >> 24)
      i += 1
    end
  end

  # ============================================================
  # UINT32 OPERATIONS (SSE4.1 improvements)
  # ============================================================

  def mul(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # SSE4.1 pmulld: 32-bit integer multiply
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pmulld %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] &* b[i]
      i += 1
    end
  end

  def max(a : Slice(UInt32)) : UInt32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    max_vec = StaticArray(UInt32, 4).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      # SSE4.1 pmaxud: unsigned 32-bit max
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pmaxud %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = max_vec[0]
    result = max_vec[1] if max_vec[1] > result
    result = max_vec[2] if max_vec[2] > result
    result = max_vec[3] if max_vec[3] > result

    while i < n
      result = a[i] if a[i] > result
      i += 1
    end

    result
  end

  def min(a : Slice(UInt32)) : UInt32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] < m }
      return m
    end

    min_vec = StaticArray(UInt32, 4).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      # SSE4.1 pminud: unsigned 32-bit min
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pminud %xmm1, %xmm0
         movdqu %xmm0, ($0)"
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
  # BSWAP for 64-bit (SSE4.1 pshufb)
  # ============================================================

  def bswap(dst : Slice(UInt64), a : Slice(UInt64)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    # pshufb shuffle mask for byte swap within each 64-bit element
    shuffle_mask = StaticArray[
      7_u8, 6_u8, 5_u8, 4_u8, 3_u8, 2_u8, 1_u8, 0_u8,       # element 0
      15_u8, 14_u8, 13_u8, 12_u8, 11_u8, 10_u8, 9_u8, 8_u8, # element 1
    ]
    shuffle_ptr = shuffle_mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pshufb %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(shuffle_ptr)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst[i] = a[i].byte_swap
      i += 1
    end
  end

  def bswap(dst : Slice(UInt16), a : Slice(UInt16)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    # pshufb shuffle mask for byte swap within each 16-bit element
    shuffle_mask = StaticArray[
      1_u8, 0_u8, 3_u8, 2_u8, 5_u8, 4_u8, 7_u8, 6_u8,
      9_u8, 8_u8, 11_u8, 10_u8, 13_u8, 12_u8, 15_u8, 14_u8,
    ]
    shuffle_ptr = shuffle_mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "movdqu ($1), %xmm0
         movdqu ($2), %xmm1
         pshufb %xmm1, %xmm0
         movdqu %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(shuffle_ptr)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst[i] = a[i].byte_swap
      i += 1
    end
  end

  # ============================================================
  # CONVERSION OPERATIONS (SSE4.1 improvements)
  # ============================================================

  def convert(dst : Slice(Float64), src : Slice(Int16)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      # SSE4.1 pmovsxwd then cvtdq2pd
      asm(
        "movd ($1), %xmm0
         pmovsxwd %xmm0, %xmm0
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
end
