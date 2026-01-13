# ------------------------------------------------------------
# SSE4.1 backend for x86_64
# Extends SSE2 with blendvps, dpps, pshufb, and other ops
# ------------------------------------------------------------
class SIMD::SSE41 < SIMD::SSE2
  # Inherit most operations from SSE2, override where SSE4.1 has better instructions

  def dot(a : Slice(Float32), b : Slice(Float32)) : Float32
    raise ArgumentError.new("length mismatch") unless a.size == b.size
    n = a.size
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    result = 0.0_f32
    result_ptr = pointerof(result)

    i = 0
    while i + VECTOR_WIDTH <= n
      # dpps with imm8 = 0xF1: multiply all 4 pairs, sum to lowest element
      asm(
        "movups ($1), %xmm0
         movups ($2), %xmm1
         dpps $$0xF1, %xmm1, %xmm0
         addss ($0), %xmm0
         movss %xmm0, ($0)"
              :: "r"(result_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      result += a[i] * b[i]
      i += 1
    end

    result
  end

  def blend(dst : Slice(Float32), t : Slice(Float32), f : Slice(Float32), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # Expand byte mask to 32-bit mask for blendvps
      mask_expanded = uninitialized StaticArray(UInt32, 4)
      mask_expanded[0] = mask[i] != 0 ? 0xFFFFFFFF_u32 : 0_u32
      mask_expanded[1] = mask[i + 1] != 0 ? 0xFFFFFFFF_u32 : 0_u32
      mask_expanded[2] = mask[i + 2] != 0 ? 0xFFFFFFFF_u32 : 0_u32
      mask_expanded[3] = mask[i + 3] != 0 ? 0xFFFFFFFF_u32 : 0_u32

      # blendvps uses xmm0 as implicit mask
      asm(
        "movups ($2), %xmm1
         movups ($3), %xmm2
         movdqu ($4), %xmm0
         blendvps %xmm0, %xmm2, %xmm1
         movups %xmm1, ($0)"
              :: "r"(dst_ptr + i), "m"(dst_ptr), "r"(f_ptr + i), "r"(t_ptr + i), "r"(mask_expanded.to_unsafe)
              : "xmm0", "xmm1", "xmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
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

  def convert(dst : Slice(Float32), src : Slice(UInt8), scale : Float32) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    scale_arr = StaticArray[scale, scale, scale, scale]
    scale_ptr = scale_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # SSE4.1 pmovzxbd: zero-extend 4 x u8 to 4 x i32
      asm(
        "movd ($1), %xmm0
         pmovzxbd %xmm0, %xmm0
         cvtdq2ps %xmm0, %xmm0
         movups ($2), %xmm1
         mulps %xmm1, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i), "r"(scale_ptr)
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

  # ============================================================
  # FLOAT64 OPERATIONS (SSE4.1 improvements)
  # ============================================================

  def dot(a : Slice(Float64), b : Slice(Float64)) : Float64
    raise ArgumentError.new("length mismatch") unless a.size == b.size
    n = a.size
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    result = 0.0_f64
    result_ptr = pointerof(result)

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      # dppd with imm8 = 0x31: multiply both pairs, sum to lowest element
      asm(
        "movupd ($1), %xmm0
         movupd ($2), %xmm1
         dppd $$0x31, %xmm1, %xmm0
         addsd ($0), %xmm0
         movsd %xmm0, ($0)"
              :: "r"(result_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "xmm0", "xmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      result += a[i] * b[i]
      i += 1
    end

    result
  end

  def blend(dst : Slice(Float64), t : Slice(Float64), f : Slice(Float64), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      # Expand byte mask to 64-bit mask for blendvpd
      mask_expanded = uninitialized StaticArray(UInt64, 2)
      mask_expanded[0] = mask[i] != 0 ? 0xFFFFFFFFFFFFFFFF_u64 : 0_u64
      mask_expanded[1] = mask[i + 1] != 0 ? 0xFFFFFFFFFFFFFFFF_u64 : 0_u64

      # blendvpd uses xmm0 as implicit mask
      asm(
        "movupd ($2), %xmm1
         movupd ($3), %xmm2
         movdqu ($4), %xmm0
         blendvpd %xmm0, %xmm2, %xmm1
         movupd %xmm1, ($0)"
              :: "r"(dst_ptr + i), "m"(dst_ptr), "r"(f_ptr + i), "r"(t_ptr + i), "r"(mask_expanded.to_unsafe)
              : "xmm0", "xmm1", "xmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
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
  # INT32 OPERATIONS (SSE4.1 improvements)
  # ============================================================

  def max(a : Slice(Int32)) : Int32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    max_vec = StaticArray(Int32, 4).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      # SSE4.1 pmaxsd: signed 32-bit max
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pmaxsd %xmm1, %xmm0
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

  def min(a : Slice(Int32)) : Int32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] < m }
      return m
    end

    min_vec = StaticArray(Int32, 4).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      # SSE4.1 pminsd: signed 32-bit min
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pminsd %xmm1, %xmm0
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
  # UINT16 OPERATIONS (SSE4.1 improvements)
  # ============================================================

  def max(a : Slice(UInt16)) : UInt16
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_16
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    max_vec = StaticArray(UInt16, 8).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_16
    while i + VECTOR_WIDTH_16 <= n
      # SSE4.1 pmaxuw: unsigned 16-bit max
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pmaxuw %xmm1, %xmm0
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

  def min(a : Slice(UInt16)) : UInt16
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_16
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] < m }
      return m
    end

    min_vec = StaticArray(UInt16, 8).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_16
    while i + VECTOR_WIDTH_16 <= n
      # SSE4.1 pminuw: unsigned 16-bit min
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pminuw %xmm1, %xmm0
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

  # ============================================================
  # INT8 OPERATIONS (SSE4.1 improvements)
  # ============================================================

  def max(a : Slice(Int8)) : Int8
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_8
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    max_vec = StaticArray(Int8, 16).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      # SSE4.1 pmaxsb: signed 8-bit max
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pmaxsb %xmm1, %xmm0
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

  def min(a : Slice(Int8)) : Int8
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_8
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] < m }
      return m
    end

    min_vec = StaticArray(Int8, 16).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      # SSE4.1 pminsb: signed 8-bit min
      asm(
        "movdqu ($0), %xmm0
         movdqu ($1), %xmm1
         pminsb %xmm1, %xmm0
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

  def blend(dst : Slice(UInt8), t : Slice(UInt8), f : Slice(UInt8), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      # Expand mask: 0x00 -> 0x00, non-zero -> 0xFF
      mask_expanded = uninitialized StaticArray(UInt8, 16)
      (0...16).each { |j| mask_expanded[j] = mask[i + j] != 0 ? 0xFF_u8 : 0x00_u8 }

      # SSE4.1 pblendvb: variable byte blend
      asm(
        "movdqu ($2), %xmm1
         movdqu ($3), %xmm2
         movdqu ($4), %xmm0
         pblendvb %xmm0, %xmm2, %xmm1
         movdqu %xmm1, ($0)"
              :: "r"(dst_ptr + i), "m"(dst_ptr), "r"(f_ptr + i), "r"(t_ptr + i), "r"(mask_expanded.to_unsafe)
              : "xmm0", "xmm1", "xmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
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

  def convert(dst : Slice(Float32), src : Slice(Int16)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # SSE4.1 pmovsxwd: sign-extend 4 x i16 to 4 x i32
      asm(
        "movq ($1), %xmm0
         pmovsxwd %xmm0, %xmm0
         cvtdq2ps %xmm0, %xmm0
         movups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = src[i].to_f32
      i += 1
    end
  end

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
