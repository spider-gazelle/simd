require "./sse41"

# ------------------------------------------------------------
# AVX2 backend for x86_64
# Processes 8 floats (256-bit) at a time
# Uses ymm registers (ymm0-ymm15)
# ------------------------------------------------------------
class SIMD::AVX2 < SIMD::SSE41
  VECTOR_WIDTH = 8 # 256-bit / 32-bit = 8 floats

  private def check_len(*slices)
    size = slices[0].as(Slice).size
    slices.each do |slice|
      raise ArgumentError.new("length mismatch") unless slice.as(Slice).size == size
    end
    size
  end

  # ============================================================
  # Additional operations (non-float / generic)
  # ============================================================

  def abs(dst : Slice(Int8), a : Slice(Int8)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vpxor %ymm1, %ymm1, %ymm1
         vmovdqu ($1), %ymm0
         vpcmpgtb %ymm0, %ymm1, %ymm2
         vpxor %ymm2, %ymm0, %ymm0
         vpsubb %ymm2, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      v = a[i]
      dst[i] = v < 0 ? (0_i8 &- v) : v
      i += 1
    end
  end

  def abs(dst : Slice(Int16), a : Slice(Int16)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vpxor %ymm1, %ymm1, %ymm1
         vmovdqu ($1), %ymm0
         vpcmpgtw %ymm0, %ymm1, %ymm2
         vpxor %ymm2, %ymm0, %ymm0
         vpsubw %ymm2, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      v = a[i]
      dst[i] = v < 0 ? (0_i16 &- v) : v
      i += 1
    end
  end

  def abs(dst : Slice(Int32), a : Slice(Int32)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vpxor %ymm1, %ymm1, %ymm1
         vmovdqu ($1), %ymm0
         vpcmpgtd %ymm0, %ymm1, %ymm2
         vpxor %ymm2, %ymm0, %ymm0
         vpsubd %ymm2, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      v = a[i]
      dst[i] = v < 0 ? (0_i32 &- v) : v
      i += 1
    end
  end

  def neg(dst : Slice(Int8), a : Slice(Int8)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vpxor %ymm0, %ymm0, %ymm0
         vmovdqu ($1), %ymm1
         vpsubb %ymm1, %ymm0, %ymm1
         vmovdqu %ymm1, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst[i] = 0_i8 &- a[i]
      i += 1
    end
  end

  def neg(dst : Slice(Int16), a : Slice(Int16)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vpxor %ymm0, %ymm0, %ymm0
         vmovdqu ($1), %ymm1
         vpsubw %ymm1, %ymm0, %ymm1
         vmovdqu %ymm1, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst[i] = 0_i16 &- a[i]
      i += 1
    end
  end

  def neg(dst : Slice(Int32), a : Slice(Int32)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vpxor %ymm0, %ymm0, %ymm0
         vmovdqu ($1), %ymm1
         vpsubd %ymm1, %ymm0, %ymm1
         vmovdqu %ymm1, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = 0_i32 &- a[i]
      i += 1
    end
  end

  def neg(dst : Slice(Int64), a : Slice(Int64)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "vpxor %ymm0, %ymm0, %ymm0
         vmovdqu ($1), %ymm1
         vpsubq %ymm1, %ymm0, %ymm1
         vmovdqu %ymm1, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst[i] = 0_i64 &- a[i]
      i += 1
    end
  end

  def min(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtb %ymm1, %ymm0, %ymm2
         vpcmpeqb %ymm3, %ymm3, %ymm3
         vpxor %ymm2, %ymm3, %ymm3
         vpand %ymm3, %ymm0, %ymm4
         vpand %ymm2, %ymm1, %ymm5
         vpor %ymm5, %ymm4, %ymm4
         vmovdqu %ymm4, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
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
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtb %ymm1, %ymm0, %ymm2
         vpcmpeqb %ymm3, %ymm3, %ymm3
         vpxor %ymm2, %ymm3, %ymm3
         vpand %ymm2, %ymm0, %ymm4
         vpand %ymm3, %ymm1, %ymm5
         vpor %ymm5, %ymm4, %ymm4
         vmovdqu %ymm4, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
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
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtw %ymm1, %ymm0, %ymm2
         vpcmpeqb %ymm3, %ymm3, %ymm3
         vpxor %ymm2, %ymm3, %ymm3
         vpand %ymm3, %ymm0, %ymm4
         vpand %ymm2, %ymm1, %ymm5
         vpor %ymm5, %ymm4, %ymm4
         vmovdqu %ymm4, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
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
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtw %ymm1, %ymm0, %ymm2
         vpcmpeqb %ymm3, %ymm3, %ymm3
         vpxor %ymm2, %ymm3, %ymm3
         vpand %ymm2, %ymm0, %ymm4
         vpand %ymm3, %ymm1, %ymm5
         vpor %ymm5, %ymm4, %ymm4
         vmovdqu %ymm4, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
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
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtd %ymm1, %ymm0, %ymm2
         vpcmpeqb %ymm3, %ymm3, %ymm3
         vpxor %ymm2, %ymm3, %ymm3
         vpand %ymm3, %ymm0, %ymm4
         vpand %ymm2, %ymm1, %ymm5
         vpor %ymm5, %ymm4, %ymm4
         vmovdqu %ymm4, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "memory"
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

  def max(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtd %ymm1, %ymm0, %ymm2
         vpcmpeqb %ymm3, %ymm3, %ymm3
         vpxor %ymm2, %ymm3, %ymm3
         vpand %ymm2, %ymm0, %ymm4
         vpand %ymm3, %ymm1, %ymm5
         vpor %ymm5, %ymm4, %ymm4
         vmovdqu %ymm4, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "memory"
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

  def cmp_eq(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst_mask, a, b)
    dst_ptr = dst_mask.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpeqb %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
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
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpeqb %ymm1, %ymm0, %ymm0
         vpcmpeqb %ymm2, %ymm2, %ymm2
         vpxor %ymm2, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
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
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtb %ymm0, %ymm1, %ymm1
         vmovdqu %ymm1, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
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
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtb %ymm1, %ymm0, %ymm0
         vpcmpeqb %ymm2, %ymm2, %ymm2
         vpxor %ymm2, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst_mask[i] = a[i] <= b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    cmp_le(dst_mask, b, a)
  end

  def cmp_ne(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      tmp = uninitialized StaticArray(Int32, 8)
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpeqd %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      (0...8).each { |j| dst_mask[i + j] = tmp[j] == 0 ? 0xFF_u8 : 0x00_u8 }
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] != b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_le(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      tmp = uninitialized StaticArray(Int32, 8)
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtd %ymm1, %ymm0, %ymm0
         vpcmpeqd %ymm2, %ymm2, %ymm2
         vpxor %ymm2, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      (0...8).each { |j| dst_mask[i + j] = tmp[j] != 0 ? 0xFF_u8 : 0x00_u8 }
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] <= b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    cmp_le(dst_mask, b, a)
  end

  def add(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vaddps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

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
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vsubps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vmulps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
    # AVX2 with FMA3 support
    n = check_len(dst, a, b, c)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    c_ptr = c.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vmovups ($3), %ymm2
         vfmadd213ps %ymm2, %ymm1, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(c_ptr + i)
              : "ymm0", "ymm1", "ymm2", "memory"
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
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray[lo, lo, lo, lo, lo, lo, lo, lo]
    hi_arr = StaticArray[hi, hi, hi, hi, hi, hi, hi, hi]
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vmovups ($3), %ymm2
         vmaxps %ymm1, %ymm0, %ymm0
         vminps %ymm2, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
  end

  def axpby(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32), alpha : Float32, beta : Float32) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    alpha_arr = StaticArray[alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha]
    beta_arr = StaticArray[beta, beta, beta, beta, beta, beta, beta, beta]
    alpha_ptr = alpha_arr.to_unsafe
    beta_ptr = beta_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # Using FMA: alpha*a + beta*b
      asm(
        "vmovups ($3), %ymm0
         vmovups ($1), %ymm1
         vmulps %ymm1, %ymm0, %ymm0
         vmovups ($4), %ymm2
         vmovups ($2), %ymm3
         vfmadd231ps %ymm3, %ymm2, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(alpha_ptr), "r"(beta_ptr)
              : "ymm0", "ymm1", "ymm2", "ymm3", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = alpha * a[i] + beta * b[i]
      i += 1
    end
  end

  def div(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vdivps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovups ($1), %ymm0
         vsqrtps %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = Math.sqrt(a[i])
      i += 1
    end
  end

  def rsqrt(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    half = StaticArray[0.5_f32, 0.5_f32, 0.5_f32, 0.5_f32, 0.5_f32, 0.5_f32, 0.5_f32, 0.5_f32]
    three_halves = StaticArray[1.5_f32, 1.5_f32, 1.5_f32, 1.5_f32, 1.5_f32, 1.5_f32, 1.5_f32, 1.5_f32]
    half_ptr = half.to_unsafe
    three_halves_ptr = three_halves.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # One Newton-Raphson iteration for accuracy:
      # y = y0 * (1.5 - 0.5*x*y0*y0)
      asm(
        "vmovups ($1), %ymm0
         vrsqrtps %ymm0, %ymm1
         vmulps %ymm1, %ymm1, %ymm2
         vmulps %ymm0, %ymm2, %ymm2
         vmovups ($2), %ymm3
         vmulps %ymm3, %ymm2, %ymm2
         vmovups ($3), %ymm3
         vsubps %ymm2, %ymm3, %ymm3
         vmulps %ymm3, %ymm1, %ymm1
         vmovups %ymm1, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(half_ptr), "r"(three_halves_ptr)
              : "ymm0", "ymm1", "ymm2", "ymm3", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = 1.0_f32 / Math.sqrt(a[i])
      i += 1
    end
  end

  def abs(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    mask = StaticArray[
      0x7fff_ffff_u32, 0x7fff_ffff_u32, 0x7fff_ffff_u32, 0x7fff_ffff_u32,
      0x7fff_ffff_u32, 0x7fff_ffff_u32, 0x7fff_ffff_u32, 0x7fff_ffff_u32,
    ]
    mask_ptr = mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vandps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(mask_ptr)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      u = a[i].unsafe_as(UInt32)
      dst[i] = (u & 0x7fff_ffff_u32).unsafe_as(Float32)
      i += 1
    end
  end

  def neg(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    sign = StaticArray[
      0x8000_0000_u32, 0x8000_0000_u32, 0x8000_0000_u32, 0x8000_0000_u32,
      0x8000_0000_u32, 0x8000_0000_u32, 0x8000_0000_u32, 0x8000_0000_u32,
    ]
    sign_ptr = sign.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vxorps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(sign_ptr)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = -a[i]
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
        "vmovups ($1), %ymm0
         vroundps $$1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "memory"
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
        "vmovups ($1), %ymm0
         vroundps $$2, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "memory"
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
        "vmovups ($1), %ymm0
         vroundps $$0, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i].round
      i += 1
    end
  end

  def min(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vminps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vmaxps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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

  def cmp_eq(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      mask_tmp = uninitialized StaticArray(UInt32, 8)
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vcmpps $$0, %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )

      dst_mask[i] = mask_tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = mask_tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 2] = mask_tmp[2] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 3] = mask_tmp[3] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 4] = mask_tmp[4] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 5] = mask_tmp[5] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 6] = mask_tmp[6] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 7] = mask_tmp[7] != 0 ? 0xFF_u8 : 0x00_u8
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] == b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ne(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      mask_tmp = uninitialized StaticArray(UInt32, 8)
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vcmpps $$4, %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )

      dst_mask[i] = mask_tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = mask_tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 2] = mask_tmp[2] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 3] = mask_tmp[3] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 4] = mask_tmp[4] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 5] = mask_tmp[5] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 6] = mask_tmp[6] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 7] = mask_tmp[7] != 0 ? 0xFF_u8 : 0x00_u8
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] != b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_lt(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      mask_tmp = uninitialized StaticArray(UInt32, 8)
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vcmpps $$1, %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )

      dst_mask[i] = mask_tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = mask_tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 2] = mask_tmp[2] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 3] = mask_tmp[3] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 4] = mask_tmp[4] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 5] = mask_tmp[5] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 6] = mask_tmp[6] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 7] = mask_tmp[7] != 0 ? 0xFF_u8 : 0x00_u8
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] < b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def sum(a : Slice(Float32)) : Float32
    n = a.size
    a_ptr = a.to_unsafe

    acc = StaticArray[0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32]
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($0), %ymm0
         vmovups ($1), %ymm1
         vaddps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    # Horizontal sum
    result = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7]

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

    acc = StaticArray[0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32]
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # Using FMA for dot product: acc += a * b
      asm(
        "vmovups ($0), %ymm0
         vmovups ($1), %ymm1
         vmovups ($2), %ymm2
         vfmadd231ps %ymm2, %ymm1, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7]

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

    max_vec = StaticArray(Float32, 8).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($0), %ymm0
         vmovups ($1), %ymm1
         vmaxps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = max_vec[0]
    (1...8).each { |j| result = max_vec[j] if max_vec[j] > result }

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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpand %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpor %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpxor %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    # AVX2 vpshufb shuffle mask for byte swap
    shuffle_mask = StaticArray[
      3_u8, 2_u8, 1_u8, 0_u8,
      7_u8, 6_u8, 5_u8, 4_u8,
      11_u8, 10_u8, 9_u8, 8_u8,
      15_u8, 14_u8, 13_u8, 12_u8,
      3_u8, 2_u8, 1_u8, 0_u8,
      7_u8, 6_u8, 5_u8, 4_u8,
      11_u8, 10_u8, 9_u8, 8_u8,
      15_u8, 14_u8, 13_u8, 12_u8,
    ]
    shuffle_ptr = shuffle_mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpshufb %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(shuffle_ptr)
              : "ymm0", "ymm1", "memory"
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

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      mask_tmp = uninitialized StaticArray(UInt32, 8)
      # vcmpps with predicate 14 = _CMP_GT_OQ (greater than, ordered, quiet)
      asm(
        "vmovups ($1), %ymm0
         vmovups ($2), %ymm1
         vcmpps $$14, %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )

      dst_mask[i] = mask_tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = mask_tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 2] = mask_tmp[2] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 3] = mask_tmp[3] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 4] = mask_tmp[4] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 5] = mask_tmp[5] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 6] = mask_tmp[6] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 7] = mask_tmp[7] != 0 ? 0xFF_u8 : 0x00_u8
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def copy(dst : Slice(UInt8), src : Slice(UInt8)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + 32 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "ymm0", "memory"
              : "volatile"
      )
      i += 32
    end

    while i < n
      dst[i] = src[i]
      i += 1
    end
  end

  def convert(dst : Slice(Float32), src : Slice(Int16)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # vpmovsxwd: sign-extend 8 x i16 to 8 x i32
      asm(
        "vmovdqu ($1), %xmm0
         vpmovsxwd %xmm0, %ymm0
         vcvtdq2ps %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "ymm0", "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = src[i].to_f32
      i += 1
    end
  end

  def convert(dst : Slice(Float32), src : Slice(UInt8), scale : Float32) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    scale_arr = StaticArray[scale, scale, scale, scale, scale, scale, scale, scale]
    scale_ptr = scale_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # vpmovzxbd: zero-extend 8 x u8 to 8 x i32
      asm(
        "vmovq ($1), %xmm0
         vpmovzxbd %xmm0, %ymm0
         vcvtdq2ps %ymm0, %ymm0
         vmovups ($2), %ymm1
         vmulps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i), "r"(scale_ptr)
              : "ymm0", "ymm1", "xmm0", "memory"
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
    taps = coeff.size
    raise ArgumentError.new("coeff empty") if taps == 0
    raise ArgumentError.new("src too small") if src.size < dst.size + taps - 1

    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= dst.size
      acc = StaticArray[0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32]
      acc_ptr = acc.to_unsafe

      k = 0
      while k < taps
        coeff_val = coeff[k]
        coeff_arr = StaticArray[coeff_val, coeff_val, coeff_val, coeff_val, coeff_val, coeff_val, coeff_val, coeff_val]
        coeff_arr_ptr = coeff_arr.to_unsafe

        asm(
          "vmovups ($0), %ymm0
           vmovups ($1), %ymm1
           vmovups ($2), %ymm2
           vfmadd231ps %ymm2, %ymm1, %ymm0
           vmovups %ymm0, ($0)"
                :: "r"(acc_ptr), "r"(src_ptr + i + k), "r"(coeff_arr_ptr)
                : "ymm0", "ymm1", "ymm2", "memory"
                : "volatile"
        )
        k += 1
      end

      (0...8).each { |j| dst[i + j] = acc[j] }
      i += VECTOR_WIDTH
    end

    while i < dst.size
      acc = 0.0_f32
      k = 0
      while k < taps
        acc += coeff[k] * src[i + k]
        k += 1
      end
      dst[i] = acc
      i += 1
    end
  end

  def xor_block16(dst : Slice(UInt8), src : Slice(UInt8), key16 : StaticArray(UInt8, 16)) : Nil
    raise ArgumentError.new("length mismatch") unless dst.size == src.size
    n = src.size
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    # Broadcast key16 to 256 bits (two copies)
    key32 = uninitialized StaticArray(UInt8, 32)
    (0...16).each { |j| key32[j] = key16[j] }
    (0...16).each { |j| key32[16 + j] = key16[j] }
    key_ptr = key32.to_unsafe

    i = 0
    while i + 32 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpxor %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i), "r"(key_ptr)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += 32
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

    min_vec = StaticArray(Float32, 8).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($0), %ymm0
         vmovups ($1), %ymm1
         vminps %ymm1, %ymm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
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
  # FLOAT64 OPERATIONS
  # ============================================================

  VECTOR_WIDTH_F64 = 4 # 256-bit / 64-bit = 4 doubles

  def add(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vaddpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vsubpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vmulpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vmovupd ($3), %ymm2
         vfmadd213pd %ymm2, %ymm1, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(c_ptr + i)
              : "ymm0", "ymm1", "ymm2", "memory"
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
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray[lo, lo, lo, lo]
    hi_arr = StaticArray[hi, hi, hi, hi]
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vmovupd ($3), %ymm2
         vmaxpd %ymm1, %ymm0, %ymm0
         vminpd %ymm2, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
  end

  def axpby(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64), alpha : Float64, beta : Float64) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    alpha_arr = StaticArray[alpha, alpha, alpha, alpha]
    beta_arr = StaticArray[beta, beta, beta, beta]
    alpha_ptr = alpha_arr.to_unsafe
    beta_ptr = beta_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($3), %ymm0
         vmovupd ($1), %ymm1
         vmulpd %ymm1, %ymm0, %ymm0
         vmovupd ($4), %ymm2
         vmovupd ($2), %ymm3
         vfmadd231pd %ymm3, %ymm2, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(alpha_ptr), "r"(beta_ptr)
              : "ymm0", "ymm1", "ymm2", "ymm3", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = alpha * a[i] + beta * b[i]
      i += 1
    end
  end

  def div(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vdivpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = a[i] / b[i]
      i += 1
    end
  end

  def sqrt(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %ymm0
         vsqrtpd %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = Math.sqrt(a[i])
      i += 1
    end
  end

  def rsqrt(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    ones = StaticArray[1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64]
    ones_ptr = ones.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %ymm0
         vsqrtpd %ymm0, %ymm0
         vmovupd ($2), %ymm1
         vdivpd %ymm0, %ymm1, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(ones_ptr)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = 1.0_f64 / Math.sqrt(a[i])
      i += 1
    end
  end

  def abs(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    mask = StaticArray[0x7fff_ffff_ffff_ffff_u64, 0x7fff_ffff_ffff_ffff_u64, 0x7fff_ffff_ffff_ffff_u64, 0x7fff_ffff_ffff_ffff_u64]
    mask_ptr = mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vandpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(mask_ptr)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = a[i].abs
      i += 1
    end
  end

  def neg(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    sign = StaticArray[0x8000_0000_0000_0000_u64, 0x8000_0000_0000_0000_u64, 0x8000_0000_0000_0000_u64, 0x8000_0000_0000_0000_u64]
    sign_ptr = sign.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vxorpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(sign_ptr)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = -a[i]
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
        "vmovupd ($1), %ymm0
         vroundpd $$1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "memory"
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
        "vmovupd ($1), %ymm0
         vroundpd $$2, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "memory"
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
        "vmovupd ($1), %ymm0
         vroundpd $$0, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "ymm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = a[i].round
      i += 1
    end
  end

  def min(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vminpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av < bv ? av : bv
      i += 1
    end
  end

  def max(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vmaxpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av > bv ? av : bv
      i += 1
    end
  end

  def cmp_eq(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      mask_tmp = uninitialized StaticArray(UInt64, 4)
      asm(
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vcmppd $$0, %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )

      dst_mask[i] = mask_tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = mask_tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 2] = mask_tmp[2] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 3] = mask_tmp[3] != 0 ? 0xFF_u8 : 0x00_u8
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
      mask_tmp = uninitialized StaticArray(UInt64, 4)
      asm(
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vcmppd $$4, %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )

      dst_mask[i] = mask_tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = mask_tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 2] = mask_tmp[2] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 3] = mask_tmp[3] != 0 ? 0xFF_u8 : 0x00_u8
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst_mask[i] = a[i] != b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def sum(a : Slice(Float64)) : Float64
    n = a.size
    a_ptr = a.to_unsafe

    acc = StaticArray[0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64]
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($0), %ymm0
         vmovupd ($1), %ymm1
         vaddpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = acc[0] + acc[1] + acc[2] + acc[3]

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

    acc = StaticArray[0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64]
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($0), %ymm0
         vmovupd ($1), %ymm1
         vmovupd ($2), %ymm2
         vfmadd231pd %ymm2, %ymm1, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = acc[0] + acc[1] + acc[2] + acc[3]

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
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    max_vec = StaticArray(Float64, 4).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_F64
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($0), %ymm0
         vmovupd ($1), %ymm1
         vmaxpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = max_vec[0]
    (1...4).each { |j| result = max_vec[j] if max_vec[j] > result }

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
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] < m }
      return m
    end

    min_vec = StaticArray(Float64, 4).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_F64
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($0), %ymm0
         vmovupd ($1), %ymm1
         vminpd %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = min_vec[0]
    (1...4).each { |j| result = min_vec[j] if min_vec[j] < result }

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
      mask_tmp = uninitialized StaticArray(UInt64, 4)
      asm(
        "vmovupd ($1), %ymm0
         vmovupd ($2), %ymm1
         vcmppd $$14, %ymm1, %ymm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )

      (0...4).each { |j| dst_mask[i + j] = mask_tmp[j] != 0 ? 0xFF_u8 : 0x00_u8 }
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  # ============================================================
  # UINT64 OPERATIONS
  # ============================================================

  VECTOR_WIDTH_64 = 4 # 256-bit / 64-bit = 4

  def add(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpaddq %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpsubq %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst[i] = a[i] &- b[i]
      i += 1
    end
  end

  def bitwise_and(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpand %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpor %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpxor %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    shuffle_mask = StaticArray[
      7_u8, 6_u8, 5_u8, 4_u8, 3_u8, 2_u8, 1_u8, 0_u8,
      15_u8, 14_u8, 13_u8, 12_u8, 11_u8, 10_u8, 9_u8, 8_u8,
      7_u8, 6_u8, 5_u8, 4_u8, 3_u8, 2_u8, 1_u8, 0_u8,
      15_u8, 14_u8, 13_u8, 12_u8, 11_u8, 10_u8, 9_u8, 8_u8,
    ]
    shuffle_ptr = shuffle_mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpshufb %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(shuffle_ptr)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst[i] = a[i].byte_swap
      i += 1
    end
  end

  # ============================================================
  # INT64 OPERATIONS (signed-specific)
  # ============================================================

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int64), b : Slice(Int64)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      mask_tmp = uninitialized StaticArray(Int64, 4)
      # AVX2 vpcmpgtq: signed 64-bit compare
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtq %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )

      (0...4).each { |j| dst_mask[i + j] = mask_tmp[j] != 0 ? 0xFF_u8 : 0x00_u8 }
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  # ============================================================
  # UINT32 OPERATIONS
  # ============================================================

  def add(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpaddd %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpsubd %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpmulld %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = a[i] &* b[i]
      i += 1
    end
  end

  def clamp(dst : Slice(UInt32), a : Slice(UInt32), lo : UInt32, hi : UInt32) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray[lo, lo, lo, lo, lo, lo, lo, lo]
    hi_arr = StaticArray[hi, hi, hi, hi, hi, hi, hi, hi]
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vmovdqu ($3), %ymm2
         vpmaxud %ymm1, %ymm0, %ymm0
         vpminud %ymm2, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
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

    max_vec = StaticArray(UInt32, 8).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($0), %ymm0
         vmovdqu ($1), %ymm1
         vpmaxud %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = max_vec[0]
    (1...8).each { |j| result = max_vec[j] if max_vec[j] > result }

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

    min_vec = StaticArray(UInt32, 8).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($0), %ymm0
         vmovdqu ($1), %ymm1
         vpminud %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
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
  # INT32 OPERATIONS (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int32), a : Slice(Int32), lo : Int32, hi : Int32) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray[lo, lo, lo, lo, lo, lo, lo, lo]
    hi_arr = StaticArray[hi, hi, hi, hi, hi, hi, hi, hi]
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vmovdqu ($3), %ymm2
         vpmaxsd %ymm1, %ymm0, %ymm0
         vpminsd %ymm2, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "ymm0", "ymm1", "ymm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
  end

  def max(a : Slice(Int32)) : Int32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    max_vec = StaticArray(Int32, 8).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($0), %ymm0
         vmovdqu ($1), %ymm1
         vpmaxsd %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = max_vec[0]
    (1...8).each { |j| result = max_vec[j] if max_vec[j] > result }

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

    min_vec = StaticArray(Int32, 8).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu ($0), %ymm0
         vmovdqu ($1), %ymm1
         vpminsd %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = min_vec[0]
    (1...8).each { |j| result = min_vec[j] if min_vec[j] < result }

    while i < n
      result = a[i] if a[i] < result
      i += 1
    end

    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      mask_tmp = uninitialized StaticArray(Int32, 8)
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtd %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )

      (0...8).each { |j| dst_mask[i + j] = mask_tmp[j] != 0 ? 0xFF_u8 : 0x00_u8 }
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

  VECTOR_WIDTH_16 = 16 # 256-bit / 16-bit = 16

  def add(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpaddw %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpsubw %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpmullw %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst[i] = a[i] &* b[i]
      i += 1
    end
  end

  def bitwise_and(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpand %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpor %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpxor %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    shuffle_mask = StaticArray[
      1_u8, 0_u8, 3_u8, 2_u8, 5_u8, 4_u8, 7_u8, 6_u8,
      9_u8, 8_u8, 11_u8, 10_u8, 13_u8, 12_u8, 15_u8, 14_u8,
      1_u8, 0_u8, 3_u8, 2_u8, 5_u8, 4_u8, 7_u8, 6_u8,
      9_u8, 8_u8, 11_u8, 10_u8, 13_u8, 12_u8, 15_u8, 14_u8,
    ]
    shuffle_ptr = shuffle_mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpshufb %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(shuffle_ptr)
              : "ymm0", "ymm1", "memory"
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
  # UINT8 OPERATIONS
  # ============================================================

  VECTOR_WIDTH_8 = 32 # 256-bit / 8-bit = 32

  def add(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpaddb %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpsubb %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst[i] = a[i] &- b[i]
      i += 1
    end
  end

  def bitwise_and(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpand %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpor %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
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
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpxor %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst[i] = a[i] ^ b[i]
      i += 1
    end
  end

  def blend(dst : Slice(UInt8), t : Slice(UInt8), f : Slice(UInt8), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      mask_expanded = uninitialized StaticArray(UInt8, 32)
      (0...32).each { |j| mask_expanded[j] = mask[i + j] != 0 ? 0xFF_u8 : 0x00_u8 }

      asm(
        "vmovdqu ($2), %ymm0
         vmovdqu ($3), %ymm1
         vmovdqu ($4), %ymm2
         vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(dst_ptr + i), "m"(dst_ptr), "r"(f_ptr + i), "r"(t_ptr + i), "r"(mask_expanded.to_unsafe)
              : "ymm0", "ymm1", "ymm2", "memory"
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
  # INT8 OPERATIONS (signed-specific)
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

    max_vec = StaticArray(Int8, 32).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu ($0), %ymm0
         vmovdqu ($1), %ymm1
         vpmaxsb %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    result = max_vec[0]
    (1...32).each { |j| result = max_vec[j] if max_vec[j] > result }

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

    min_vec = StaticArray(Int8, 32).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu ($0), %ymm0
         vmovdqu ($1), %ymm1
         vpminsb %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    result = min_vec[0]
    (1...32).each { |j| result = min_vec[j] if min_vec[j] < result }

    while i < n
      result = a[i] if a[i] < result
      i += 1
    end

    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      mask_tmp = uninitialized StaticArray(Int8, 32)
      asm(
        "vmovdqu ($1), %ymm0
         vmovdqu ($2), %ymm1
         vpcmpgtb %ymm1, %ymm0, %ymm0
         vmovdqu %ymm0, ($0)"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "ymm0", "ymm1", "memory"
              : "volatile"
      )

      (0...32).each { |j| dst_mask[i + j] = mask_tmp[j] != 0 ? 0xFF_u8 : 0x00_u8 }
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  # ============================================================
  # CONVERSION OPERATIONS
  # ============================================================

  def convert(dst : Slice(Float64), src : Slice(Int32)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovdqu ($1), %xmm0
         vcvtdq2pd %xmm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "ymm0", "xmm0", "memory"
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
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovq ($1), %xmm0
         vpmovsxwd %xmm0, %xmm0
         vcvtdq2pd %xmm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "ymm0", "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = src[i].to_f64
      i += 1
    end
  end

  def convert(dst : Slice(Float64), src : Slice(Float32)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovups ($1), %xmm0
         vcvtps2pd %xmm0, %ymm0
         vmovupd %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "ymm0", "xmm0", "memory"
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
        "vmovupd ($1), %ymm0
         vcvtpd2ps %ymm0, %xmm0
         vmovups %xmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "ymm0", "xmm0", "memory"
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
