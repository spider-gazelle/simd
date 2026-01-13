# ------------------------------------------------------------
# AVX-512 backend for x86_64
# Processes 16 floats (512-bit) at a time
# Uses zmm registers (zmm0-zmm31) and mask registers (k0-k7)
# ------------------------------------------------------------
class SIMD::AVX512 < SIMD::Base
  VECTOR_WIDTH     = 16 # 512-bit / 32-bit = 16 floats
  VECTOR_WIDTH_F64 =  8 # 512-bit / 64-bit = 8 doubles
  VECTOR_WIDTH_64  =  8 # 512-bit / 64-bit = 8 Int64/UInt64
  VECTOR_WIDTH_16  = 32 # 512-bit / 16-bit = 32 Int16/UInt16
  VECTOR_WIDTH_8   = 64 # 512-bit / 8-bit = 64 Int8/UInt8

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
        "vmovups ($1), %zmm0
         vmovups ($2), %zmm1
         vaddps %zmm1, %zmm0, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovups ($1), %zmm0
         vmovups ($2), %zmm1
         vsubps %zmm1, %zmm0, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovups ($1), %zmm0
         vmovups ($2), %zmm1
         vmulps %zmm1, %zmm0, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
    n = check_len(dst, a, b, c)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    c_ptr = c.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($1), %zmm0
         vmovups ($2), %zmm1
         vmovups ($3), %zmm2
         vfmadd213ps %zmm2, %zmm1, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(c_ptr + i)
              : "zmm0", "zmm1", "zmm2", "memory"
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

    lo_arr = StaticArray(Float32, 16).new(lo)
    hi_arr = StaticArray(Float32, 16).new(hi)
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($1), %zmm0
         vmovups ($2), %zmm1
         vmovups ($3), %zmm2
         vmaxps %zmm1, %zmm0, %zmm0
         vminps %zmm2, %zmm0, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "zmm0", "zmm1", "zmm2", "memory"
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

    alpha_arr = StaticArray(Float32, 16).new(alpha)
    beta_arr = StaticArray(Float32, 16).new(beta)
    alpha_ptr = alpha_arr.to_unsafe
    beta_ptr = beta_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($3), %zmm0
         vmovups ($1), %zmm1
         vmulps %zmm1, %zmm0, %zmm0
         vmovups ($4), %zmm2
         vmovups ($2), %zmm3
         vfmadd231ps %zmm3, %zmm2, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(alpha_ptr), "r"(beta_ptr)
              : "zmm0", "zmm1", "zmm2", "zmm3", "memory"
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

    acc = StaticArray(Float32, 16).new(0.0_f32)
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($0), %zmm0
         vmovups ($1), %zmm1
         vaddps %zmm1, %zmm0, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = 0.0_f32
    (0...16).each { |j| result += acc[j] }

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

    acc = StaticArray(Float32, 16).new(0.0_f32)
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($0), %zmm0
         vmovups ($1), %zmm1
         vmovups ($2), %zmm2
         vfmadd231ps %zmm2, %zmm1, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "zmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = 0.0_f32
    (0...16).each { |j| result += acc[j] }

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

    max_vec = StaticArray(Float32, 16).new { |j| j < n ? a[j] : a[0] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($0), %zmm0
         vmovups ($1), %zmm1
         vmaxps %zmm1, %zmm0, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = max_vec[0]
    (1...16).each { |j| result = max_vec[j] if max_vec[j] > result }

    while i < n
      result = a[i] if a[i] > result
      i += 1
    end

    result
  end

  def min(a : Slice(Float32)) : Float32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] < m }
      return m
    end

    min_vec = StaticArray(Float32, 16).new { |j| j < n ? a[j] : a[0] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovups ($0), %zmm0
         vmovups ($1), %zmm1
         vminps %zmm1, %zmm0, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = min_vec[0]
    (1...16).each { |j| result = min_vec[j] if min_vec[j] < result }

    while i < n
      result = a[i] if a[i] < result
      i += 1
    end

    result
  end

  # ============================================================
  # Float64 operations
  # ============================================================

  def add(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %zmm0
         vmovupd ($2), %zmm1
         vaddpd %zmm1, %zmm0, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovupd ($1), %zmm0
         vmovupd ($2), %zmm1
         vsubpd %zmm1, %zmm0, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovupd ($1), %zmm0
         vmovupd ($2), %zmm1
         vmulpd %zmm1, %zmm0, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovupd ($1), %zmm0
         vmovupd ($2), %zmm1
         vmovupd ($3), %zmm2
         vfmadd213pd %zmm2, %zmm1, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(c_ptr + i)
              : "zmm0", "zmm1", "zmm2", "memory"
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

    lo_arr = StaticArray(Float64, 8).new(lo)
    hi_arr = StaticArray(Float64, 8).new(hi)
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($1), %zmm0
         vmovupd ($2), %zmm1
         vmovupd ($3), %zmm2
         vmaxpd %zmm1, %zmm0, %zmm0
         vminpd %zmm2, %zmm0, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "zmm0", "zmm1", "zmm2", "memory"
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

    alpha_arr = StaticArray(Float64, 8).new(alpha)
    beta_arr = StaticArray(Float64, 8).new(beta)
    alpha_ptr = alpha_arr.to_unsafe
    beta_ptr = beta_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($3), %zmm0
         vmovupd ($1), %zmm1
         vmulpd %zmm1, %zmm0, %zmm0
         vmovupd ($4), %zmm2
         vmovupd ($2), %zmm3
         vfmadd231pd %zmm3, %zmm2, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(alpha_ptr), "r"(beta_ptr)
              : "zmm0", "zmm1", "zmm2", "zmm3", "memory"
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

    acc = StaticArray(Float64, 8).new(0.0)
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($0), %zmm0
         vmovupd ($1), %zmm1
         vaddpd %zmm1, %zmm0, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = 0.0
    (0...8).each { |j| result += acc[j] }

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

    acc = StaticArray(Float64, 8).new(0.0)
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($0), %zmm0
         vmovupd ($1), %zmm1
         vmovupd ($2), %zmm2
         vfmadd231pd %zmm2, %zmm1, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(acc_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "zmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = 0.0
    (0...8).each { |j| result += acc[j] }

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

    max_vec = StaticArray(Float64, 8).new { |j| j < n ? a[j] : a[0] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_F64
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($0), %zmm0
         vmovupd ($1), %zmm1
         vmaxpd %zmm1, %zmm0, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = max_vec[0]
    (1...8).each { |j| result = max_vec[j] if max_vec[j] > result }

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

    min_vec = StaticArray(Float64, 8).new { |j| j < n ? a[j] : a[0] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_F64
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "vmovupd ($0), %zmm0
         vmovupd ($1), %zmm1
         vminpd %zmm1, %zmm0, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = min_vec[0]
    (1...8).each { |j| result = min_vec[j] if min_vec[j] < result }

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
      mask_val = 0_u32
      asm(
        "vmovupd ($1), %zmm0
         vmovupd ($2), %zmm1
         vcmppd $$14, %zmm1, %zmm0, %k1
         kmovb %k1, $0"
              : "=r"(mask_val)
              : "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "k1"
              : "volatile"
      )

      (0...8).each do |j|
        dst_mask[i + j] = ((mask_val >> j) & 1) != 0 ? 0xFF_u8 : 0x00_u8
      end
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def blend(dst : Slice(Float64), t : Slice(Float64), f : Slice(Float64), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      mask_byte = 0_u32
      (0...8).each do |j|
        mask_byte |= (1_u32 << j) if mask[i + j] != 0
      end

      asm(
        "kmovb $3, %k1
         vmovupd ($1), %zmm0
         vmovupd ($2), %zmm1
         vblendmpd %zmm1, %zmm0, %zmm0 {%k1}
         vmovupd %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(f_ptr + i), "r"(t_ptr + i), "r"(mask_byte)
              : "zmm0", "zmm1", "k1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  def compress(dst : Slice(Float64), src : Slice(Float64), mask : Slice(UInt8)) : Int32
    raise ArgumentError.new("length mismatch") unless src.size == mask.size
    n = src.size
    src_ptr = src.to_unsafe
    dst_ptr = dst.to_unsafe
    outp = 0

    i = 0
    while i + VECTOR_WIDTH_F64 <= n && outp + VECTOR_WIDTH_F64 <= dst.size
      mask_byte = 0_u32
      (0...8).each do |j|
        mask_byte |= (1_u32 << j) if mask[i + j] != 0
      end

      count = mask_byte.popcount

      asm(
        "kmovb $2, %k1
         vmovupd ($1), %zmm0
         vcompresspd %zmm0, ($0) {%k1}"
              :: "r"(dst_ptr + outp), "r"(src_ptr + i), "r"(mask_byte)
              : "zmm0", "k1", "memory"
              : "volatile"
      )

      outp += count
      i += VECTOR_WIDTH_F64
    end

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
    taps = coeff.size
    raise ArgumentError.new("coeff empty") if taps == 0
    raise ArgumentError.new("src too small") if src.size < dst.size + taps - 1

    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= dst.size
      acc = StaticArray(Float64, 8).new(0.0)
      acc_ptr = acc.to_unsafe

      k = 0
      while k < taps
        coeff_val = coeff[k]
        coeff_arr = StaticArray(Float64, 8).new(coeff_val)
        coeff_arr_ptr = coeff_arr.to_unsafe

        asm(
          "vmovupd ($0), %zmm0
           vmovupd ($1), %zmm1
           vmovupd ($2), %zmm2
           vfmadd231pd %zmm2, %zmm1, %zmm0
           vmovupd %zmm0, ($0)"
                :: "r"(acc_ptr), "r"(src_ptr + i + k), "r"(coeff_arr_ptr)
                : "zmm0", "zmm1", "zmm2", "memory"
                : "volatile"
        )
        k += 1
      end

      (0...8).each { |j| dst[i + j] = acc[j] }
      i += VECTOR_WIDTH_F64
    end

    while i < dst.size
      acc = 0.0
      k = 0
      while k < taps
        acc += coeff[k] * src[i + k]
        k += 1
      end
      dst[i] = acc
      i += 1
    end
  end

  # ============================================================
  # UInt64 operations
  # ============================================================

  def add(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "vmovdqu64 ($1), %zmm0
         vmovdqu64 ($2), %zmm1
         vpaddq %zmm1, %zmm0, %zmm0
         vmovdqu64 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu64 ($1), %zmm0
         vmovdqu64 ($2), %zmm1
         vpsubq %zmm1, %zmm0, %zmm0
         vmovdqu64 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "vmovdqu64 ($1), %zmm0
         vmovdqu64 ($2), %zmm1
         vpmullq %zmm1, %zmm0, %zmm0
         vmovdqu64 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst[i] = a[i] &* b[i]
      i += 1
    end
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
        "vmovdqu64 ($1), %zmm0
         vmovdqu64 ($2), %zmm1
         vpandq %zmm1, %zmm0, %zmm0
         vmovdqu64 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu64 ($1), %zmm0
         vmovdqu64 ($2), %zmm1
         vporq %zmm1, %zmm0, %zmm0
         vmovdqu64 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu64 ($1), %zmm0
         vmovdqu64 ($2), %zmm1
         vpxorq %zmm1, %zmm0, %zmm0
         vmovdqu64 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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

    # AVX-512 vpshufb shuffle mask for 64-bit byte swap
    shuffle_mask = StaticArray[
      7_u8, 6_u8, 5_u8, 4_u8, 3_u8, 2_u8, 1_u8, 0_u8,
      15_u8, 14_u8, 13_u8, 12_u8, 11_u8, 10_u8, 9_u8, 8_u8,
      7_u8, 6_u8, 5_u8, 4_u8, 3_u8, 2_u8, 1_u8, 0_u8,
      15_u8, 14_u8, 13_u8, 12_u8, 11_u8, 10_u8, 9_u8, 8_u8,
      7_u8, 6_u8, 5_u8, 4_u8, 3_u8, 2_u8, 1_u8, 0_u8,
      15_u8, 14_u8, 13_u8, 12_u8, 11_u8, 10_u8, 9_u8, 8_u8,
      7_u8, 6_u8, 5_u8, 4_u8, 3_u8, 2_u8, 1_u8, 0_u8,
      15_u8, 14_u8, 13_u8, 12_u8, 11_u8, 10_u8, 9_u8, 8_u8,
    ]
    shuffle_ptr = shuffle_mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "vmovdqu64 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpshufb %zmm1, %zmm0, %zmm0
         vmovdqu64 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(shuffle_ptr)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_64
    end

    while i < n
      v = a[i]
      dst[i] = ((v & 0x00000000000000FF_u64) << 56) |
               ((v & 0x000000000000FF00_u64) << 40) |
               ((v & 0x0000000000FF0000_u64) << 24) |
               ((v & 0x00000000FF000000_u64) << 8) |
               ((v & 0x000000FF00000000_u64) >> 8) |
               ((v & 0x0000FF0000000000_u64) >> 24) |
               ((v & 0x00FF000000000000_u64) >> 40) |
               ((v & 0xFF00000000000000_u64) >> 56)
      i += 1
    end
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    SIMD::SCALAR_FALLBACK.cmp_gt_mask(dst_mask, a, b)
  end

  def blend(dst : Slice(UInt64), t : Slice(UInt64), f : Slice(UInt64), mask : Slice(UInt8)) : Nil
    SIMD::SCALAR_FALLBACK.blend(dst, t, f, mask)
  end

  # ============================================================
  # Int64 operations (signed-specific)
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
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      mask_val = 0_u32
      asm(
        "vmovdqu64 ($1), %zmm0
         vmovdqu64 ($2), %zmm1
         vpcmpgtq %zmm1, %zmm0, %k1
         kmovb %k1, $0"
              : "=r"(mask_val)
              : "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "k1"
              : "volatile"
      )

      (0...8).each do |j|
        dst_mask[i + j] = ((mask_val >> j) & 1) != 0 ? 0xFF_u8 : 0x00_u8
      end
      i += VECTOR_WIDTH_64
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  # ============================================================
  # UInt32 operations
  # ============================================================

  def add(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vpaddd %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vpsubd %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vpmulld %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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

    lo_arr = StaticArray(UInt32, 16).new(lo)
    hi_arr = StaticArray(UInt32, 16).new(hi)
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vmovdqu32 ($3), %zmm2
         vpmaxud %zmm1, %zmm0, %zmm0
         vpminud %zmm2, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "zmm0", "zmm1", "zmm2", "memory"
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

  def sum(a : Slice(UInt32)) : UInt32
    SIMD::SCALAR_FALLBACK.sum(a)
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

    max_vec = StaticArray(UInt32, 16).new { |j| j < n ? a[j] : a[0] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu32 ($0), %zmm0
         vmovdqu32 ($1), %zmm1
         vpmaxud %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = max_vec[0]
    (1...16).each { |j| result = max_vec[j] if max_vec[j] > result }

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

    min_vec = StaticArray(UInt32, 16).new { |j| j < n ? a[j] : a[0] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu32 ($0), %zmm0
         vmovdqu32 ($1), %zmm1
         vpminud %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = min_vec[0]
    (1...16).each { |j| result = min_vec[j] if min_vec[j] < result }

    while i < n
      result = a[i] if a[i] < result
      i += 1
    end

    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      mask_val = 0_u32
      asm(
        "vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vpcmpud $$6, %zmm1, %zmm0, %k1
         kmovw %k1, $0"
              : "=r"(mask_val)
              : "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "k1"
              : "volatile"
      )

      (0...16).each do |j|
        dst_mask[i + j] = ((mask_val >> j) & 1) != 0 ? 0xFF_u8 : 0x00_u8
      end
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def blend(dst : Slice(UInt32), t : Slice(UInt32), f : Slice(UInt32), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      mask_word = 0_u32
      (0...16).each do |j|
        mask_word |= (1_u32 << j) if mask[i + j] != 0
      end

      asm(
        "kmovw $3, %k1
         vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vpblendmd %zmm1, %zmm0, %zmm0 {%k1}
         vmovdqu32 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(f_ptr + i), "r"(t_ptr + i), "r"(mask_word)
              : "zmm0", "zmm1", "k1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  # ============================================================
  # Int32 operations (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int32), a : Slice(Int32), lo : Int32, hi : Int32) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray(Int32, 16).new(lo)
    hi_arr = StaticArray(Int32, 16).new(hi)
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vmovdqu32 ($3), %zmm2
         vpmaxsd %zmm1, %zmm0, %zmm0
         vpminsd %zmm2, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "zmm0", "zmm1", "zmm2", "memory"
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

    max_vec = StaticArray(Int32, 16).new { |j| j < n ? a[j] : a[0] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu32 ($0), %zmm0
         vmovdqu32 ($1), %zmm1
         vpmaxsd %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = max_vec[0]
    (1...16).each { |j| result = max_vec[j] if max_vec[j] > result }

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

    min_vec = StaticArray(Int32, 16).new { |j| j < n ? a[j] : a[0] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu32 ($0), %zmm0
         vmovdqu32 ($1), %zmm1
         vpminsd %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = min_vec[0]
    (1...16).each { |j| result = min_vec[j] if min_vec[j] < result }

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
      mask_val = 0_u32
      asm(
        "vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vpcmpgtd %zmm1, %zmm0, %k1
         kmovw %k1, $0"
              : "=r"(mask_val)
              : "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "k1"
              : "volatile"
      )

      (0...16).each do |j|
        dst_mask[i + j] = ((mask_val >> j) & 1) != 0 ? 0xFF_u8 : 0x00_u8
      end
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def bitwise_and(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vpandd %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vpord %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu32 ($1), %zmm0
         vmovdqu32 ($2), %zmm1
         vpxord %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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

    # AVX-512 vpshufb shuffle mask for byte swap (64 bytes)
    shuffle_mask = StaticArray[
      3_u8, 2_u8, 1_u8, 0_u8, 7_u8, 6_u8, 5_u8, 4_u8,
      11_u8, 10_u8, 9_u8, 8_u8, 15_u8, 14_u8, 13_u8, 12_u8,
      3_u8, 2_u8, 1_u8, 0_u8, 7_u8, 6_u8, 5_u8, 4_u8,
      11_u8, 10_u8, 9_u8, 8_u8, 15_u8, 14_u8, 13_u8, 12_u8,
      3_u8, 2_u8, 1_u8, 0_u8, 7_u8, 6_u8, 5_u8, 4_u8,
      11_u8, 10_u8, 9_u8, 8_u8, 15_u8, 14_u8, 13_u8, 12_u8,
      3_u8, 2_u8, 1_u8, 0_u8, 7_u8, 6_u8, 5_u8, 4_u8,
      11_u8, 10_u8, 9_u8, 8_u8, 15_u8, 14_u8, 13_u8, 12_u8,
    ]
    shuffle_ptr = shuffle_mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "vmovdqu32 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpshufb %zmm1, %zmm0, %zmm0
         vmovdqu32 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(shuffle_ptr)
              : "zmm0", "zmm1", "memory"
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
  # UInt16 operations
  # ============================================================

  def add(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu16 ($1), %zmm0
         vmovdqu16 ($2), %zmm1
         vpaddw %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu16 ($1), %zmm0
         vmovdqu16 ($2), %zmm1
         vpsubw %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu16 ($1), %zmm0
         vmovdqu16 ($2), %zmm1
         vpmullw %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray(UInt16, 32).new(lo)
    hi_arr = StaticArray(UInt16, 32).new(hi)
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu16 ($1), %zmm0
         vmovdqu16 ($2), %zmm1
         vmovdqu16 ($3), %zmm2
         vpmaxuw %zmm1, %zmm0, %zmm0
         vpminuw %zmm2, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "zmm0", "zmm1", "zmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
  end

  def sum(a : Slice(UInt16)) : UInt16
    SIMD::SCALAR_FALLBACK.sum(a)
  end

  def max(a : Slice(UInt16)) : UInt16
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_16
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    max_vec = StaticArray(UInt16, 32).new { |j| j < n ? a[j] : a[0] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_16
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu16 ($0), %zmm0
         vmovdqu16 ($1), %zmm1
         vpmaxuw %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    result = max_vec[0]
    (1...32).each { |j| result = max_vec[j] if max_vec[j] > result }

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

    min_vec = StaticArray(UInt16, 32).new { |j| j < n ? a[j] : a[0] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_16
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu16 ($0), %zmm0
         vmovdqu16 ($1), %zmm1
         vpminuw %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    result = min_vec[0]
    (1...32).each { |j| result = min_vec[j] if min_vec[j] < result }

    while i < n
      result = a[i] if a[i] < result
      i += 1
    end

    result
  end

  def bitwise_and(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu16 ($1), %zmm0
         vmovdqu16 ($2), %zmm1
         vpandq %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu16 ($1), %zmm0
         vmovdqu16 ($2), %zmm1
         vporq %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu16 ($1), %zmm0
         vmovdqu16 ($2), %zmm1
         vpxorq %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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

    # AVX-512 vpshufb shuffle mask for 16-bit byte swap
    shuffle_mask = StaticArray[
      1_u8, 0_u8, 3_u8, 2_u8, 5_u8, 4_u8, 7_u8, 6_u8,
      9_u8, 8_u8, 11_u8, 10_u8, 13_u8, 12_u8, 15_u8, 14_u8,
      1_u8, 0_u8, 3_u8, 2_u8, 5_u8, 4_u8, 7_u8, 6_u8,
      9_u8, 8_u8, 11_u8, 10_u8, 13_u8, 12_u8, 15_u8, 14_u8,
      1_u8, 0_u8, 3_u8, 2_u8, 5_u8, 4_u8, 7_u8, 6_u8,
      9_u8, 8_u8, 11_u8, 10_u8, 13_u8, 12_u8, 15_u8, 14_u8,
      1_u8, 0_u8, 3_u8, 2_u8, 5_u8, 4_u8, 7_u8, 6_u8,
      9_u8, 8_u8, 11_u8, 10_u8, 13_u8, 12_u8, 15_u8, 14_u8,
    ]
    shuffle_ptr = shuffle_mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu16 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpshufb %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(shuffle_ptr)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      v = a[i]
      dst[i] = ((v & 0x00FF_u16) << 8) | ((v & 0xFF00_u16) >> 8)
      i += 1
    end
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      mask_val = 0_u32
      asm(
        "vmovdqu16 ($1), %zmm0
         vmovdqu16 ($2), %zmm1
         vpcmpuw $$6, %zmm1, %zmm0, %k1
         kmovd %k1, $0"
              : "=r"(mask_val)
              : "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "k1"
              : "volatile"
      )

      (0...32).each do |j|
        dst_mask[i + j] = ((mask_val >> j) & 1) != 0 ? 0xFF_u8 : 0x00_u8
      end
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def blend(dst : Slice(UInt16), t : Slice(UInt16), f : Slice(UInt16), mask : Slice(UInt8)) : Nil
    SIMD::SCALAR_FALLBACK.blend(dst, t, f, mask)
  end

  # ============================================================
  # Int16 operations (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int16), a : Slice(Int16), lo : Int16, hi : Int16) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray(Int16, 32).new(lo)
    hi_arr = StaticArray(Int16, 32).new(hi)
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu16 ($1), %zmm0
         vmovdqu16 ($2), %zmm1
         vmovdqu16 ($3), %zmm2
         vpmaxsw %zmm1, %zmm0, %zmm0
         vpminsw %zmm2, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "zmm0", "zmm1", "zmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
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

    max_vec = StaticArray(Int16, 32).new { |j| j < n ? a[j] : a[0] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_16
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu16 ($0), %zmm0
         vmovdqu16 ($1), %zmm1
         vpmaxsw %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    result = max_vec[0]
    (1...32).each { |j| result = max_vec[j] if max_vec[j] > result }

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

    min_vec = StaticArray(Int16, 32).new { |j| j < n ? a[j] : a[0] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_16
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "vmovdqu16 ($0), %zmm0
         vmovdqu16 ($1), %zmm1
         vpminsw %zmm1, %zmm0, %zmm0
         vmovdqu16 %zmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    result = min_vec[0]
    (1...32).each { |j| result = min_vec[j] if min_vec[j] < result }

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
      mask_val = 0_u32
      asm(
        "vmovdqu16 ($1), %zmm0
         vmovdqu16 ($2), %zmm1
         vpcmpgtw %zmm1, %zmm0, %k1
         kmovd %k1, $0"
              : "=r"(mask_val)
              : "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "k1"
              : "volatile"
      )

      (0...32).each do |j|
        dst_mask[i + j] = ((mask_val >> j) & 1) != 0 ? 0xFF_u8 : 0x00_u8
      end
      i += VECTOR_WIDTH_16
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  # ============================================================
  # UInt8 operations
  # ============================================================

  def add(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpaddb %zmm1, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpsubb %zmm1, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
    SIMD::SCALAR_FALLBACK.mul(dst, a, b)
  end

  def clamp(dst : Slice(UInt8), a : Slice(UInt8), lo : UInt8, hi : UInt8) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray(UInt8, 64).new(lo)
    hi_arr = StaticArray(UInt8, 64).new(hi)
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vmovdqu8 ($3), %zmm2
         vpmaxub %zmm1, %zmm0, %zmm0
         vpminub %zmm2, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "zmm0", "zmm1", "zmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
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

    max_vec = StaticArray(UInt8, 64).new { |j| j < n ? a[j] : a[0] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu8 ($0), %zmm0
         vmovdqu8 ($1), %zmm1
         vpmaxub %zmm1, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    result = max_vec[0]
    (1...64).each { |j| result = max_vec[j] if max_vec[j] > result }

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

    min_vec = StaticArray(UInt8, 64).new { |j| j < n ? a[j] : a[0] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu8 ($0), %zmm0
         vmovdqu8 ($1), %zmm1
         vpminub %zmm1, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    result = min_vec[0]
    (1...64).each { |j| result = min_vec[j] if min_vec[j] < result }

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
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpandq %zmm1, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vporq %zmm1, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpxorq %zmm1, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "memory"
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
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      mask_val = 0_u64
      asm(
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpcmpub $$6, %zmm1, %zmm0, %k1
         kmovq %k1, $0"
              : "=r"(mask_val)
              : "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "k1"
              : "volatile"
      )

      (0...64).each do |j|
        dst_mask[i + j] = ((mask_val >> j) & 1) != 0 ? 0xFF_u8 : 0x00_u8
      end
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
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
      # Build 64-bit mask
      mask_qword = 0_u64
      (0...64).each do |j|
        mask_qword |= (1_u64 << j) if mask[i + j] != 0
      end

      asm(
        "kmovq $3, %k1
         vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpblendmb %zmm1, %zmm0, %zmm0 {%k1}
         vmovdqu8 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(f_ptr + i), "r"(t_ptr + i), "r"(mask_qword)
              : "zmm0", "zmm1", "k1", "memory"
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
  # Int8 operations (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int8), a : Slice(Int8), lo : Int8, hi : Int8) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray(Int8, 64).new(lo)
    hi_arr = StaticArray(Int8, 64).new(hi)
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vmovdqu8 ($3), %zmm2
         vpmaxsb %zmm1, %zmm0, %zmm0
         vpminsb %zmm2, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "zmm0", "zmm1", "zmm2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
  end

  def max(a : Slice(Int8)) : Int8
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    a_ptr = a.to_unsafe

    if n < VECTOR_WIDTH_8
      m = a[0]
      (1...n).each { |j| m = a[j] if a[j] > m }
      return m
    end

    max_vec = StaticArray(Int8, 64).new { |j| j < n ? a[j] : a[0] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu8 ($0), %zmm0
         vmovdqu8 ($1), %zmm1
         vpmaxsb %zmm1, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    result = max_vec[0]
    (1...64).each { |j| result = max_vec[j] if max_vec[j] > result }

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

    min_vec = StaticArray(Int8, 64).new { |j| j < n ? a[j] : a[0] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "vmovdqu8 ($0), %zmm0
         vmovdqu8 ($1), %zmm1
         vpminsb %zmm1, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    result = min_vec[0]
    (1...64).each { |j| result = min_vec[j] if min_vec[j] < result }

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
      mask_val = 0_u64
      asm(
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpcmpgtb %zmm1, %zmm0, %k1
         kmovq %k1, $0"
              : "=r"(mask_val)
              : "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "k1"
              : "volatile"
      )

      (0...64).each do |j|
        dst_mask[i + j] = ((mask_val >> j) & 1) != 0 ? 0xFF_u8 : 0x00_u8
      end
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def popcount(a : Slice(UInt8)) : UInt64
    # Scalar popcount is faster even with AVX-512
    SIMD::SCALAR_FALLBACK.popcount(a)
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # AVX-512 vcmpps produces k register mask
      mask_val = 0_u32
      asm(
        "vmovups ($1), %zmm0
         vmovups ($2), %zmm1
         vcmpps $$14, %zmm1, %zmm0, %k1
         kmovw %k1, $0"
              : "=r"(mask_val)
              : "r"(a_ptr + i), "r"(b_ptr + i)
              : "zmm0", "zmm1", "k1"
              : "volatile"
      )

      (0...16).each do |j|
        dst_mask[i + j] = ((mask_val >> j) & 1) != 0 ? 0xFF_u8 : 0x00_u8
      end
      i += VECTOR_WIDTH
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def blend(dst : Slice(Float32), t : Slice(Float32), f : Slice(Float32), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # Build mask word
      mask_word = 0_u32
      (0...16).each do |j|
        mask_word |= (1_u32 << j) if mask[i + j] != 0
      end

      asm(
        "kmovw $3, %k1
         vmovups ($1), %zmm0
         vmovups ($2), %zmm1
         vblendmps %zmm1, %zmm0, %zmm0 {%k1}
         vmovups %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(f_ptr + i), "r"(t_ptr + i), "r"(mask_word)
              : "zmm0", "zmm1", "k1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  def compress(dst : Slice(Float32), src : Slice(Float32), mask : Slice(UInt8)) : Int32
    # AVX-512 has compress store instruction
    raise ArgumentError.new("length mismatch") unless src.size == mask.size
    n = src.size
    src_ptr = src.to_unsafe
    dst_ptr = dst.to_unsafe
    outp = 0

    i = 0
    while i + VECTOR_WIDTH <= n && outp + VECTOR_WIDTH <= dst.size
      mask_word = 0_u32
      (0...16).each do |j|
        mask_word |= (1_u32 << j) if mask[i + j] != 0
      end

      count = mask_word.popcount

      asm(
        "kmovw $2, %k1
         vmovups ($1), %zmm0
         vcompressps %zmm0, ($0) {%k1}"
              :: "r"(dst_ptr + outp), "r"(src_ptr + i), "r"(mask_word)
              : "zmm0", "k1", "memory"
              : "volatile"
      )

      outp += count
      i += VECTOR_WIDTH
    end

    # Scalar tail
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
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + 64 <= n
      asm(
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "zmm0", "memory"
              : "volatile"
      )
      i += 64
    end

    while i < n
      dst[i] = src[i]
      i += 1
    end
  end

  def fill(dst : Slice(UInt8), value : UInt8) : Nil
    # Scalar fill is faster (LLVM optimizes to memset)
    Log.trace { "fill forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.fill(dst, value)
  end

  def convert(dst : Slice(Float32), src : Slice(Int16)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # vpmovsxwd: sign-extend 16 x i16 to 16 x i32
      asm(
        "vmovdqu ($1), %ymm0
         vpmovsxwd %ymm0, %zmm0
         vcvtdq2ps %zmm0, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "zmm0", "ymm0", "memory"
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

    scale_arr = StaticArray(Float32, 16).new(scale)
    scale_ptr = scale_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # vpmovzxbd: zero-extend 16 x u8 to 16 x i32
      asm(
        "vmovdqu ($1), %xmm0
         vpmovzxbd %xmm0, %zmm0
         vcvtdq2ps %zmm0, %zmm0
         vmovups ($2), %zmm1
         vmulps %zmm1, %zmm0, %zmm0
         vmovups %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i), "r"(scale_ptr)
              : "zmm0", "zmm1", "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = src[i].to_f32 * scale
      i += 1
    end
  end

  # Convert Float32 to Float64 (widening)
  def convert(dst : Slice(Float64), src : Slice(Float32)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      # vcvtps2pd: convert 8 x Float32 to 8 x Float64
      asm(
        "vmovups ($1), %ymm0
         vcvtps2pd %ymm0, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "zmm0", "ymm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = src[i].to_f64
      i += 1
    end
  end

  # Convert Float64 to Float32 (narrowing)
  def convert(dst : Slice(Float32), src : Slice(Float64)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      # vcvtpd2ps: convert 8 x Float64 to 8 x Float32
      asm(
        "vmovupd ($1), %zmm0
         vcvtpd2ps %zmm0, %ymm0
         vmovups %ymm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "zmm0", "ymm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = src[i].to_f32
      i += 1
    end
  end

  # Convert Int32 to Float64
  def convert(dst : Slice(Float64), src : Slice(Int32)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      # vcvtdq2pd: convert 8 x Int32 to 8 x Float64
      asm(
        "vmovdqu ($1), %ymm0
         vcvtdq2pd %ymm0, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "zmm0", "ymm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = src[i].to_f64
      i += 1
    end
  end

  # Convert Int16 to Float64
  def convert(dst : Slice(Float64), src : Slice(Int16)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      # vpmovsxwd: sign-extend 8 x i16 to 8 x i32, then vcvtdq2pd
      asm(
        "vmovdqu ($1), %xmm0
         vpmovsxwd %xmm0, %ymm0
         vcvtdq2pd %ymm0, %zmm0
         vmovupd %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "zmm0", "ymm0", "xmm0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = src[i].to_f64
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
      acc = StaticArray(Float32, 16).new(0.0_f32)
      acc_ptr = acc.to_unsafe

      k = 0
      while k < taps
        coeff_val = coeff[k]
        coeff_arr = StaticArray(Float32, 16).new(coeff_val)
        coeff_arr_ptr = coeff_arr.to_unsafe

        asm(
          "vmovups ($0), %zmm0
           vmovups ($1), %zmm1
           vmovups ($2), %zmm2
           vfmadd231ps %zmm2, %zmm1, %zmm0
           vmovups %zmm0, ($0)"
                :: "r"(acc_ptr), "r"(src_ptr + i + k), "r"(coeff_arr_ptr)
                : "zmm0", "zmm1", "zmm2", "memory"
                : "volatile"
        )
        k += 1
      end

      (0...16).each { |j| dst[i + j] = acc[j] }
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

    # Broadcast key16 to 512 bits (four copies)
    key64 = uninitialized StaticArray(UInt8, 64)
    (0...4).each do |quad|
      (0...16).each { |j| key64[quad * 16 + j] = key16[j] }
    end
    key_ptr = key64.to_unsafe

    i = 0
    while i + 64 <= n
      asm(
        "vmovdqu8 ($1), %zmm0
         vmovdqu8 ($2), %zmm1
         vpxord %zmm1, %zmm0, %zmm0
         vmovdqu8 %zmm0, ($0)"
              :: "r"(dst_ptr + i), "r"(src_ptr + i), "r"(key_ptr)
              : "zmm0", "zmm1", "memory"
              : "volatile"
      )
      i += 64
    end

    while i < n
      dst[i] = src[i] ^ key16[i & 15]
      i += 1
    end
  end
end
