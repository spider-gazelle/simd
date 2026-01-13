# ------------------------------------------------------------
# NEON backend for AArch64
# Processes 4 floats (128-bit) at a time using v0-v31 registers
# NEON is baseline on AArch64
# ------------------------------------------------------------
class SIMD::NEON < SIMD::Base
  VECTOR_WIDTH     =  4 # 128-bit / 32-bit = 4 floats
  VECTOR_WIDTH_F64 =  2 # 128-bit / 64-bit = 2 doubles
  VECTOR_WIDTH_64  =  2 # 128-bit / 64-bit = 2 Int64/UInt64
  VECTOR_WIDTH_16  =  8 # 128-bit / 16-bit = 8 Int16/UInt16
  VECTOR_WIDTH_8   = 16 # 128-bit / 8-bit = 16 Int8/UInt8

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
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         fadd v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         fsub v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         fmul v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
    # NEON has FMA instruction
    n = check_len(dst, a, b, c)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    c_ptr = c.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         ld1 {v2.4s}, [$3]
         fmla v2.4s, v0.4s, v1.4s
         st1 {v2.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(c_ptr + i)
              : "v0", "v1", "v2", "memory"
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

    lo_arr = StaticArray[lo, lo, lo, lo]
    hi_arr = StaticArray[hi, hi, hi, hi]
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         ld1 {v2.4s}, [$3]
         fmax v0.4s, v0.4s, v1.4s
         fmin v0.4s, v0.4s, v2.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "v0", "v1", "v2", "memory"
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

    alpha_arr = StaticArray[alpha, alpha, alpha, alpha]
    beta_arr = StaticArray[beta, beta, beta, beta]
    alpha_ptr = alpha_arr.to_unsafe
    beta_ptr = beta_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         ld1 {v2.4s}, [$3]
         ld1 {v3.4s}, [$4]
         fmul v0.4s, v0.4s, v2.4s
         fmla v0.4s, v1.4s, v3.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(alpha_ptr), "r"(beta_ptr)
              : "v0", "v1", "v2", "v3", "memory"
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
        "ld1 {v0.4s}, [$0]
         ld1 {v1.4s}, [$1]
         fadd v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    # Horizontal sum using faddp
    result = 0.0_f32
    result_ptr = pointerof(result)
    asm(
      "ld1 {v0.4s}, [$1]
       faddp v0.4s, v0.4s, v0.4s
       faddp v0.2s, v0.2s, v0.2s
       st1 {v0.s}[0], [$0]"
            :: "r"(result_ptr), "r"(acc_ptr)
            : "v0", "memory"
            : "volatile"
    )

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
        "ld1 {v0.4s}, [$0]
         ld1 {v1.4s}, [$1]
         ld1 {v2.4s}, [$2]
         fmla v0.4s, v1.4s, v2.4s
         st1 {v0.4s}, [$0]"
              :: "r"(acc_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "v2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = 0.0_f32
    result_ptr = pointerof(result)
    asm(
      "ld1 {v0.4s}, [$1]
       faddp v0.4s, v0.4s, v0.4s
       faddp v0.2s, v0.2s, v0.2s
       st1 {v0.s}[0], [$0]"
            :: "r"(result_ptr), "r"(acc_ptr)
            : "v0", "memory"
            : "volatile"
    )

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

    max_vec = StaticArray(Float32, 4).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "ld1 {v0.4s}, [$0]
         ld1 {v1.4s}, [$1]
         fmax v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    # Horizontal max using fmaxp
    result = 0.0_f32
    result_ptr = pointerof(result)
    asm(
      "ld1 {v0.4s}, [$1]
       fmaxp v0.4s, v0.4s, v0.4s
       fmaxp v0.2s, v0.2s, v0.2s
       st1 {v0.s}[0], [$0]"
            :: "r"(result_ptr), "r"(max_ptr)
            : "v0", "memory"
            : "volatile"
    )

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

    min_vec = StaticArray(Float32, 4).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "ld1 {v0.4s}, [$0]
         ld1 {v1.4s}, [$1]
         fmin v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    # Horizontal min using fminp
    result = 0.0_f32
    result_ptr = pointerof(result)
    asm(
      "ld1 {v0.4s}, [$1]
       fminp v0.4s, v0.4s, v0.4s
       fminp v0.2s, v0.2s, v0.2s
       st1 {v0.s}[0], [$0]"
            :: "r"(result_ptr), "r"(min_ptr)
            : "v0", "memory"
            : "volatile"
    )

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
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         fadd v0.2d, v0.2d, v1.2d
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         fsub v0.2d, v0.2d, v1.2d
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         fmul v0.2d, v0.2d, v1.2d
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         ld1 {v2.2d}, [$3]
         fmla v2.2d, v0.2d, v1.2d
         st1 {v2.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(c_ptr + i)
              : "v0", "v1", "v2", "memory"
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

    lo_arr = StaticArray[lo, lo]
    hi_arr = StaticArray[hi, hi]
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         ld1 {v2.2d}, [$3]
         fmax v0.2d, v0.2d, v1.2d
         fmin v0.2d, v0.2d, v2.2d
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "v0", "v1", "v2", "memory"
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

    alpha_arr = StaticArray[alpha, alpha]
    beta_arr = StaticArray[beta, beta]
    alpha_ptr = alpha_arr.to_unsafe
    beta_ptr = beta_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         ld1 {v2.2d}, [$3]
         ld1 {v3.2d}, [$4]
         fmul v0.2d, v0.2d, v2.2d
         fmla v0.2d, v1.2d, v3.2d
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(alpha_ptr), "r"(beta_ptr)
              : "v0", "v1", "v2", "v3", "memory"
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

    acc = StaticArray[0.0, 0.0]
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "ld1 {v0.2d}, [$0]
         ld1 {v1.2d}, [$1]
         fadd v0.2d, v0.2d, v1.2d
         st1 {v0.2d}, [$0]"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    # Horizontal sum
    result = 0.0
    result_ptr = pointerof(result)
    asm(
      "ld1 {v0.2d}, [$1]
       faddp d0, v0.2d
       str d0, [$0]"
            :: "r"(result_ptr), "r"(acc_ptr)
            : "v0", "memory"
            : "volatile"
    )

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

    acc = StaticArray[0.0, 0.0]
    acc_ptr = acc.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "ld1 {v0.2d}, [$0]
         ld1 {v1.2d}, [$1]
         ld1 {v2.2d}, [$2]
         fmla v0.2d, v1.2d, v2.2d
         st1 {v0.2d}, [$0]"
              :: "r"(acc_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "v2", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = 0.0
    result_ptr = pointerof(result)
    asm(
      "ld1 {v0.2d}, [$1]
       faddp d0, v0.2d
       str d0, [$0]"
            :: "r"(result_ptr), "r"(acc_ptr)
            : "v0", "memory"
            : "volatile"
    )

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

    max_vec = StaticArray(Float64, 2).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_F64
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "ld1 {v0.2d}, [$0]
         ld1 {v1.2d}, [$1]
         fmax v0.2d, v0.2d, v1.2d
         st1 {v0.2d}, [$0]"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = 0.0
    result_ptr = pointerof(result)
    asm(
      "ld1 {v0.2d}, [$1]
       fmaxp d0, v0.2d
       str d0, [$0]"
            :: "r"(result_ptr), "r"(max_ptr)
            : "v0", "memory"
            : "volatile"
    )

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

    min_vec = StaticArray(Float64, 2).new { |j| a[j] }
    min_ptr = min_vec.to_unsafe

    i = VECTOR_WIDTH_F64
    while i + VECTOR_WIDTH_F64 <= n
      asm(
        "ld1 {v0.2d}, [$0]
         ld1 {v1.2d}, [$1]
         fmin v0.2d, v0.2d, v1.2d
         st1 {v0.2d}, [$0]"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    result = 0.0
    result_ptr = pointerof(result)
    asm(
      "ld1 {v0.2d}, [$1]
       fminp d0, v0.2d
       str d0, [$0]"
            :: "r"(result_ptr), "r"(min_ptr)
            : "v0", "memory"
            : "volatile"
    )

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
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         fcmgt v0.2d, v0.2d, v1.2d
         st1 {v0.2d}, [$0]"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      mask_expanded = uninitialized StaticArray(UInt64, 2)
      mask_expanded[0] = mask[i] != 0 ? 0xFFFFFFFFFFFFFFFF_u64 : 0_u64
      mask_expanded[1] = mask[i + 1] != 0 ? 0xFFFFFFFFFFFFFFFF_u64 : 0_u64

      asm(
        "ld1 {v0.2d}, [$3]
         ld1 {v1.2d}, [$1]
         ld1 {v2.2d}, [$2]
         bsl v0.16b, v1.16b, v2.16b
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(t_ptr + i), "r"(f_ptr + i), "r"(mask_expanded.to_unsafe)
              : "v0", "v1", "v2", "memory"
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
    # NEON lacks compress instruction; use scalar
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
    taps = coeff.size
    raise ArgumentError.new("coeff empty") if taps == 0
    raise ArgumentError.new("src too small") if src.size < dst.size + taps - 1

    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= dst.size
      acc = StaticArray[0.0, 0.0]
      acc_ptr = acc.to_unsafe

      k = 0
      while k < taps
        coeff_val = coeff[k]
        coeff_arr = StaticArray[coeff_val, coeff_val]
        coeff_arr_ptr = coeff_arr.to_unsafe

        asm(
          "ld1 {v0.2d}, [$0]
           ld1 {v1.2d}, [$1]
           ld1 {v2.2d}, [$2]
           fmla v0.2d, v1.2d, v2.2d
           st1 {v0.2d}, [$0]"
                :: "r"(acc_ptr), "r"(src_ptr + i + k), "r"(coeff_arr_ptr)
                : "v0", "v1", "v2", "memory"
                : "volatile"
        )
        k += 1
      end

      dst[i] = acc[0]
      dst[i + 1] = acc[1]
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
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         add v0.2d, v0.2d, v1.2d
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         sub v0.2d, v0.2d, v1.2d
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
    # NEON doesn't have native 64-bit multiply
    SIMD::SCALAR_FALLBACK.mul(dst, a, b)
  end

  def bitwise_and(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      asm(
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         and v0.16b, v0.16b, v1.16b
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         orr v0.16b, v0.16b, v1.16b
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         eor v0.16b, v0.16b, v1.16b
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      # rev64 reverses bytes within each 64-bit element
      asm(
        "ld1 {v0.2d}, [$1]
         rev64 v0.16b, v0.16b
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "v0", "memory"
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

  # ============================================================
  # Int64 operations (signed-specific)
  # ============================================================

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int64), b : Slice(Int64)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_64 <= n
      mask_tmp = uninitialized StaticArray(Int64, 2)
      asm(
        "ld1 {v0.2d}, [$1]
         ld1 {v1.2d}, [$2]
         cmgt v0.2d, v0.2d, v1.2d
         st1 {v0.2d}, [$0]"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )

      dst_mask[i] = mask_tmp[0] != 0 ? 0xFF_u8 : 0x00_u8
      dst_mask[i + 1] = mask_tmp[1] != 0 ? 0xFF_u8 : 0x00_u8
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
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         add v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         sub v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         mul v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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

    lo_arr = StaticArray[lo, lo, lo, lo]
    hi_arr = StaticArray[hi, hi, hi, hi]
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         ld1 {v2.4s}, [$3]
         umax v0.4s, v0.4s, v1.4s
         umin v0.4s, v0.4s, v2.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "v0", "v1", "v2", "memory"
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

    max_vec = StaticArray(UInt32, 4).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "ld1 {v0.4s}, [$0]
         ld1 {v1.4s}, [$1]
         umax v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    # Horizontal max
    result = max_vec[0]
    (1...4).each { |j| result = max_vec[j] if max_vec[j] > result }

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
      asm(
        "ld1 {v0.4s}, [$0]
         ld1 {v1.4s}, [$1]
         umin v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = min_vec[0]
    (1...4).each { |j| result = min_vec[j] if min_vec[j] < result }

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
      mask_tmp = uninitialized StaticArray(UInt32, 4)
      asm(
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         cmhi v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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

  def blend(dst : Slice(UInt32), t : Slice(UInt32), f : Slice(UInt32), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      mask_expanded = uninitialized StaticArray(UInt32, 4)
      mask_expanded[0] = mask[i] != 0 ? 0xFFFFFFFF_u32 : 0_u32
      mask_expanded[1] = mask[i + 1] != 0 ? 0xFFFFFFFF_u32 : 0_u32
      mask_expanded[2] = mask[i + 2] != 0 ? 0xFFFFFFFF_u32 : 0_u32
      mask_expanded[3] = mask[i + 3] != 0 ? 0xFFFFFFFF_u32 : 0_u32

      asm(
        "ld1 {v0.4s}, [$3]
         ld1 {v1.4s}, [$1]
         ld1 {v2.4s}, [$2]
         bsl v0.16b, v1.16b, v2.16b
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(t_ptr + i), "r"(f_ptr + i), "r"(mask_expanded.to_unsafe)
              : "v0", "v1", "v2", "memory"
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

    lo_arr = StaticArray[lo, lo, lo, lo]
    hi_arr = StaticArray[hi, hi, hi, hi]
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         ld1 {v2.4s}, [$3]
         smax v0.4s, v0.4s, v1.4s
         smin v0.4s, v0.4s, v2.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "v0", "v1", "v2", "memory"
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

    max_vec = StaticArray(Int32, 4).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH
    while i + VECTOR_WIDTH <= n
      asm(
        "ld1 {v0.4s}, [$0]
         ld1 {v1.4s}, [$1]
         smax v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = max_vec[0]
    (1...4).each { |j| result = max_vec[j] if max_vec[j] > result }

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
      asm(
        "ld1 {v0.4s}, [$0]
         ld1 {v1.4s}, [$1]
         smin v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    result = min_vec[0]
    (1...4).each { |j| result = min_vec[j] if min_vec[j] < result }

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
      mask_tmp = uninitialized StaticArray(Int32, 4)
      asm(
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         cmgt v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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

  def bitwise_and(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      asm(
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         and v0.16b, v0.16b, v1.16b
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         orr v0.16b, v0.16b, v1.16b
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         eor v0.16b, v0.16b, v1.16b
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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

    i = 0
    while i + VECTOR_WIDTH <= n
      # NEON rev32 reverses bytes within each 32-bit element
      asm(
        "ld1 {v0.4s}, [$1]
         rev32 v0.16b, v0.16b
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "v0", "memory"
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
        "ld1 {v0.8h}, [$1]
         ld1 {v1.8h}, [$2]
         add v0.8h, v0.8h, v1.8h
         st1 {v0.8h}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.8h}, [$1]
         ld1 {v1.8h}, [$2]
         sub v0.8h, v0.8h, v1.8h
         st1 {v0.8h}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.8h}, [$1]
         ld1 {v1.8h}, [$2]
         mul v0.8h, v0.8h, v1.8h
         st1 {v0.8h}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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

    lo_arr = StaticArray[lo, lo, lo, lo, lo, lo, lo, lo]
    hi_arr = StaticArray[hi, hi, hi, hi, hi, hi, hi, hi]
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "ld1 {v0.8h}, [$1]
         ld1 {v1.8h}, [$2]
         ld1 {v2.8h}, [$3]
         umax v0.8h, v0.8h, v1.8h
         umin v0.8h, v0.8h, v2.8h
         st1 {v0.8h}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "v0", "v1", "v2", "memory"
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
      asm(
        "ld1 {v0.8h}, [$0]
         ld1 {v1.8h}, [$1]
         umax v0.8h, v0.8h, v1.8h
         st1 {v0.8h}, [$0]"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
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
      asm(
        "ld1 {v0.8h}, [$0]
         ld1 {v1.8h}, [$1]
         umin v0.8h, v0.8h, v1.8h
         st1 {v0.8h}, [$0]"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
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

  def sum(a : Slice(UInt16)) : UInt16
    n = a.size
    return 0_u16 if n == 0
    a_ptr = a.to_unsafe

    acc_vec = StaticArray(UInt16, 8).new(0_u16)
    acc_ptr = acc_vec.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "ld1 {v0.8h}, [$0]
         ld1 {v1.8h}, [$1]
         add v0.8h, v0.8h, v1.8h
         st1 {v0.8h}, [$0]"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_16
    end

    result = 0_u16
    (0...8).each { |j| result = result &+ acc_vec[j] }

    while i < n
      result = result &+ a[i]
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
        "ld1 {v0.8h}, [$1]
         ld1 {v1.8h}, [$2]
         and v0.16b, v0.16b, v1.16b
         st1 {v0.8h}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.8h}, [$1]
         ld1 {v1.8h}, [$2]
         orr v0.16b, v0.16b, v1.16b
         st1 {v0.8h}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.8h}, [$1]
         ld1 {v1.8h}, [$2]
         eor v0.16b, v0.16b, v1.16b
         st1 {v0.8h}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      # rev16 reverses bytes within each 16-bit element
      asm(
        "ld1 {v0.8h}, [$1]
         rev16 v0.16b, v0.16b
         st1 {v0.8h}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "v0", "memory"
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
      mask_tmp = uninitialized StaticArray(UInt16, 8)
      asm(
        "ld1 {v0.8h}, [$1]
         ld1 {v1.8h}, [$2]
         cmhi v0.8h, v0.8h, v1.8h
         st1 {v0.8h}, [$0]"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
  # Int16 operations (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int16), a : Slice(Int16), lo : Int16, hi : Int16) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray[lo, lo, lo, lo, lo, lo, lo, lo]
    hi_arr = StaticArray[hi, hi, hi, hi, hi, hi, hi, hi]
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "ld1 {v0.8h}, [$1]
         ld1 {v1.8h}, [$2]
         ld1 {v2.8h}, [$3]
         smax v0.8h, v0.8h, v1.8h
         smin v0.8h, v0.8h, v2.8h
         st1 {v0.8h}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "v0", "v1", "v2", "memory"
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

    max_vec = StaticArray(Int16, 8).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_16
    while i + VECTOR_WIDTH_16 <= n
      asm(
        "ld1 {v0.8h}, [$0]
         ld1 {v1.8h}, [$1]
         smax v0.8h, v0.8h, v1.8h
         st1 {v0.8h}, [$0]"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.8h}, [$0]
         ld1 {v1.8h}, [$1]
         smin v0.8h, v0.8h, v1.8h
         st1 {v0.8h}, [$0]"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.8h}, [$1]
         ld1 {v1.8h}, [$2]
         cmgt v0.8h, v0.8h, v1.8h
         st1 {v0.8h}, [$0]"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         add v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         sub v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         mul v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst[i] = a[i] &* b[i]
      i += 1
    end
  end

  def clamp(dst : Slice(UInt8), a : Slice(UInt8), lo : UInt8, hi : UInt8) : Nil
    n = check_len(dst, a)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe

    lo_arr = StaticArray(UInt8, 16).new(lo)
    hi_arr = StaticArray(UInt8, 16).new(hi)
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         ld1 {v2.16b}, [$3]
         umax v0.16b, v0.16b, v1.16b
         umin v0.16b, v0.16b, v2.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "v0", "v1", "v2", "memory"
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
        "ld1 {v0.16b}, [$0]
         ld1 {v1.16b}, [$1]
         umax v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.16b}, [$0]
         ld1 {v1.16b}, [$1]
         umin v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
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

  def sum(a : Slice(UInt8)) : UInt8
    n = a.size
    return 0_u8 if n == 0
    a_ptr = a.to_unsafe

    acc_vec = StaticArray(UInt8, 16).new(0_u8)
    acc_ptr = acc_vec.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "ld1 {v0.16b}, [$0]
         ld1 {v1.16b}, [$1]
         add v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    result = 0_u8
    (0...16).each { |j| result = result &+ acc_vec[j] }

    while i < n
      result = result &+ a[i]
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
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         and v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         orr v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         eor v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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
    dst_ptr = dst_mask.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      # cmhi sets all 1s if a > b (unsigned)
      asm(
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         cmhi v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
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
    mask_ptr = mask.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      # bsl: bitwise select (mask ? t : f)
      asm(
        "ld1 {v0.16b}, [$3]
         ld1 {v1.16b}, [$1]
         ld1 {v2.16b}, [$2]
         bsl v0.16b, v1.16b, v2.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(t_ptr + i), "r"(f_ptr + i), "r"(mask_ptr + i)
              : "v0", "v1", "v2", "memory"
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

    lo_arr = StaticArray(Int8, 16).new(lo)
    hi_arr = StaticArray(Int8, 16).new(hi)
    lo_ptr = lo_arr.to_unsafe
    hi_ptr = hi_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         ld1 {v2.16b}, [$3]
         smax v0.16b, v0.16b, v1.16b
         smin v0.16b, v0.16b, v2.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(lo_ptr), "r"(hi_ptr)
              : "v0", "v1", "v2", "memory"
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

    max_vec = StaticArray(Int8, 16).new { |j| a[j] }
    max_ptr = max_vec.to_unsafe

    i = VECTOR_WIDTH_8
    while i + VECTOR_WIDTH_8 <= n
      asm(
        "ld1 {v0.16b}, [$0]
         ld1 {v1.16b}, [$1]
         smax v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(max_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
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
      asm(
        "ld1 {v0.16b}, [$0]
         ld1 {v1.16b}, [$1]
         smin v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(min_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
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

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst_mask, a, b)
    dst_ptr = dst_mask.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_8 <= n
      # cmgt sets all 1s if a > b (signed)
      asm(
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         cmgt v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_8
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def popcount(a : Slice(UInt8)) : UInt64
    n = a.size
    a_ptr = a.to_unsafe

    acc = 0_u64
    acc_arr = StaticArray[0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8,
      0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8]
    acc_ptr = acc_arr.to_unsafe

    i = 0
    while i + 16 <= n
      # NEON cnt counts bits in each byte
      asm(
        "ld1 {v0.16b}, [$0]
         ld1 {v1.16b}, [$1]
         cnt v1.16b, v1.16b
         add v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(acc_ptr), "r"(a_ptr + i)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += 16
    end

    # Sum all bytes in accumulator
    (0...16).each { |j| acc += acc_arr[j].to_u64 }

    # Scalar tail
    while i < n
      acc += a[i].popcount.to_u64
      i += 1
    end

    acc
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      mask_tmp = uninitialized StaticArray(UInt32, 4)
      # fcmgt sets all 1s if greater
      asm(
        "ld1 {v0.4s}, [$1]
         ld1 {v1.4s}, [$2]
         fcmgt v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(mask_tmp.to_unsafe), "r"(a_ptr + i), "r"(b_ptr + i)
              : "v0", "v1", "memory"
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

  def blend(dst : Slice(Float32), t : Slice(Float32), f : Slice(Float32), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # Expand byte mask to 32-bit for bsl
      mask_expanded = uninitialized StaticArray(UInt32, 4)
      mask_expanded[0] = mask[i] != 0 ? 0xFFFFFFFF_u32 : 0_u32
      mask_expanded[1] = mask[i + 1] != 0 ? 0xFFFFFFFF_u32 : 0_u32
      mask_expanded[2] = mask[i + 2] != 0 ? 0xFFFFFFFF_u32 : 0_u32
      mask_expanded[3] = mask[i + 3] != 0 ? 0xFFFFFFFF_u32 : 0_u32

      # bsl: bitwise select (mask ? t : f)
      asm(
        "ld1 {v0.4s}, [$3]
         ld1 {v1.4s}, [$1]
         ld1 {v2.4s}, [$2]
         bsl v0.16b, v1.16b, v2.16b
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(t_ptr + i), "r"(f_ptr + i), "r"(mask_expanded.to_unsafe)
              : "v0", "v1", "v2", "memory"
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
    # NEON lacks compress instruction; use scalar
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
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "ld1 {v0.16b}, [$1]
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "v0", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      dst[i] = src[i]
      i += 1
    end
  end

  def fill(dst : Slice(UInt8), value : UInt8) : Nil
    n = dst.size
    dst_ptr = dst.to_unsafe

    fill_arr = StaticArray(UInt8, 16).new(value)
    fill_ptr = fill_arr.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "ld1 {v0.16b}, [$1]
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(fill_ptr)
              : "v0", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      dst[i] = value
      i += 1
    end
  end

  def convert(dst : Slice(Float32), src : Slice(Int16)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # sxtl: sign-extend 4 x i16 to 4 x i32, then scvtf to float
      asm(
        "ld1 {v0.4h}, [$1]
         sxtl v0.4s, v0.4h
         scvtf v0.4s, v0.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "v0", "memory"
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

    scale_arr = StaticArray[scale, scale, scale, scale]
    scale_ptr = scale_arr.to_unsafe

    i = 0
    while i + VECTOR_WIDTH <= n
      # uxtl: zero-extend u8 to u16, then uxtl again to u32, then ucvtf
      asm(
        "ld1 {v0.s}[0], [$1]
         uxtl v0.8h, v0.8b
         uxtl v0.4s, v0.4h
         ucvtf v0.4s, v0.4s
         ld1 {v1.4s}, [$2]
         fmul v0.4s, v0.4s, v1.4s
         st1 {v0.4s}, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i), "r"(scale_ptr)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH
    end

    while i < n
      dst[i] = src[i].to_f32 * scale
      i += 1
    end
  end

  # Float64 <-> Float32 conversions
  def convert(dst : Slice(Float32), src : Slice(Float64)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      # fcvtn: narrow 2 × f64 to 2 × f32 (low half of v0)
      asm(
        "ld1 {v0.2d}, [$1]
         fcvtn v0.2s, v0.2d
         str d0, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "v0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = src[i].to_f32
      i += 1
    end
  end

  def convert(dst : Slice(Float64), src : Slice(Float32)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      # fcvtl: widen 2 × f32 (low half) to 2 × f64
      asm(
        "ldr d0, [$1]
         fcvtl v0.2d, v0.2s
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "v0", "memory"
              : "volatile"
      )
      i += VECTOR_WIDTH_F64
    end

    while i < n
      dst[i] = src[i].to_f64
      i += 1
    end
  end

  def convert(dst : Slice(Float64), src : Slice(Int32)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe

    i = 0
    while i + VECTOR_WIDTH_F64 <= n
      # sxtl: sign-extend 2 × i32 to 2 × i64, then scvtf to double
      asm(
        "ldr d0, [$1]
         sxtl v0.2d, v0.2s
         scvtf v0.2d, v0.2d
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "v0", "memory"
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
      # Need to widen i16 to i64: sxtl to i32, then sxtl to i64
      asm(
        "ldr s0, [$1]
         sxtl v0.4s, v0.4h
         sxtl v0.2d, v0.2s
         scvtf v0.2d, v0.2d
         st1 {v0.2d}, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "v0", "memory"
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
      acc = StaticArray[0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32]
      acc_ptr = acc.to_unsafe

      k = 0
      while k < taps
        coeff_val = coeff[k]
        coeff_arr = StaticArray[coeff_val, coeff_val, coeff_val, coeff_val]
        coeff_arr_ptr = coeff_arr.to_unsafe

        asm(
          "ld1 {v0.4s}, [$0]
           ld1 {v1.4s}, [$1]
           ld1 {v2.4s}, [$2]
           fmla v0.4s, v1.4s, v2.4s
           st1 {v0.4s}, [$0]"
                :: "r"(acc_ptr), "r"(src_ptr + i + k), "r"(coeff_arr_ptr)
                : "v0", "v1", "v2", "memory"
                : "volatile"
        )
        k += 1
      end

      dst[i] = acc[0]
      dst[i + 1] = acc[1]
      dst[i + 2] = acc[2]
      dst[i + 3] = acc[3]
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
    key_ptr = key16.to_unsafe

    i = 0
    while i + 16 <= n
      asm(
        "ld1 {v0.16b}, [$1]
         ld1 {v1.16b}, [$2]
         eor v0.16b, v0.16b, v1.16b
         st1 {v0.16b}, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i), "r"(key_ptr)
              : "v0", "v1", "memory"
              : "volatile"
      )
      i += 16
    end

    while i < n
      dst[i] = src[i] ^ key16[i & 15]
      i += 1
    end
  end
end
