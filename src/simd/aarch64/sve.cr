# ------------------------------------------------------------
# SVE (Scalable Vector Extension) backend for AArch64
# SVE has variable vector lengths (128-2048 bits in 128-bit increments)
# Uses z0-z31 vector registers and p0-p15 predicate registers
# ------------------------------------------------------------
class SIMD::SVE < SIMD::NEON
  # SVE vector length is implementation-defined (128-2048 bits)
  # We query it at runtime using cntw (count words = 32-bit elements)
  # SVE vector widths for different element sizes (computed at runtime)
  @vector_width : Int32    # 32-bit elements
  @vector_width_64 : Int32 # 64-bit elements
  @vector_width_16 : Int32 # 16-bit elements
  @vector_width_8 : Int32  # 8-bit elements

  def initialize
    @vector_width = get_vector_width
    @vector_width_64 = get_vector_width_64
    @vector_width_16 = get_vector_width_16
    @vector_width_8 = get_vector_width_8
  end

  private def get_vector_width : Int32
    # cntw returns the number of 32-bit elements in a vector
    count = 0_i32
    asm(
      "cntw $0"
            : "=r"(count)
    )
    count
  end

  private def get_vector_width_64 : Int32
    # cntd returns the number of 64-bit elements in a vector
    count = 0_i32
    asm(
      "cntd $0"
            : "=r"(count)
    )
    count
  end

  private def get_vector_width_16 : Int32
    # cnth returns the number of 16-bit elements in a vector
    count = 0_i32
    asm(
      "cnth $0"
            : "=r"(count)
    )
    count
  end

  private def get_vector_width_8 : Int32
    # cntb returns the number of 8-bit elements in a vector
    count = 0_i32
    asm(
      "cntb $0"
            : "=r"(count)
    )
    count
  end

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

  def abs(dst : Slice(UInt8), a : Slice(UInt8)) : Nil
    Log.trace { "abs(UInt8) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  def abs(dst : Slice(Int8), a : Slice(Int8)) : Nil
    Log.trace { "abs(Int8) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  def abs(dst : Slice(UInt16), a : Slice(UInt16)) : Nil
    Log.trace { "abs(UInt16) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  def abs(dst : Slice(Int16), a : Slice(Int16)) : Nil
    Log.trace { "abs(Int16) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  def abs(dst : Slice(UInt32), a : Slice(UInt32)) : Nil
    Log.trace { "abs(UInt32) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  def abs(dst : Slice(Int32), a : Slice(Int32)) : Nil
    Log.trace { "abs(Int32) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  def abs(dst : Slice(UInt64), a : Slice(UInt64)) : Nil
    Log.trace { "abs(UInt64) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  def abs(dst : Slice(Int64), a : Slice(Int64)) : Nil
    Log.trace { "abs(Int64) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.abs(dst, a)
  end

  def neg(dst : Slice(Int8), a : Slice(Int8)) : Nil
    Log.trace { "neg(Int8) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.neg(dst, a)
  end

  def neg(dst : Slice(Int16), a : Slice(Int16)) : Nil
    Log.trace { "neg(Int16) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.neg(dst, a)
  end

  def neg(dst : Slice(Int32), a : Slice(Int32)) : Nil
    Log.trace { "neg(Int32) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.neg(dst, a)
  end

  def neg(dst : Slice(Int64), a : Slice(Int64)) : Nil
    Log.trace { "neg(Int64) forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.neg(dst, a)
  end

  def min(dst : Slice(T), a : Slice(T), b : Slice(T)) : Nil forall T
    Log.trace { "min forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.min(dst, a, b)
  end

  def max(dst : Slice(T), a : Slice(T), b : Slice(T)) : Nil forall T
    Log.trace { "max forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.max(dst, a, b)
  end

  def cmp_eq(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    Log.trace { "cmp_eq forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.cmp_eq(dst_mask, a, b)
  end

  def cmp_ne(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    Log.trace { "cmp_ne forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.cmp_ne(dst_mask, a, b)
  end

  def cmp_lt(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    Log.trace { "cmp_lt forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.cmp_lt(dst_mask, a, b)
  end

  def cmp_le(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    Log.trace { "cmp_le forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.cmp_le(dst_mask, a, b)
  end

  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    Log.trace { "cmp_ge forwarding to scalar (faster)" }
    SIMD::SCALAR_FALLBACK.cmp_ge(dst_mask, a, b)
  end

  def add(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width

    i = 0
    while i + vw <= n
      # ptrue p0.s sets all 32-bit lanes to true
      # ld1w loads 32-bit elements
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fadd z0.s, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    # Scalar tail with predicated load for remaining elements
    remaining = n - i
    if remaining > 0
      # whilelt generates predicate for remaining elements
      asm(
        "mov x9, $4
         whilelt p0.s, xzr, x9
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fadd z0.s, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "m"(dst_ptr), "r"(remaining.to_i64)
              : "z0", "z1", "p0", "x9", "memory"
              : "volatile"
      )
    end
  end

  def sub(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fsub z0.s, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    remaining = n - i
    if remaining > 0
      asm(
        "mov x9, $4
         whilelt p0.s, xzr, x9
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fsub z0.s, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "m"(dst_ptr), "r"(remaining.to_i64)
              : "z0", "z1", "p0", "x9", "memory"
              : "volatile"
      )
    end
  end

  def mul(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fmul z0.s, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    remaining = n - i
    if remaining > 0
      asm(
        "mov x9, $4
         whilelt p0.s, xzr, x9
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fmul z0.s, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "m"(dst_ptr), "r"(remaining.to_i64)
              : "z0", "z1", "p0", "x9", "memory"
              : "volatile"
      )
    end
  end

  def fma(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32), c : Slice(Float32)) : Nil
    n = check_len(dst, a, b, c)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    c_ptr = c.to_unsafe
    vw = @vector_width

    i = 0
    while i + vw <= n
      # SVE fmla: z2 = z0 * z1 + z2
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         ld1w {z2.s}, p0/z, [$3]
         fmla z2.s, p0/m, z0.s, z1.s
         st1w {z2.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(c_ptr + i)
              : "z0", "z1", "z2", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      # SVE fmax/fmin with immediate isn't available; use dup to broadcast
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         fdup z1, $2
         fdup z2, $3
         fmax z0.s, p0/m, z0.s, z1.s
         fmin z0.s, p0/m, z0.s, z2.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "w"(lo), "w"(hi)
              : "z0", "z1", "z2", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fdup z2, $3
         fdup z3, $4
         fmul z0.s, z0.s, z2.s
         fmla z0.s, p0/m, z1.s, z3.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "w"(alpha), "w"(beta)
              : "z0", "z1", "z2", "z3", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fdiv z0.s, p0/m, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         fsqrt z0.s, p0/m, z0.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    one = 1.0_f32

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         fsqrt z0.s, p0/m, z0.s
         fdup z1, $2
         fdiv z0.s, p0/m, z1.s, z0.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "w"(one)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         fabs z0.s, p0/m, z0.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         fneg z0.s, p0/m, z0.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         frintm z0.s, p0/m, z0.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         frintp z0.s, p0/m, z0.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         frintn z0.s, p0/m, z0.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fmin z0.s, p0/m, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fmax z0.s, p0/m, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt32, 64).new(0_u32)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fcmeq p1.s, p0/z, z0.s, z1.s
         mov z2.s, p0/z, #0
         mov z2.s, p1/m, #-1
         st1w {z2.s}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt32, 64).new(0_u32)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fcmne p1.s, p0/z, z0.s, z1.s
         mov z2.s, p0/z, #0
         mov z2.s, p1/m, #-1
         st1w {z2.s}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt32, 64).new(0_u32)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fcmlt p1.s, p0/z, z0.s, z1.s
         mov z2.s, p0/z, #0
         mov z2.s, p1/m, #-1
         st1w {z2.s}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
    end

    while i < n
      dst_mask[i] = a[i] < b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_le(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt32, 64).new(0_u32)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fcmle p1.s, p0/z, z0.s, z1.s
         mov z2.s, p0/z, #0
         mov z2.s, p1/m, #-1
         st1w {z2.s}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
    end

    while i < n
      dst_mask[i] = a[i] <= b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt32, 64).new(0_u32)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fcmge p1.s, p0/z, z0.s, z1.s
         mov z2.s, p0/z, #0
         mov z2.s, p1/m, #-1
         st1w {z2.s}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
    end

    while i < n
      dst_mask[i] = a[i] >= b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def sum(a : Slice(Float32)) : Float32
    n = a.size
    a_ptr = a.to_unsafe
    vw = @vector_width

    result = 0.0_f32
    result_ptr = pointerof(result)

    # Initialize accumulator to zero
    i = 0
    if n >= vw
      # First vector starts the accumulation
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$0]"
              :: "r"(a_ptr)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i = vw

      # Add remaining full vectors
      while i + vw <= n
        asm(
          "ptrue p0.s
           ld1w {z1.s}, p0/z, [$0]
           fadd z0.s, z0.s, z1.s"
                :: "r"(a_ptr + i)
                : "z0", "z1", "p0", "memory"
                : "volatile"
        )
        i += vw
      end

      # Horizontal reduction
      asm(
        "ptrue p0.s
         faddv s0, p0, z0.s
         str s0, [$0]"
              :: "r"(result_ptr)
              : "z0", "p0", "s0", "memory"
              : "volatile"
      )
    end

    # Scalar tail
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
    vw = @vector_width

    result = 0.0_f32
    result_ptr = pointerof(result)

    i = 0
    if n >= vw
      # Initialize accumulator with first vector product
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$0]
         ld1w {z1.s}, p0/z, [$1]
         fmul z2.s, z0.s, z1.s"
              :: "r"(a_ptr), "r"(b_ptr)
              : "z0", "z1", "z2", "p0", "memory"
              : "volatile"
      )
      i = vw

      while i + vw <= n
        asm(
          "ptrue p0.s
           ld1w {z0.s}, p0/z, [$0]
           ld1w {z1.s}, p0/z, [$1]
           fmla z2.s, p0/m, z0.s, z1.s"
                :: "r"(a_ptr + i), "r"(b_ptr + i)
                : "z0", "z1", "z2", "p0", "memory"
                : "volatile"
        )
        i += vw
      end

      asm(
        "ptrue p0.s
         faddv s0, p0, z2.s
         str s0, [$0]"
              :: "r"(result_ptr)
              : "z2", "p0", "s0", "memory"
              : "volatile"
      )
    end

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
    vw = @vector_width

    result = a[0]
    result_ptr = pointerof(result)

    i = 0
    if n >= vw
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$0]"
              :: "r"(a_ptr)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i = vw

      while i + vw <= n
        asm(
          "ptrue p0.s
           ld1w {z1.s}, p0/z, [$0]
           fmax z0.s, p0/m, z0.s, z1.s"
                :: "r"(a_ptr + i)
                : "z0", "z1", "p0", "memory"
                : "volatile"
        )
        i += vw
      end

      asm(
        "ptrue p0.s
         fmaxv s0, p0, z0.s
         str s0, [$0]"
              :: "r"(result_ptr)
              : "z0", "p0", "s0", "memory"
              : "volatile"
      )
    end

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
    vw = @vector_width

    result = a[0]
    result_ptr = pointerof(result)

    i = 0
    if n >= vw
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$0]"
              :: "r"(a_ptr)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i = vw

      while i + vw <= n
        asm(
          "ptrue p0.s
           ld1w {z1.s}, p0/z, [$0]
           fmin z0.s, p0/m, z0.s, z1.s"
                :: "r"(a_ptr + i)
                : "z0", "z1", "p0", "memory"
                : "volatile"
        )
        i += vw
      end

      asm(
        "ptrue p0.s
         fminv s0, p0, z0.s
         str s0, [$0]"
              :: "r"(result_ptr)
              : "z0", "p0", "s0", "memory"
              : "volatile"
      )
    end

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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fadd z0.d, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fsub z0.d, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fmul z0.d, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         ld1d {z2.d}, p0/z, [$3]
         fmla z2.d, p0/m, z0.d, z1.d
         st1d {z2.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "r"(c_ptr + i)
              : "z0", "z1", "z2", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         fdup z1, $2
         fdup z2, $3
         fmax z0.d, p0/m, z0.d, z1.d
         fmin z0.d, p0/m, z0.d, z2.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "w"(lo), "w"(hi)
              : "z0", "z1", "z2", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fdup z2, $3
         fdup z3, $4
         fmul z0.d, z0.d, z2.d
         fmla z0.d, p0/m, z1.d, z3.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i), "w"(alpha), "w"(beta)
              : "z0", "z1", "z2", "z3", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fdiv z0.d, p0/m, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         fsqrt z0.d, p0/m, z0.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    one = 1.0_f64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         fsqrt z0.d, p0/m, z0.d
         fdup z1, $2
         fdiv z0.d, p0/m, z1.d, z0.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "w"(one)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         fabs z0.d, p0/m, z0.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         fneg z0.d, p0/m, z0.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         frintm z0.d, p0/m, z0.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         frintp z0.d, p0/m, z0.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         frintn z0.d, p0/m, z0.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fmin z0.d, p0/m, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fmax z0.d, p0/m, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt64, 32).new(0_u64)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fcmeq p1.d, p0/z, z0.d, z1.d
         mov z2.d, p0/z, #0
         mov z2.d, p1/m, #-1
         st1d {z2.d}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt64, 32).new(0_u64)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fcmne p1.d, p0/z, z0.d, z1.d
         mov z2.d, p0/z, #0
         mov z2.d, p1/m, #-1
         st1d {z2.d}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
    end

    while i < n
      dst_mask[i] = a[i] != b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_lt(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt64, 32).new(0_u64)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fcmlt p1.d, p0/z, z0.d, z1.d
         mov z2.d, p0/z, #0
         mov z2.d, p1/m, #-1
         st1d {z2.d}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
    end

    while i < n
      dst_mask[i] = a[i] < b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_le(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt64, 32).new(0_u64)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fcmle p1.d, p0/z, z0.d, z1.d
         mov z2.d, p0/z, #0
         mov z2.d, p1/m, #-1
         st1d {z2.d}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
    end

    while i < n
      dst_mask[i] = a[i] <= b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt64, 32).new(0_u64)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fcmge p1.d, p0/z, z0.d, z1.d
         mov z2.d, p0/z, #0
         mov z2.d, p1/m, #-1
         st1d {z2.d}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
    end

    while i < n
      dst_mask[i] = a[i] >= b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def sum(a : Slice(Float64)) : Float64
    n = a.size
    a_ptr = a.to_unsafe
    vw = @vector_width_64

    result = 0.0_f64
    result_ptr = pointerof(result)

    i = 0
    if n >= vw
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$0]"
              :: "r"(a_ptr)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i = vw

      while i + vw <= n
        asm(
          "ptrue p0.d
           ld1d {z1.d}, p0/z, [$0]
           fadd z0.d, z0.d, z1.d"
                :: "r"(a_ptr + i)
                : "z0", "z1", "p0", "memory"
                : "volatile"
        )
        i += vw
      end

      asm(
        "ptrue p0.d
         faddv d0, p0, z0.d
         str d0, [$0]"
              :: "r"(result_ptr)
              : "z0", "p0", "d0", "memory"
              : "volatile"
      )
    end

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
    vw = @vector_width_64

    result = 0.0_f64
    result_ptr = pointerof(result)

    i = 0
    if n >= vw
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$0]
         ld1d {z1.d}, p0/z, [$1]
         fmul z2.d, z0.d, z1.d"
              :: "r"(a_ptr), "r"(b_ptr)
              : "z0", "z1", "z2", "p0", "memory"
              : "volatile"
      )
      i = vw

      while i + vw <= n
        asm(
          "ptrue p0.d
           ld1d {z0.d}, p0/z, [$0]
           ld1d {z1.d}, p0/z, [$1]
           fmla z2.d, p0/m, z0.d, z1.d"
                :: "r"(a_ptr + i), "r"(b_ptr + i)
                : "z0", "z1", "z2", "p0", "memory"
                : "volatile"
        )
        i += vw
      end

      asm(
        "ptrue p0.d
         faddv d0, p0, z2.d
         str d0, [$0]"
              :: "r"(result_ptr)
              : "z2", "p0", "d0", "memory"
              : "volatile"
      )
    end

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
    vw = @vector_width_64

    result = a[0]
    result_ptr = pointerof(result)

    i = 0
    if n >= vw
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$0]"
              :: "r"(a_ptr)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i = vw

      while i + vw <= n
        asm(
          "ptrue p0.d
           ld1d {z1.d}, p0/z, [$0]
           fmax z0.d, p0/m, z0.d, z1.d"
                :: "r"(a_ptr + i)
                : "z0", "z1", "p0", "memory"
                : "volatile"
        )
        i += vw
      end

      asm(
        "ptrue p0.d
         fmaxv d0, p0, z0.d
         str d0, [$0]"
              :: "r"(result_ptr)
              : "z0", "p0", "d0", "memory"
              : "volatile"
      )
    end

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
    vw = @vector_width_64

    result = a[0]
    result_ptr = pointerof(result)

    i = 0
    if n >= vw
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$0]"
              :: "r"(a_ptr)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i = vw

      while i + vw <= n
        asm(
          "ptrue p0.d
           ld1d {z1.d}, p0/z, [$0]
           fmin z0.d, p0/m, z0.d, z1.d"
                :: "r"(a_ptr + i)
                : "z0", "z1", "p0", "memory"
                : "volatile"
        )
        i += vw
      end

      asm(
        "ptrue p0.d
         fminv d0, p0, z0.d
         str d0, [$0]"
              :: "r"(result_ptr)
              : "z0", "p0", "d0", "memory"
              : "volatile"
      )
    end

    while i < n
      result = a[i] if a[i] < result
      i += 1
    end

    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         add z0.d, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         sub z0.d, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         mul z0.d, p0/m, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    while i < n
      dst[i] = a[i] &* b[i]
      i += 1
    end
  end

  def clamp(dst : Slice(UInt64), a : Slice(UInt64), lo : UInt64, hi : UInt64) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
  end

  def sum(a : Slice(UInt64)) : UInt64
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = 0_u64
    i = 0
    while i < n
      result &+= a[i]
      i += 1
    end
    result
  end

  def max(a : Slice(UInt64)) : UInt64
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] > result
      i += 1
    end
    result
  end

  def min(a : Slice(UInt64)) : UInt64
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] < result
      i += 1
    end
    result
  end

  def bitwise_and(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         and z0.d, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         orr z0.d, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         eor z0.d, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         revb z0.d, p0/m, z0.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int64), b : Slice(Int64)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def clamp(dst : Slice(Int64), a : Slice(Int64), lo : Int64, hi : Int64) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
  end

  def max(a : Slice(Int64)) : Int64
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] > result
      i += 1
    end
    result
  end

  def min(a : Slice(Int64)) : Int64
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] < result
      i += 1
    end
    result
  end

  def bitwise_and(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         and z0.d, z0.d, z1.d
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         orr z0.d, z0.d, z1.d
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         eor z0.d, z0.d, z1.d
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      # SVE revb reverses bytes within each element
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         revb z0.s, p0/m, z0.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
  # UInt32 operations
  # ============================================================

  def add(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    dst_ptr = dst.to_unsafe
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         add z0.s, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         sub z0.s, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         mul z0.s, p0/m, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    while i < n
      dst[i] = a[i] &* b[i]
      i += 1
    end
  end

  def clamp(dst : Slice(UInt32), a : Slice(UInt32), lo : UInt32, hi : UInt32) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
  end

  def sum(a : Slice(UInt32)) : UInt32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = 0_u32
    i = 0
    while i < n
      result &+= a[i]
      i += 1
    end
    result
  end

  def max(a : Slice(UInt32)) : UInt32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] > result
      i += 1
    end
    result
  end

  def min(a : Slice(UInt32)) : UInt32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] < result
      i += 1
    end
    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  # ============================================================
  # Int32 operations (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int32), a : Slice(Int32), lo : Int32, hi : Int32) : Nil
    n = check_len(dst, a)
    i = 0
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
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] > result
      i += 1
    end
    result
  end

  def min(a : Slice(Int32)) : Int32
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] < result
      i += 1
    end
    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
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
    vw = @vector_width_16

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.h
         ld1h {z0.h}, p0/z, [$1]
         ld1h {z1.h}, p0/z, [$2]
         add z0.h, z0.h, z1.h
         st1h {z0.h}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_16

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.h
         ld1h {z0.h}, p0/z, [$1]
         ld1h {z1.h}, p0/z, [$2]
         sub z0.h, z0.h, z1.h
         st1h {z0.h}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_16

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.h
         ld1h {z0.h}, p0/z, [$1]
         ld1h {z1.h}, p0/z, [$2]
         mul z0.h, p0/m, z0.h, z1.h
         st1h {z0.h}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    while i < n
      dst[i] = a[i] &* b[i]
      i += 1
    end
  end

  def clamp(dst : Slice(UInt16), a : Slice(UInt16), lo : UInt16, hi : UInt16) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
  end

  def sum(a : Slice(UInt16)) : UInt16
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = 0_u16
    i = 0
    while i < n
      result &+= a[i]
      i += 1
    end
    result
  end

  def max(a : Slice(UInt16)) : UInt16
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] > result
      i += 1
    end
    result
  end

  def min(a : Slice(UInt16)) : UInt16
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] < result
      i += 1
    end
    result
  end

  def sum(a : Slice(UInt16)) : UInt16
    n = a.size
    return 0_u16 if n == 0
    result = 0_u16
    i = 0
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
    vw = @vector_width_16

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.h
         ld1h {z0.h}, p0/z, [$1]
         ld1h {z1.h}, p0/z, [$2]
         and z0.d, z0.d, z1.d
         st1h {z0.h}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_16

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.h
         ld1h {z0.h}, p0/z, [$1]
         ld1h {z1.h}, p0/z, [$2]
         orr z0.d, z0.d, z1.d
         st1h {z0.h}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_16

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.h
         ld1h {z0.h}, p0/z, [$1]
         ld1h {z1.h}, p0/z, [$2]
         eor z0.d, z0.d, z1.d
         st1h {z0.h}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_16

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.h
         ld1h {z0.h}, p0/z, [$1]
         revb z0.h, p0/m, z0.h
         st1h {z0.h}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    while i < n
      v = a[i]
      dst[i] = ((v & 0x00FF_u16) << 8) | ((v & 0xFF00_u16) >> 8)
      i += 1
    end
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
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
    i = 0
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
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] > result
      i += 1
    end
    result
  end

  def min(a : Slice(Int16)) : Int16
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] < result
      i += 1
    end
    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int16), b : Slice(Int16)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
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
    vw = @vector_width_8

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.b
         ld1b {z0.b}, p0/z, [$1]
         ld1b {z1.b}, p0/z, [$2]
         add z0.b, z0.b, z1.b
         st1b {z0.b}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_8

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.b
         ld1b {z0.b}, p0/z, [$1]
         ld1b {z1.b}, p0/z, [$2]
         sub z0.b, z0.b, z1.b
         st1b {z0.b}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_8

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.b
         ld1b {z0.b}, p0/z, [$1]
         ld1b {z1.b}, p0/z, [$2]
         mul z0.b, p0/m, z0.b, z1.b
         st1b {z0.b}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    while i < n
      dst[i] = a[i] &* b[i]
      i += 1
    end
  end

  def clamp(dst : Slice(UInt8), a : Slice(UInt8), lo : UInt8, hi : UInt8) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      v = a[i]
      v = lo if v < lo
      v = hi if v > hi
      dst[i] = v
      i += 1
    end
  end

  def sum(a : Slice(UInt8)) : UInt8
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = 0_u8
    i = 0
    while i < n
      result &+= a[i]
      i += 1
    end
    result
  end

  def max(a : Slice(UInt8)) : UInt8
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] > result
      i += 1
    end
    result
  end

  def min(a : Slice(UInt8)) : UInt8
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] < result
      i += 1
    end
    result
  end

  def sum(a : Slice(UInt8)) : UInt8
    n = a.size
    return 0_u8 if n == 0
    result = 0_u8
    i = 0
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
    vw = @vector_width_8

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.b
         ld1b {z0.b}, p0/z, [$1]
         ld1b {z1.b}, p0/z, [$2]
         and z0.d, z0.d, z1.d
         st1b {z0.b}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_8

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.b
         ld1b {z0.b}, p0/z, [$1]
         ld1b {z1.b}, p0/z, [$2]
         orr z0.d, z0.d, z1.d
         st1b {z0.b}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_8

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.b
         ld1b {z0.b}, p0/z, [$1]
         ld1b {z1.b}, p0/z, [$2]
         eor z0.d, z0.d, z1.d
         st1b {z0.b}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    while i < n
      dst[i] = a[i] ^ b[i]
      i += 1
    end
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
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
  # Int8 operations (signed-specific)
  # ============================================================

  def clamp(dst : Slice(Int8), a : Slice(Int8), lo : Int8, hi : Int8) : Nil
    n = check_len(dst, a)
    i = 0
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
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] > result
      i += 1
    end
    result
  end

  def min(a : Slice(Int8)) : Int8
    raise ArgumentError.new("empty") if a.empty?
    n = a.size
    result = a[0]
    i = 1
    while i < n
      result = a[i] if a[i] < result
      i += 1
    end
    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def popcount(a : Slice(UInt8)) : UInt64
    n = a.size
    a_ptr = a.to_unsafe

    # Get byte vector width
    vw_bytes = @vector_width * 4 # 4 bytes per 32-bit element

    result = 0_u64

    i = 0
    if n >= vw_bytes
      # SVE cnt counts bits per element
      asm(
        "ptrue p0.b
         ld1b {z0.b}, p0/z, [$0]
         cnt z1.b, p0/m, z0.b"
              :: "r"(a_ptr)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i = vw_bytes

      while i + vw_bytes <= n
        asm(
          "ptrue p0.b
           ld1b {z0.b}, p0/z, [$0]
           cnt z0.b, p0/m, z0.b
           add z1.b, z1.b, z0.b"
                :: "r"(a_ptr + i)
                : "z0", "z1", "p0", "memory"
                : "volatile"
        )
        i += vw_bytes
      end

      # Sum all bytes - use uaddv for unsigned add reduction
      asm(
        "ptrue p0.b
         uaddv d0, p0, z1.b
         fmov $0, d0"
              : "=r"(result)
              :
: "z1", "p0", "d0"
              : "volatile"
      )
    end

    while i < n
      result += a[i].popcount.to_u64
      i += 1
    end

    result
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def copy(dst : Slice(UInt8), src : Slice(UInt8)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe
    vw_bytes = @vector_width * 4

    i = 0
    while i + vw_bytes <= n
      asm(
        "ptrue p0.b
         ld1b {z0.b}, p0/z, [$1]
         st1b {z0.b}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw_bytes
    end

    while i < n
      dst[i] = src[i]
      i += 1
    end
  end

  def fill(dst : Slice(UInt8), value : UInt8) : Nil
    n = dst.size
    dst_ptr = dst.to_unsafe
    vw_bytes = @vector_width * 4

    i = 0
    while i + vw_bytes <= n
      asm(
        "ptrue p0.b
         dup z0.b, $1
         st1b {z0.b}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(value.to_u32)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw_bytes
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      # sunpklo: sign-extend lower half of i16 to i32
      asm(
        "ptrue p0.s
         ld1h {z0.s}, p0/z, [$1]
         scvtf z0.s, p0/m, z0.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.s
         ld1b {z0.s}, p0/z, [$1]
         ucvtf z0.s, p0/m, z0.s
         fdup z1, $2
         fmul z0.s, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i), "w"(scale)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    while i < n
      dst[i] = src[i].to_f32 * scale
      i += 1
    end
  end

  # Float64 conversion functions
  def convert(dst : Slice(Float32), src : Slice(Float64)) : Nil
    n = check_len(dst, src)
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         fcvt z0.s, p0/m, z0.d
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1w {z0.s}, p0/z, [$1]
         fcvt z0.d, p0/m, z0.s
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1sw {z0.d}, p0/z, [$1]
         scvtf z0.d, p0/m, z0.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      asm(
        "ptrue p0.d
         ld1sh {z0.d}, p0/z, [$1]
         scvtf z0.d, p0/m, z0.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
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

    # Fall back to scalar for simplicity with FIR
    SIMD::SCALAR_FALLBACK.fir(dst, src, coeff)
  end

  def fir(dst : Slice(Float64), src : Slice(Float64), coeff : Slice(Float64)) : Nil
    taps = coeff.size
    raise ArgumentError.new("coeff empty") if taps == 0
    raise ArgumentError.new("src too small") if src.size < dst.size + taps - 1

    SIMD::SCALAR_FALLBACK.fir(dst, src, coeff)
  end

  def xor_block16(dst : Slice(UInt8), src : Slice(UInt8), key16 : StaticArray(UInt8, 16)) : Nil
    raise ArgumentError.new("length mismatch") unless dst.size == src.size
    n = src.size

    # For key16 pattern, we need to replicate the 16-byte key
    # This is complex with SVE's variable length, use scalar for simplicity
    i = 0
    while i < n
      dst[i] = src[i] ^ key16[i & 15]
      i += 1
    end
  end
end
