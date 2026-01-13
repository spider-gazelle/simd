# ------------------------------------------------------------
# SVE2 (Scalable Vector Extension 2) backend for AArch64
# Extends SVE with additional instructions for cryptography,
# complex math, and improved gather/scatter operations
# ------------------------------------------------------------
class SIMD::SVE2 < SIMD::SVE
  # SVE2 inherits most operations from SVE
  # Override where SVE2 has better instructions

  def xor_block16(dst : Slice(UInt8), src : Slice(UInt8), key16 : StaticArray(UInt8, 16)) : Nil
    raise ArgumentError.new("length mismatch") unless dst.size == src.size
    n = src.size
    dst_ptr = dst.to_unsafe
    src_ptr = src.to_unsafe
    key_ptr = key16.to_unsafe

    # SVE2 has better support for table lookups with tbl instruction
    # We can use index generation to create the repeating pattern
    i = 0
    while i + 16 <= n
      # Process 16 bytes at a time using the exact key length
      asm(
        "ptrue p0.b, vl16
         ld1b {z0.b}, p0/z, [$1]
         ld1b {z1.b}, p0/z, [$2]
         eor z0.d, z0.d, z1.d
         st1b {z0.b}, p0, [$0]"
              :: "r"(dst_ptr + i), "r"(src_ptr + i), "r"(key_ptr)
              : "z0", "z1", "p0", "memory"
              : "volatile"
      )
      i += 16
    end

    # Scalar tail
    while i < n
      dst[i] = src[i] ^ key16[i & 15]
      i += 1
    end
  end

  def popcount(a : Slice(UInt8)) : UInt64
    # SVE2 has improved popcount support
    n = a.size
    a_ptr = a.to_unsafe
    vw_bytes = @vector_width * 4

    result = 0_u64

    i = 0
    if n >= vw_bytes
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
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width

    i = 0
    while i + vw <= n
      # SVE2 fcmgt and convert predicate to mask
      mask_arr = StaticArray(UInt32, 16).new(0_u32) # Max SVE width is 2048 bits = 64 floats
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$1]
         ld1w {z1.s}, p0/z, [$2]
         fcmgt p1.s, p0/z, z0.s, z1.s
         mov z2.s, p1/z, #-1
         mov z2.s, p1/m, #-1
         st1w {z2.s}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw && i + j < n
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
    end

    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst_mask, a, b)
    a_ptr = a.to_unsafe
    b_ptr = b.to_unsafe
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt64, 8).new(0_u64)
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$1]
         ld1d {z1.d}, p0/z, [$2]
         fcmgt p1.d, p0/z, z0.d, z1.d
         mov z2.d, p1/z, #-1
         mov z2.d, p1/m, #-1
         st1d {z2.d}, p0, [$0]"
              :: "r"(mask_ptr), "r"(a_ptr + i), "r"(b_ptr + i)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )

      j = 0
      while j < vw && i + j < n
        dst_mask[i + j] = mask_arr[j] != 0 ? 0xFF_u8 : 0x00_u8
        j += 1
      end
      i += vw
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
    vw = @vector_width

    i = 0
    while i + vw <= n
      # Build mask vector
      mask_arr = StaticArray(UInt32, 16).new(0_u32)
      j = 0
      while j < vw && i + j < n
        mask_arr[j] = mask[i + j] != 0 ? 0xFFFFFFFF_u32 : 0_u32
        j += 1
      end
      mask_ptr = mask_arr.to_unsafe

      # SVE2 sel instruction for blending
      asm(
        "ptrue p0.s
         ld1w {z0.s}, p0/z, [$2]
         ld1w {z1.s}, p0/z, [$3]
         ld1w {z2.s}, p0/z, [$4]
         cmpne p1.s, p0/z, z2.s, #0
         sel z0.s, p1, z0.s, z1.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i), "m"(dst_ptr), "r"(t_ptr + i), "r"(f_ptr + i), "r"(mask_ptr)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )
      i += vw
    end

    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  def blend(dst : Slice(Float64), t : Slice(Float64), f : Slice(Float64), mask : Slice(UInt8)) : Nil
    n = check_len(dst, t, f, mask)
    dst_ptr = dst.to_unsafe
    t_ptr = t.to_unsafe
    f_ptr = f.to_unsafe
    vw = @vector_width_64

    i = 0
    while i + vw <= n
      mask_arr = StaticArray(UInt64, 8).new(0_u64)
      j = 0
      while j < vw && i + j < n
        mask_arr[j] = mask[i + j] != 0 ? 0xFFFFFFFFFFFFFFFF_u64 : 0_u64
        j += 1
      end
      mask_ptr = mask_arr.to_unsafe

      asm(
        "ptrue p0.d
         ld1d {z0.d}, p0/z, [$2]
         ld1d {z1.d}, p0/z, [$3]
         ld1d {z2.d}, p0/z, [$4]
         cmpne p1.d, p0/z, z2.d, #0
         sel z0.d, p1, z0.d, z1.d
         st1d {z0.d}, p0, [$0]"
              :: "r"(dst_ptr + i), "m"(dst_ptr), "r"(t_ptr + i), "r"(f_ptr + i), "r"(mask_ptr)
              : "z0", "z1", "z2", "p0", "p1", "memory"
              : "volatile"
      )
      i += vw
    end

    while i < n
      dst[i] = mask[i] == 0 ? f[i] : t[i]
      i += 1
    end
  end

  def compress(dst : Slice(Float32), src : Slice(Float32), mask : Slice(UInt8)) : Int32
    # SVE2 has compact instruction
    raise ArgumentError.new("length mismatch") unless src.size == mask.size
    n = src.size
    src_ptr = src.to_unsafe
    dst_ptr = dst.to_unsafe
    vw = @vector_width
    outp = 0

    i = 0
    while i + vw <= n && outp + vw <= dst.size
      # Build predicate from mask
      mask_arr = StaticArray(UInt32, 16).new(0_u32)
      j = 0
      while j < vw && i + j < n
        mask_arr[j] = mask[i + j] != 0 ? 0xFFFFFFFF_u32 : 0_u32
        j += 1
      end
      mask_ptr = mask_arr.to_unsafe

      # Count active elements
      active_count = 0
      j = 0
      while j < vw && i + j < n
        active_count += 1 if mask[i + j] != 0
        j += 1
      end

      if active_count > 0
        asm(
          "ptrue p0.s
           ld1w {z0.s}, p0/z, [$2]
           cmpne p1.s, p0/z, z0.s, #0
           ld1w {z1.s}, p0/z, [$1]
           compact z1.s, p1, z1.s
           st1w {z1.s}, p1, [$0]"
                :: "r"(dst_ptr + outp), "r"(src_ptr + i), "r"(mask_ptr)
                : "z0", "z1", "p0", "p1", "memory"
                : "volatile"
        )
        outp += active_count
      end
      i += vw
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

  def compress(dst : Slice(Float64), src : Slice(Float64), mask : Slice(UInt8)) : Int32
    raise ArgumentError.new("length mismatch") unless src.size == mask.size
    n = src.size
    src_ptr = src.to_unsafe
    dst_ptr = dst.to_unsafe
    vw = @vector_width_64
    outp = 0

    i = 0
    while i + vw <= n && outp + vw <= dst.size
      mask_arr = StaticArray(UInt64, 8).new(0_u64)
      j = 0
      while j < vw && i + j < n
        mask_arr[j] = mask[i + j] != 0 ? 0xFFFFFFFFFFFFFFFF_u64 : 0_u64
        j += 1
      end
      mask_ptr = mask_arr.to_unsafe

      active_count = 0
      j = 0
      while j < vw && i + j < n
        active_count += 1 if mask[i + j] != 0
        j += 1
      end

      if active_count > 0
        asm(
          "ptrue p0.d
           ld1d {z0.d}, p0/z, [$2]
           cmpne p1.d, p0/z, z0.d, #0
           ld1d {z1.d}, p0/z, [$1]
           compact z1.d, p1, z1.d
           st1d {z1.d}, p1, [$0]"
                :: "r"(dst_ptr + outp), "r"(src_ptr + i), "r"(mask_ptr)
                : "z0", "z1", "p0", "p1", "memory"
                : "volatile"
        )
        outp += active_count
      end
      i += vw
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

  def fir(dst : Slice(Float32), src : Slice(Float32), coeff : Slice(Float32)) : Nil
    taps = coeff.size
    raise ArgumentError.new("coeff empty") if taps == 0
    raise ArgumentError.new("src too small") if src.size < dst.size + taps - 1

    src_ptr = src.to_unsafe
    dst_ptr = dst.to_unsafe
    vw = @vector_width

    i = 0
    while i + vw <= dst.size
      # Zero accumulator
      asm(
        "ptrue p0.s
         fmov z0.s, #0"
 :
:
: "z0", "p0"
 : "volatile"
      )

      k = 0
      while k < taps
        coeff_val = coeff[k]
        asm(
          "ptrue p0.s
           ld1w {z1.s}, p0/z, [$0]
           fdup z2, $1
           fmla z0.s, p0/m, z1.s, z2.s"
                :: "r"(src_ptr + i + k), "w"(coeff_val)
                : "z0", "z1", "z2", "p0", "memory"
                : "volatile"
        )
        k += 1
      end

      asm(
        "ptrue p0.s
         st1w {z0.s}, p0, [$0]"
              :: "r"(dst_ptr + i)
              : "z0", "p0", "memory"
              : "volatile"
      )
      i += vw
    end

    # Scalar tail
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
end
