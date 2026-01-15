# SIMD-oriented primitive interface.
# Backends implement these with:
# - Scalar fallback
# - x86: AVX2 / AVX-512
# - ARM: NEON / SVE / SVE2
#
# The key is: higher-level code calls these primitives, not intrinsics directly.
#
# Type support:
# - Floating point: Float32, Float64
# - Integers: Int64/UInt64, Int32/UInt32, Int16/UInt16, Int8/UInt8
#
# For signed integers, most operations forward to unsigned versions since
# SIMD instructions operate on the same bit patterns.
abstract class SIMD::Base
  # Shared length check helper.
  # Backends often override this locally for their own methods.
  private def check_len(*slices)
    size = slices[0].as(Slice).size
    slices.each do |slice|
      raise ArgumentError.new("length mismatch") unless slice.as(Slice).size == size
    end
    size
  end

  # ============================================================
  # FLOAT32 OPERATIONS (existing)
  # ============================================================

  # ---------------------------
  # Basic vector math (Float32)
  # ---------------------------

  # dst[i] = a[i] + b[i]
  abstract def add(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil

  # dst[i] = a[i] - b[i]
  abstract def sub(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil

  # dst[i] = a[i] * b[i]
  abstract def mul(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil

  # dst[i] = a[i] * b[i] + c[i]   (FMA on capable backends)
  abstract def fma(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32), c : Slice(Float32)) : Nil

  # dst[i] = clamp(a[i], lo, hi)
  abstract def clamp(dst : Slice(Float32), a : Slice(Float32), lo : Float32, hi : Float32) : Nil

  # dst[i] = alpha * a[i] + beta * b[i]
  abstract def axpby(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32), alpha : Float32, beta : Float32) : Nil

  # ---------------------------
  # Additional math (Float32)
  # ---------------------------

  # dst[i] = a[i] / b[i]
  abstract def div(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil

  # dst[i] = sqrt(a[i])
  abstract def sqrt(dst : Slice(Float32), a : Slice(Float32)) : Nil

  # dst[i] = 1 / sqrt(a[i])  (fast approx on capable backends)
  abstract def rsqrt(dst : Slice(Float32), a : Slice(Float32)) : Nil

  # dst[i] = abs(a[i])
  abstract def abs(dst : Slice(Float32), a : Slice(Float32)) : Nil

  # dst[i] = -a[i]
  abstract def neg(dst : Slice(Float32), a : Slice(Float32)) : Nil

  # dst[i] = floor(a[i])
  abstract def floor(dst : Slice(Float32), a : Slice(Float32)) : Nil

  # dst[i] = ceil(a[i])
  abstract def ceil(dst : Slice(Float32), a : Slice(Float32)) : Nil

  # dst[i] = round(a[i])
  abstract def round(dst : Slice(Float32), a : Slice(Float32)) : Nil

  # ---------------------------
  # Reductions (Float32)
  # ---------------------------

  # Returns sum(a[i])
  abstract def sum(a : Slice(Float32)) : Float32

  # Returns dot(a,b) = sum(a[i] * b[i])  (can use FMA)
  abstract def dot(a : Slice(Float32), b : Slice(Float32)) : Float32

  # Returns max(a[i])
  abstract def max(a : Slice(Float32)) : Float32

  # Returns min(a[i])
  abstract def min(a : Slice(Float32)) : Float32

  # ---------------------------
  # Compare / mask (Float32)
  # ---------------------------

  # Writes an output mask byte per element: 0xFF if a[i] > b[i], else 0x00
  abstract def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil

  # ---------------------------
  # Convolution (Float32)
  # ---------------------------

  # 1D FIR: dst[i] = sum_{k=0..taps-1} coeff[k] * src[i+k]
  abstract def fir(dst : Slice(Float32), src : Slice(Float32), coeff : Slice(Float32)) : Nil

  # ============================================================
  # FLOAT64 OPERATIONS
  # ============================================================

  # ---------------------------
  # Basic vector math (Float64)
  # ---------------------------

  abstract def add(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
  abstract def sub(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
  abstract def mul(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
  abstract def fma(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64), c : Slice(Float64)) : Nil
  abstract def clamp(dst : Slice(Float64), a : Slice(Float64), lo : Float64, hi : Float64) : Nil
  abstract def axpby(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64), alpha : Float64, beta : Float64) : Nil

  # ---------------------------
  # Additional math (Float64)
  # ---------------------------

  abstract def div(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
  abstract def sqrt(dst : Slice(Float64), a : Slice(Float64)) : Nil
  abstract def rsqrt(dst : Slice(Float64), a : Slice(Float64)) : Nil
  abstract def abs(dst : Slice(Float64), a : Slice(Float64)) : Nil
  abstract def neg(dst : Slice(Float64), a : Slice(Float64)) : Nil
  abstract def floor(dst : Slice(Float64), a : Slice(Float64)) : Nil
  abstract def ceil(dst : Slice(Float64), a : Slice(Float64)) : Nil
  abstract def round(dst : Slice(Float64), a : Slice(Float64)) : Nil

  # ---------------------------
  # Reductions (Float64)
  # ---------------------------

  abstract def sum(a : Slice(Float64)) : Float64
  abstract def dot(a : Slice(Float64), b : Slice(Float64)) : Float64
  abstract def max(a : Slice(Float64)) : Float64
  abstract def min(a : Slice(Float64)) : Float64

  # ---------------------------
  # Compare / mask (Float64)
  # ---------------------------

  abstract def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil

  # ---------------------------
  # Convolution (Float64)
  # ---------------------------

  abstract def fir(dst : Slice(Float64), src : Slice(Float64), coeff : Slice(Float64)) : Nil

  # ============================================================
  # UINT64 OPERATIONS
  # ============================================================

  # ---------------------------
  # Arithmetic (UInt64)
  # ---------------------------

  abstract def add(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
  abstract def sub(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
  abstract def mul(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
  abstract def clamp(dst : Slice(UInt64), a : Slice(UInt64), lo : UInt64, hi : UInt64) : Nil

  # ---------------------------
  # Reductions (UInt64)
  # ---------------------------

  abstract def sum(a : Slice(UInt64)) : UInt64
  abstract def max(a : Slice(UInt64)) : UInt64
  abstract def min(a : Slice(UInt64)) : UInt64

  # ---------------------------
  # Bitwise (UInt64)
  # ---------------------------

  abstract def bitwise_and(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
  abstract def bitwise_or(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
  abstract def xor(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
  abstract def bswap(dst : Slice(UInt64), a : Slice(UInt64)) : Nil

  # ---------------------------
  # Compare / mask (UInt64)
  # ---------------------------

  abstract def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt64), b : Slice(UInt64)) : Nil

  # ============================================================
  # INT64 OPERATIONS (forward to UInt64 where possible)
  # ============================================================

  # Arithmetic - same bit operations
  def add(dst : Slice(Int64), a : Slice(Int64), b : Slice(Int64)) : Nil
    add(uint64_view(dst), uint64_view(a), uint64_view(b))
  end

  def sub(dst : Slice(Int64), a : Slice(Int64), b : Slice(Int64)) : Nil
    sub(uint64_view(dst), uint64_view(a), uint64_view(b))
  end

  def mul(dst : Slice(Int64), a : Slice(Int64), b : Slice(Int64)) : Nil
    mul(uint64_view(dst), uint64_view(a), uint64_view(b))
  end

  # Clamp needs signed comparison - abstract
  abstract def clamp(dst : Slice(Int64), a : Slice(Int64), lo : Int64, hi : Int64) : Nil

  # Reductions - sum same bits, max/min need signed comparison
  def sum(a : Slice(Int64)) : Int64
    sum(uint64_view(a)).to_i64!
  end

  abstract def max(a : Slice(Int64)) : Int64
  abstract def min(a : Slice(Int64)) : Int64

  # Bitwise - same bit operations
  def bitwise_and(dst : Slice(Int64), a : Slice(Int64), b : Slice(Int64)) : Nil
    bitwise_and(uint64_view(dst), uint64_view(a), uint64_view(b))
  end

  def bitwise_or(dst : Slice(Int64), a : Slice(Int64), b : Slice(Int64)) : Nil
    bitwise_or(uint64_view(dst), uint64_view(a), uint64_view(b))
  end

  def xor(dst : Slice(Int64), a : Slice(Int64), b : Slice(Int64)) : Nil
    xor(uint64_view(dst), uint64_view(a), uint64_view(b))
  end

  def bswap(dst : Slice(Int64), a : Slice(Int64)) : Nil
    bswap(uint64_view(dst), uint64_view(a))
  end

  # Compare/mask - needs signed comparison
  abstract def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int64), b : Slice(Int64)) : Nil

  # ============================================================
  # UINT32 OPERATIONS (existing, now with full set)
  # ============================================================

  # ---------------------------
  # Arithmetic (UInt32)
  # ---------------------------

  abstract def add(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
  abstract def sub(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
  abstract def mul(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
  abstract def clamp(dst : Slice(UInt32), a : Slice(UInt32), lo : UInt32, hi : UInt32) : Nil

  # ---------------------------
  # Reductions (UInt32)
  # ---------------------------

  abstract def sum(a : Slice(UInt32)) : UInt32
  abstract def max(a : Slice(UInt32)) : UInt32
  abstract def min(a : Slice(UInt32)) : UInt32

  # ---------------------------
  # Bitwise (UInt32) - existing
  # ---------------------------

  abstract def bitwise_and(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
  abstract def bitwise_or(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
  abstract def xor(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
  abstract def bswap(dst : Slice(UInt32), a : Slice(UInt32)) : Nil

  # ---------------------------
  # Compare / mask (UInt32)
  # ---------------------------

  abstract def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt32), b : Slice(UInt32)) : Nil

  # ============================================================
  # INT32 OPERATIONS (forward to UInt32 where possible)
  # ============================================================

  def add(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    add(uint32_view(dst), uint32_view(a), uint32_view(b))
  end

  def sub(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    sub(uint32_view(dst), uint32_view(a), uint32_view(b))
  end

  def mul(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    mul(uint32_view(dst), uint32_view(a), uint32_view(b))
  end

  abstract def clamp(dst : Slice(Int32), a : Slice(Int32), lo : Int32, hi : Int32) : Nil

  def sum(a : Slice(Int32)) : Int32
    sum(uint32_view(a)).to_i32!
  end

  abstract def max(a : Slice(Int32)) : Int32
  abstract def min(a : Slice(Int32)) : Int32

  def bitwise_and(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    bitwise_and(uint32_view(dst), uint32_view(a), uint32_view(b))
  end

  def bitwise_or(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    bitwise_or(uint32_view(dst), uint32_view(a), uint32_view(b))
  end

  def xor(dst : Slice(Int32), a : Slice(Int32), b : Slice(Int32)) : Nil
    xor(uint32_view(dst), uint32_view(a), uint32_view(b))
  end

  def bswap(dst : Slice(Int32), a : Slice(Int32)) : Nil
    bswap(uint32_view(dst), uint32_view(a))
  end

  abstract def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int32), b : Slice(Int32)) : Nil

  # ============================================================
  # UINT16 OPERATIONS
  # ============================================================

  abstract def add(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
  abstract def sub(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
  abstract def mul(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
  abstract def clamp(dst : Slice(UInt16), a : Slice(UInt16), lo : UInt16, hi : UInt16) : Nil

  abstract def sum(a : Slice(UInt16)) : UInt16
  abstract def max(a : Slice(UInt16)) : UInt16
  abstract def min(a : Slice(UInt16)) : UInt16

  abstract def bitwise_and(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
  abstract def bitwise_or(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
  abstract def xor(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
  abstract def bswap(dst : Slice(UInt16), a : Slice(UInt16)) : Nil

  abstract def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt16), b : Slice(UInt16)) : Nil

  # ============================================================
  # INT16 OPERATIONS (forward to UInt16 where possible)
  # ============================================================

  def add(dst : Slice(Int16), a : Slice(Int16), b : Slice(Int16)) : Nil
    add(uint16_view(dst), uint16_view(a), uint16_view(b))
  end

  def sub(dst : Slice(Int16), a : Slice(Int16), b : Slice(Int16)) : Nil
    sub(uint16_view(dst), uint16_view(a), uint16_view(b))
  end

  def mul(dst : Slice(Int16), a : Slice(Int16), b : Slice(Int16)) : Nil
    mul(uint16_view(dst), uint16_view(a), uint16_view(b))
  end

  abstract def clamp(dst : Slice(Int16), a : Slice(Int16), lo : Int16, hi : Int16) : Nil

  def sum(a : Slice(Int16)) : Int16
    sum(uint16_view(a)).to_i16!
  end

  abstract def max(a : Slice(Int16)) : Int16
  abstract def min(a : Slice(Int16)) : Int16

  def bitwise_and(dst : Slice(Int16), a : Slice(Int16), b : Slice(Int16)) : Nil
    bitwise_and(uint16_view(dst), uint16_view(a), uint16_view(b))
  end

  def bitwise_or(dst : Slice(Int16), a : Slice(Int16), b : Slice(Int16)) : Nil
    bitwise_or(uint16_view(dst), uint16_view(a), uint16_view(b))
  end

  def xor(dst : Slice(Int16), a : Slice(Int16), b : Slice(Int16)) : Nil
    xor(uint16_view(dst), uint16_view(a), uint16_view(b))
  end

  def bswap(dst : Slice(Int16), a : Slice(Int16)) : Nil
    bswap(uint16_view(dst), uint16_view(a))
  end

  abstract def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int16), b : Slice(Int16)) : Nil

  # ============================================================
  # UINT8 OPERATIONS
  # ============================================================

  abstract def add(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
  abstract def sub(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
  abstract def mul(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
  abstract def clamp(dst : Slice(UInt8), a : Slice(UInt8), lo : UInt8, hi : UInt8) : Nil

  abstract def sum(a : Slice(UInt8)) : UInt8
  abstract def max(a : Slice(UInt8)) : UInt8
  abstract def min(a : Slice(UInt8)) : UInt8

  abstract def bitwise_and(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
  abstract def bitwise_or(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
  abstract def xor(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil

  abstract def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
  abstract def blend(dst : Slice(UInt8), t : Slice(UInt8), f : Slice(UInt8), mask : Slice(UInt8)) : Nil

  # popcount returns total bits set across slice
  abstract def popcount(a : Slice(UInt8)) : UInt64

  # ============================================================
  # INT8 OPERATIONS (forward to UInt8 where possible)
  # ============================================================

  def add(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    add(uint8_view(dst), uint8_view(a), uint8_view(b))
  end

  def sub(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    sub(uint8_view(dst), uint8_view(a), uint8_view(b))
  end

  def mul(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    mul(uint8_view(dst), uint8_view(a), uint8_view(b))
  end

  abstract def clamp(dst : Slice(Int8), a : Slice(Int8), lo : Int8, hi : Int8) : Nil

  def sum(a : Slice(Int8)) : Int8
    sum(uint8_view(a)).to_i8!
  end

  abstract def max(a : Slice(Int8)) : Int8
  abstract def min(a : Slice(Int8)) : Int8

  def bitwise_and(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    bitwise_and(uint8_view(dst), uint8_view(a), uint8_view(b))
  end

  def bitwise_or(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    bitwise_or(uint8_view(dst), uint8_view(a), uint8_view(b))
  end

  def xor(dst : Slice(Int8), a : Slice(Int8), b : Slice(Int8)) : Nil
    xor(uint8_view(dst), uint8_view(a), uint8_view(b))
  end

  abstract def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil

  def blend(dst : Slice(Int8), t : Slice(Int8), f : Slice(Int8), mask : Slice(UInt8)) : Nil
    blend(uint8_view(dst), uint8_view(t), uint8_view(f), mask)
  end

  # ============================================================
  # MEMORY / LAYOUT HELPERS
  # ============================================================

  # dst[i] = src[i]  (may use wide loads/stores + prefetch)
  abstract def copy(dst : Slice(UInt8), src : Slice(UInt8)) : Nil

  # Set all bytes to value (memset-like)
  abstract def fill(dst : Slice(UInt8), value : UInt8) : Nil

  # ============================================================
  # CONVERSION OPERATIONS
  # ============================================================

  # Int16 -> Float32
  abstract def convert(dst : Slice(Float32), src : Slice(Int16)) : Nil

  # UInt8 -> Float32 with scale
  abstract def convert(dst : Slice(Float32), src : Slice(UInt8), scale : Float32) : Nil

  # Int32 -> Float64
  abstract def convert(dst : Slice(Float64), src : Slice(Int32)) : Nil

  # Int16 -> Float64
  abstract def convert(dst : Slice(Float64), src : Slice(Int16)) : Nil

  # Float32 -> Float64 (widening)
  abstract def convert(dst : Slice(Float64), src : Slice(Float32)) : Nil

  # Float64 -> Float32 (narrowing)
  abstract def convert(dst : Slice(Float32), src : Slice(Float64)) : Nil

  # ============================================================
  # CRYPTO-ADJACENT BLOCK OPS
  # ============================================================

  # XOR a stream with a 16-byte repeated block key
  abstract def xor_block16(dst : Slice(UInt8), src : Slice(UInt8), key16 : StaticArray(UInt8, 16)) : Nil

  # ============================================================
  # ADDITIONAL OPERATIONS
  # ============================================================

  # --------------------------------
  # Math helpers
  # --------------------------------

  abstract def abs(dst : Slice(UInt8), a : Slice(UInt8)) : Nil
  abstract def abs(dst : Slice(Int8), a : Slice(Int8)) : Nil
  abstract def abs(dst : Slice(UInt16), a : Slice(UInt16)) : Nil
  abstract def abs(dst : Slice(Int16), a : Slice(Int16)) : Nil
  abstract def abs(dst : Slice(UInt32), a : Slice(UInt32)) : Nil
  abstract def abs(dst : Slice(Int32), a : Slice(Int32)) : Nil
  abstract def abs(dst : Slice(UInt64), a : Slice(UInt64)) : Nil
  abstract def abs(dst : Slice(Int64), a : Slice(Int64)) : Nil

  abstract def neg(dst : Slice(Int8), a : Slice(Int8)) : Nil
  abstract def neg(dst : Slice(Int16), a : Slice(Int16)) : Nil
  abstract def neg(dst : Slice(Int32), a : Slice(Int32)) : Nil
  abstract def neg(dst : Slice(Int64), a : Slice(Int64)) : Nil

  # --------------------------------
  # Comparisons / masks
  # --------------------------------

  abstract def cmp_eq(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
  abstract def cmp_ne(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
  abstract def cmp_lt(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
  abstract def cmp_le(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
  abstract def cmp_ge(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T

  # --------------------------------
  # Element-wise min/max
  # --------------------------------

  abstract def min(dst : Slice(T), a : Slice(T), b : Slice(T)) : Nil forall T
  abstract def max(dst : Slice(T), a : Slice(T), b : Slice(T)) : Nil forall T

  # ============================================================
  # PRIVATE HELPERS FOR SLICE VIEWS
  # ============================================================

  @[AlwaysInline]
  private def uint64_view(slice : Slice(Int64)) : Slice(UInt64)
    Slice(UInt64).new(slice.to_unsafe.as(Pointer(UInt64)), slice.size)
  end

  @[AlwaysInline]
  private def uint32_view(slice : Slice(Int32)) : Slice(UInt32)
    Slice(UInt32).new(slice.to_unsafe.as(Pointer(UInt32)), slice.size)
  end

  @[AlwaysInline]
  private def uint16_view(slice : Slice(Int16)) : Slice(UInt16)
    Slice(UInt16).new(slice.to_unsafe.as(Pointer(UInt16)), slice.size)
  end

  @[AlwaysInline]
  private def uint8_view(slice : Slice(Int8)) : Slice(UInt8)
    Slice(UInt8).new(slice.to_unsafe.as(Pointer(UInt8)), slice.size)
  end
end
