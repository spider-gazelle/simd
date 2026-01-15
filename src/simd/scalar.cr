# ------------------------------------------------------------
# Scalar backend so higher-level code works everywhere
# ------------------------------------------------------------
class SIMD::Scalar < SIMD::Base
  private def check_len(*slices)
    n = slices[0].as(Slice).size
    slices.each do |slice|
      raise ArgumentError.new("length mismatch") unless slice.as(Slice).size == n
    end
    n
  end

  # ============================================================
  # FLOAT32 OPERATIONS
  # ============================================================

  def add(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] + b[i]
      i += 1
    end
  end

  def sub(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] - b[i]
      i += 1
    end
  end

  def mul(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] * b[i]
      i += 1
    end
  end

  def fma(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32), c : Slice(Float32)) : Nil
    n = check_len(dst, a, b, c)
    i = 0
    while i < n
      dst[i] = a[i] * b[i] + c[i]
      i += 1
    end
  end

  def clamp(dst : Slice(Float32), a : Slice(Float32), lo : Float32, hi : Float32) : Nil
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

  def axpby(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32), alpha : Float32, beta : Float32) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = alpha * a[i] + beta * b[i]
      i += 1
    end
  end

  def div(dst : Slice(Float32), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] / b[i]
      i += 1
    end
  end

  def sqrt(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = Math.sqrt(a[i])
      i += 1
    end
  end

  def rsqrt(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = 1.0_f32 / Math.sqrt(a[i])
      i += 1
    end
  end

  def abs(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      # Float32#abs returns Float64; preserve Float32 and clear sign bit (incl. -0.0).
      u = a[i].unsafe_as(UInt32)
      dst[i] = (u & 0x7fff_ffff_u32).unsafe_as(Float32)
      i += 1
    end
  end

  def neg(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = -a[i]
      i += 1
    end
  end

  def floor(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i].floor
      i += 1
    end
  end

  def ceil(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i].ceil
      i += 1
    end
  end

  def round(dst : Slice(Float32), a : Slice(Float32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i].round
      i += 1
    end
  end

  def sum(a : Slice(Float32)) : Float32
    acc = 0.0_f32
    a.each { |v| acc += v }
    acc
  end

  def dot(a : Slice(Float32), b : Slice(Float32)) : Float32
    raise ArgumentError.new("length mismatch") unless a.size == b.size
    acc = 0.0_f32
    i = 0
    n = a.size
    while i < n
      acc += a[i] * b[i]
      i += 1
    end
    acc
  end

  def max(a : Slice(Float32)) : Float32
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v > m
      i += 1
    end
    m
  end

  def min(a : Slice(Float32)) : Float32
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v < m
      i += 1
    end
    m
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float32), b : Slice(Float32)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def fir(dst : Slice(Float32), src : Slice(Float32), coeff : Slice(Float32)) : Nil
    taps = coeff.size
    raise ArgumentError.new("coeff empty") if taps == 0
    raise ArgumentError.new("src too small") if src.size < dst.size + taps - 1
    i = 0
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

  # ============================================================
  # FLOAT64 OPERATIONS
  # ============================================================

  def add(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] + b[i]
      i += 1
    end
  end

  def sub(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] - b[i]
      i += 1
    end
  end

  def mul(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] * b[i]
      i += 1
    end
  end

  def fma(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64), c : Slice(Float64)) : Nil
    n = check_len(dst, a, b, c)
    i = 0
    while i < n
      dst[i] = a[i] * b[i] + c[i]
      i += 1
    end
  end

  def clamp(dst : Slice(Float64), a : Slice(Float64), lo : Float64, hi : Float64) : Nil
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

  def axpby(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64), alpha : Float64, beta : Float64) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = alpha * a[i] + beta * b[i]
      i += 1
    end
  end

  def div(dst : Slice(Float64), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] / b[i]
      i += 1
    end
  end

  def sqrt(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = Math.sqrt(a[i])
      i += 1
    end
  end

  def rsqrt(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = 1.0_f64 / Math.sqrt(a[i])
      i += 1
    end
  end

  def abs(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i].abs
      i += 1
    end
  end

  def neg(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = -a[i]
      i += 1
    end
  end

  def floor(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i].floor
      i += 1
    end
  end

  def ceil(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i].ceil
      i += 1
    end
  end

  def round(dst : Slice(Float64), a : Slice(Float64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i].round
      i += 1
    end
  end

  def sum(a : Slice(Float64)) : Float64
    acc = 0.0_f64
    a.each { |v| acc += v }
    acc
  end

  def dot(a : Slice(Float64), b : Slice(Float64)) : Float64
    raise ArgumentError.new("length mismatch") unless a.size == b.size
    acc = 0.0_f64
    i = 0
    n = a.size
    while i < n
      acc += a[i] * b[i]
      i += 1
    end
    acc
  end

  def max(a : Slice(Float64)) : Float64
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v > m
      i += 1
    end
    m
  end

  def min(a : Slice(Float64)) : Float64
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v < m
      i += 1
    end
    m
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Float64), b : Slice(Float64)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def fir(dst : Slice(Float64), src : Slice(Float64), coeff : Slice(Float64)) : Nil
    taps = coeff.size
    raise ArgumentError.new("coeff empty") if taps == 0
    raise ArgumentError.new("src too small") if src.size < dst.size + taps - 1
    i = 0
    while i < dst.size
      acc = 0.0_f64
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
  # UINT64 OPERATIONS
  # ============================================================

  def add(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] &+ b[i]
      i += 1
    end
  end

  def sub(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] &- b[i]
      i += 1
    end
  end

  def mul(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    i = 0
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

  def abs(dst : Slice(UInt64), a : Slice(UInt64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i]
      i += 1
    end
  end

  def sum(a : Slice(UInt64)) : UInt64
    acc = 0_u64
    a.each { |v| acc &+= v }
    acc
  end

  def max(a : Slice(UInt64)) : UInt64
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v > m
      i += 1
    end
    m
  end

  def min(a : Slice(UInt64)) : UInt64
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v < m
      i += 1
    end
    m
  end

  def bitwise_and(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] & b[i]
      i += 1
    end
  end

  def bitwise_or(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] | b[i]
      i += 1
    end
  end

  def xor(dst : Slice(UInt64), a : Slice(UInt64), b : Slice(UInt64)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] ^ b[i]
      i += 1
    end
  end

  def bswap(dst : Slice(UInt64), a : Slice(UInt64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i].byte_swap
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

  # ============================================================
  # INT64 OPERATIONS (signed-specific)
  # ============================================================

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

  def abs(dst : Slice(Int64), a : Slice(Int64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      v = a[i]
      dst[i] = v < 0 ? (0_i64 &- v) : v
      i += 1
    end
  end

  def neg(dst : Slice(Int64), a : Slice(Int64)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = 0_i64 &- a[i]
      i += 1
    end
  end

  def max(a : Slice(Int64)) : Int64
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v > m
      i += 1
    end
    m
  end

  def min(a : Slice(Int64)) : Int64
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v < m
      i += 1
    end
    m
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int64), b : Slice(Int64)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
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
    i = 0
    while i < n
      dst[i] = a[i] &+ b[i]
      i += 1
    end
  end

  def sub(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] &- b[i]
      i += 1
    end
  end

  def mul(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    i = 0
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

  def abs(dst : Slice(UInt32), a : Slice(UInt32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i]
      i += 1
    end
  end

  def sum(a : Slice(UInt32)) : UInt32
    acc = 0_u32
    a.each { |v| acc &+= v }
    acc
  end

  def max(a : Slice(UInt32)) : UInt32
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v > m
      i += 1
    end
    m
  end

  def min(a : Slice(UInt32)) : UInt32
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v < m
      i += 1
    end
    m
  end

  def bitwise_and(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] & b[i]
      i += 1
    end
  end

  def bitwise_or(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] | b[i]
      i += 1
    end
  end

  def xor(dst : Slice(UInt32), a : Slice(UInt32), b : Slice(UInt32)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] ^ b[i]
      i += 1
    end
  end

  def bswap(dst : Slice(UInt32), a : Slice(UInt32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i].byte_swap
      i += 1
    end
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
  # INT32 OPERATIONS (signed-specific)
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

  def abs(dst : Slice(Int32), a : Slice(Int32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      v = a[i]
      dst[i] = v < 0 ? (0_i32 &- v) : v
      i += 1
    end
  end

  def neg(dst : Slice(Int32), a : Slice(Int32)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = 0_i32 &- a[i]
      i += 1
    end
  end

  def max(a : Slice(Int32)) : Int32
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v > m
      i += 1
    end
    m
  end

  def min(a : Slice(Int32)) : Int32
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v < m
      i += 1
    end
    m
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
  # UINT16 OPERATIONS
  # ============================================================

  def add(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] &+ b[i]
      i += 1
    end
  end

  def sub(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] &- b[i]
      i += 1
    end
  end

  def mul(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    i = 0
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

  def abs(dst : Slice(UInt16), a : Slice(UInt16)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i]
      i += 1
    end
  end

  def sum(a : Slice(UInt16)) : UInt16
    acc = 0_u16
    a.each { |v| acc &+= v }
    acc
  end

  def max(a : Slice(UInt16)) : UInt16
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v > m
      i += 1
    end
    m
  end

  def min(a : Slice(UInt16)) : UInt16
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v < m
      i += 1
    end
    m
  end

  def bitwise_and(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] & b[i]
      i += 1
    end
  end

  def bitwise_or(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] | b[i]
      i += 1
    end
  end

  def xor(dst : Slice(UInt16), a : Slice(UInt16), b : Slice(UInt16)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] ^ b[i]
      i += 1
    end
  end

  def bswap(dst : Slice(UInt16), a : Slice(UInt16)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = a[i].byte_swap
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
  # INT16 OPERATIONS (signed-specific)
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

  def abs(dst : Slice(Int16), a : Slice(Int16)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      v = a[i]
      dst[i] = v < 0 ? (0_i16 &- v) : v
      i += 1
    end
  end

  def neg(dst : Slice(Int16), a : Slice(Int16)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = 0_i16 &- a[i]
      i += 1
    end
  end

  def max(a : Slice(Int16)) : Int16
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v > m
      i += 1
    end
    m
  end

  def min(a : Slice(Int16)) : Int16
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v < m
      i += 1
    end
    m
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
  # UINT8 OPERATIONS
  # ============================================================

  def add(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] &+ b[i]
      i += 1
    end
  end

  def sub(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] &- b[i]
      i += 1
    end
  end

  def mul(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    i = 0
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

  def abs(dst : Slice(UInt8), a : Slice(UInt8)) : Nil
    copy(dst, a)
  end

  def sum(a : Slice(UInt8)) : UInt8
    acc = 0_u8
    a.each { |v| acc &+= v }
    acc
  end

  def max(a : Slice(UInt8)) : UInt8
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v > m
      i += 1
    end
    m
  end

  def min(a : Slice(UInt8)) : UInt8
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v < m
      i += 1
    end
    m
  end

  def bitwise_and(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] & b[i]
      i += 1
    end
  end

  def bitwise_or(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    i = 0
    while i < n
      dst[i] = a[i] | b[i]
      i += 1
    end
  end

  def xor(dst : Slice(UInt8), a : Slice(UInt8), b : Slice(UInt8)) : Nil
    n = check_len(dst, a, b)
    i = 0
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

  def popcount(a : Slice(UInt8)) : UInt64
    acc = 0_u64
    a.each do |v|
      acc += v.popcount.to_u64
    end
    acc
  end

  # ============================================================
  # INT8 OPERATIONS (signed-specific)
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

  def abs(dst : Slice(Int8), a : Slice(Int8)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      v = a[i]
      dst[i] = v < 0 ? (0_i8 &- v) : v
      i += 1
    end
  end

  def neg(dst : Slice(Int8), a : Slice(Int8)) : Nil
    n = check_len(dst, a)
    i = 0
    while i < n
      dst[i] = 0_i8 &- a[i]
      i += 1
    end
  end

  def max(a : Slice(Int8)) : Int8
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v > m
      i += 1
    end
    m
  end

  def min(a : Slice(Int8)) : Int8
    raise ArgumentError.new("empty") if a.empty?
    m = a[0]
    i = 1
    while i < a.size
      v = a[i]
      m = v if v < m
      i += 1
    end
    m
  end

  def cmp_gt_mask(dst_mask : Slice(UInt8), a : Slice(Int8), b : Slice(Int8)) : Nil
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] > b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  # ============================================================
  # MEMORY / LAYOUT HELPERS
  # ============================================================

  def copy(dst : Slice(UInt8), src : Slice(UInt8)) : Nil
    n = check_len(dst, src)
    i = 0
    while i < n
      dst[i] = src[i]
      i += 1
    end
  end

  def fill(dst : Slice(UInt8), value : UInt8) : Nil
    i = 0
    while i < dst.size
      dst[i] = value
      i += 1
    end
  end

  # ============================================================
  # CONVERSION OPERATIONS
  # ============================================================

  def convert(dst : Slice(Float32), src : Slice(Int16)) : Nil
    n = check_len(dst, src)
    i = 0
    while i < n
      dst[i] = src[i].to_f32
      i += 1
    end
  end

  def convert(dst : Slice(Float32), src : Slice(UInt8), scale : Float32) : Nil
    n = check_len(dst, src)
    i = 0
    while i < n
      dst[i] = src[i].to_f32 * scale
      i += 1
    end
  end

  def convert(dst : Slice(Float64), src : Slice(Int32)) : Nil
    n = check_len(dst, src)
    i = 0
    while i < n
      dst[i] = src[i].to_f64
      i += 1
    end
  end

  def convert(dst : Slice(Float64), src : Slice(Int16)) : Nil
    n = check_len(dst, src)
    i = 0
    while i < n
      dst[i] = src[i].to_f64
      i += 1
    end
  end

  def convert(dst : Slice(Float64), src : Slice(Float32)) : Nil
    n = check_len(dst, src)
    i = 0
    while i < n
      dst[i] = src[i].to_f64
      i += 1
    end
  end

  def convert(dst : Slice(Float32), src : Slice(Float64)) : Nil
    n = check_len(dst, src)
    i = 0
    while i < n
      dst[i] = src[i].to_f32
      i += 1
    end
  end

  # ============================================================
  # CRYPTO-ADJACENT BLOCK OPS
  # ============================================================

  def xor_block16(dst : Slice(UInt8), src : Slice(UInt8), key16 : StaticArray(UInt8, 16)) : Nil
    raise ArgumentError.new("length mismatch") unless dst.size == src.size
    i = 0
    while i < src.size
      dst[i] = src[i] ^ key16[i & 15]
      i += 1
    end
  end

  # ============================================================
  # ADDITIONAL OPERATIONS
  # ============================================================

  def cmp_eq(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] == b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ne(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] != b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_lt(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] < b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_le(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] <= b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def cmp_ge(dst_mask : Slice(UInt8), a : Slice(T), b : Slice(T)) : Nil forall T
    n = check_len(dst_mask, a, b)
    i = 0
    while i < n
      dst_mask[i] = a[i] >= b[i] ? 0xFF_u8 : 0x00_u8
      i += 1
    end
  end

  def min(dst : Slice(T), a : Slice(T), b : Slice(T)) : Nil forall T
    n = check_len(dst, a, b)
    i = 0
    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av < bv ? av : bv
      i += 1
    end
  end

  def max(dst : Slice(T), a : Slice(T), b : Slice(T)) : Nil forall T
    n = check_len(dst, a, b)
    i = 0
    while i < n
      av = a[i]
      bv = b[i]
      dst[i] = av > bv ? av : bv
      i += 1
    end
  end
end
