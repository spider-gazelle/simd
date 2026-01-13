# SIMD

A Crystal library providing a standardized interface for SIMD (Single Instruction, Multiple Data) operations across different CPU architectures. The library automatically selects the best available instruction set at runtime and falls back to scalar operations when SIMD is not available or slower.

## Features

- **Cross-platform**: Supports x86_64 (SSE2, SSE4.1, AVX2, AVX-512) and AArch64 (NEON, SVE, SVE2)
- **Automatic dispatch**: Detects CPU capabilities at runtime and uses the fastest available implementation
- **Smart fallbacks**: Operations that are slower with SIMD automatically use scalar implementations
- **Zero-copy**: Works directly with Crystal `Slice` types for efficient memory usage

## Installation

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     simd:
       github: spider-gazelle/simd
   ```

2. Run `shards install`

## Usage

```crystal
require "simd"

# Get the optimal SIMD implementation for this CPU
simd = SIMD.instance

# Check what's supported
puts SIMD.supported_instruction_sets  # => SSE2 | SSE41 | AVX2 | AVX512

# Create some data
a = Slice(Float32).new(1024) { |i| i.to }
b = Slice(Float32).new(1024) { |i| (i * 2).to }
dst = Slice(Float32).new(1024)

# Perform SIMD-accelerated operations
simd.add(dst, a, b)      # dst[i] = a[i] + b[i]
simd.mul(dst, a, b)      # dst[i] = a[i] * b[i]
simd.fma(dst, a, b, dst) # dst[i] = a[i] * b[i] + dst[i]

# Reductions
sum = simd.sum(a)        # Returns sum of all elements
dot = simd.dot(a, b)     # Returns dot product
max = simd.max(a)        # Returns maximum value

# Integer operations
u32_a = Slice(UInt32).new(1024) { |i| i.to }
u32_b = Slice(UInt32).new(1024) { |i| (i * 3).to }
u32_dst = Slice(UInt32).new(1024)

simd.xor(u32_dst, u32_a, u32_b)  # Bitwise XOR
simd.bswap(u32_dst, u32_a)       # Byte swap (endian conversion)
```

### Available Operations

#### Vector Math (Float32)

| Method | Description |
|--------|-------------|
| `add(dst, a, b)` | `dst[i] = a[i] + b[i]` |
| `sub(dst, a, b)` | `dst[i] = a[i] - b[i]` |
| `mul(dst, a, b)` | `dst[i] = a[i] * b[i]` |
| `fma(dst, a, b, c)` | `dst[i] = a[i] * b[i] + c[i]` (fused multiply-add) |
| `clamp(dst, a, lo, hi)` | `dst[i] = clamp(a[i], lo, hi)` |
| `axpby(dst, a, b, alpha, beta)` | `dst[i] = alpha * a[i] + beta * b[i]` |

#### Reductions (Float32)

| Method | Description |
|--------|-------------|
| `sum(a)` | Returns sum of all elements |
| `dot(a, b)` | Returns dot product `sum(a[i] * b[i])` |
| `max(a)` | Returns maximum value |

#### Integer/Bitwise (UInt32)

| Method | Description |
|--------|-------------|
| `bitwise_and(dst, a, b)` | `dst[i] = a[i] & b[i]` |
| `bitwise_or(dst, a, b)` | `dst[i] = a[i] \| b[i]` |
| `xor(dst, a, b)` | `dst[i] = a[i] ^ b[i]` |
| `bswap(dst, a)` | Byte swap each element (endian conversion) |
| `popcount(a)` | Count total bits set across all bytes |

#### Compare/Mask/Filter

| Method | Description |
|--------|-------------|
| `cmp_gt_mask(mask, a, b)` | `mask[i] = a[i] > b[i] ? 0xFF : 0x00` |
| `blend(dst, t, f, mask)` | `dst[i] = mask[i] ? t[i] : f[i]` |
| `compress(dst, src, mask)` | Copy elements where `mask[i] != 0`, returns count |

#### Memory Operations

| Method | Description |
|--------|-------------|
| `copy(dst, src)` | Fast memory copy |
| `fill(dst, value)` | Fill memory with value |

#### Type Conversion

| Method | Description |
|--------|-------------|
| `convert(dst, src)` | Convert Int16 to Float32 |
| `convert(dst, src, scale)` | Convert UInt8 to Float32 with scaling |

#### DSP/Convolution

| Method | Description |
|--------|-------------|
| `fir(dst, src, coeff)` | FIR filter: `dst[i] = sum(coeff[k] * src[i+k])` |

#### Crypto-adjacent

| Method | Description |
|--------|-------------|
| `xor_block16(dst, src, key16)` | XOR with repeating 16-byte key |

### Using Specific Implementations

You can also instantiate specific implementations directly:

```crystal
{% if flag?(:x86_64) %}
  sse2 = SIMD::SSE2.new
  avx2 = SIMD::AVX2.new if SIMD.supported_instruction_sets.avx2?
{% elsif flag?(:aarch64) %}
  neon = SIMD::NEON.new if SIMD.supported_instruction_sets.neon?
{% end %}
```

## Benchmarking

The library includes a comprehensive benchmark suite to compare implementations.

### Building the Benchmark

```bash
shards build --release
```

### Running Benchmarks

```bash
# Run all benchmarks
./bin/benchmark

# List available benchmarks
./bin/benchmark --list

# Run specific benchmark(s)
./bin/benchmark -t add
./bin/benchmark -t add -t mul -t dot

# Show help
./bin/benchmark --help
```

### Available Benchmarks

```
add         - dst = a + b
mul         - dst = a * b
fma         - dst = a * b + c
sum         - horizontal sum
dot         - dot product
max         - find maximum
clamp       - clamp to range
xor         - bitwise XOR
bswap       - byte swap
popcount     - population count
copy         - memory copy
fill         - memory fill
convert      - int16 to float conversion
xor_block16     - XOR with 16-byte key
fir         - 5-tap FIR filter
```

### Example Output

```
======================================================================
SIMD Benchmark
======================================================================

System Information:
  SIMD Support: SSE2 | SSE41 | AVX2 | AVX512
  Available Implementations: SIMD::SSE2, SIMD::SSE41, SIMD::AVX2, SIMD::AVX512

----------------------------------------------------------------------
add (dst = a + b) - 4096 elements
----------------------------------------------------------------------
Scalar 957.37k (  1.04µs) (± 0.58%)  0.0B/op   7.81× slower
  SSE2 954.17k (  1.05µs) (± 0.65%)  0.0B/op   7.84× slower
 SSE41 957.07k (  1.04µs) (± 0.47%)  0.0B/op   7.82× slower
  AVX2   4.38M (228.51ns) (± 1.14%)  0.0B/op   1.71× slower
AVX512   7.48M (133.66ns) (± 0.91%)  0.0B/op        fastest
```

## Future Work

### Additional Data Types

Currently the library focuses on `Float32` and `UInt32`. Future versions could add:

#### Float64 (Double Precision)

```crystal
# Planned API
simd.add_f64(dst, a, b)
simd.mul_f64(dst, a, b)
simd.fma_f64(dst, a, b, c)
simd.sum_f64(a)
simd.dot_f64(a, b)
```

Note: SIMD width is halved for Float64 (e.g., AVX2 processes 4 doubles vs 8 floats).

#### Additional Integer Types

```crystal
# Signed integers
simd.add_i32(dst, a, b)
simd.add_i64(dst, a, b)

# Unsigned integers
simd.add_u64(dst, a, b)

# Smaller types (useful for image/audio processing)
simd.add_i16(dst, a, b)
simd.add(dst, a, b)    # With saturation options
```

### Method Overloading

Replace type-suffixed methods with overloaded versions that dispatch based on slice type:

```crystal
# Current API
simd.mul(dst, a, b)
simd.xor(dst, a, b)

# Proposed API with overloading
simd.mul(dst, a, b)   # Dispatches to Float32 version
simd.mul(dst_f64, a_f64, b_f64)   # Dispatches to Float64 version
simd.xor(dst, a, b)   # Dispatches to UInt32 version
simd.xor(dst_u64, a_u64, b_u64)   # Dispatches to UInt64 version
```

### Additional Operations

#### Math Functions

```crystal
simd.div(dst, a, b)        # Element-wise division
simd.sqrt(dst, a)          # Square root
simd.rsqrt(dst, a)         # Reciprocal square root (fast approximation)
simd.abs(dst, a)           # Absolute value
simd.neg(dst, a)           # Negation
simd.min(dst, a, b)        # Element-wise minimum
simd.max(dst, a, b)        # Element-wise maximum (binary version)
simd.floor(dst, a)         # Floor
simd.ceil(dst, a)          # Ceiling
simd.round(dst, a)         # Round to nearest
```

#### Reduction Operations

```crystal
simd.min(a)                # Find minimum value
simd.argmax(a)             # Index of maximum value
simd.argmin(a)             # Index of minimum value
simd.any_nonzero(a)        # Check if any element is non-zero
simd.all_nonzero(a)        # Check if all elements are non-zero
```

#### Comparison Operations

```crystal
simd.cmp_eq(mask, a, b)    # Equal
simd.cmp_ne(mask, a, b)    # Not equal
simd.cmp_lt(mask, a, b)    # Less than
simd.cmp_le(mask, a, b)    # Less than or equal
simd.cmp_ge(mask, a, b)    # Greater than or equal
```

#### Gather/Scatter

```crystal
simd.gather(dst, src, indices)   # dst[i] = src[indices[i]]
simd.scatter(dst, src, indices)  # dst[indices[i]] = src[i]
```

#### Matrix Operations

```crystal
simd.gemv(dst, matrix, vector, alpha, beta)  # Matrix-vector multiply
simd.gemm(c, a, b, alpha, beta)              # Matrix-matrix multiply
simd.transpose(dst, src, rows, cols)         # Matrix transpose
```

#### String/Text Processing

```crystal
simd.find_byte(haystack, needle)      # Find first occurrence of byte
simd.find_any_byte(haystack, needles) # Find first occurrence of any needle
simd.count_byte(haystack, needle)     # Count occurrences
simd.to_lowercase(dst, src)           # ASCII lowercase conversion
simd.to_uppercase(dst, src)           # ASCII uppercase conversion
```

#### Image Processing

```crystal
simd.rgb_to_grayscale(dst, src)       # Color conversion
simd.premultiply_alpha(dst, src)      # Alpha premultiplication
simd.bilinear_sample(dst, src, ...)   # Bilinear interpolation
```

### Performance Improvements

- **Prefetching**: Add software prefetch hints for large sequential operations
- **Non-temporal stores**: Use streaming stores for write-only large buffers
- **Alignment hints**: Optimize for aligned memory access patterns
- **Loop unrolling**: Process multiple vectors per iteration for better ILP

## Contributing

1. Fork it (<https://github.com/spider-gazelle/simd/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- [Stephen von Takach](https://github.com/stakach) - creator and maintainer
