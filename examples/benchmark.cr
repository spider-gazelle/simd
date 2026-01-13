require "../src/simd"
require "benchmark"
require "option_parser"

# Benchmark comparing SIMD vs Scalar performance
# Run with: crystal build --release examples/benchmark.cr && ./benchmark
#
# Options:
#   --list, -l           List available benchmarks
#   --test, -t NAME      Run specific benchmark (can be used multiple times)
#   (no options)         Run all benchmarks

def check_implementations : Array(SIMD::Base)
  code_paths = [] of SIMD::Base
  supports = SIMD.supported_instruction_sets

  {% if flag?(:x86_64) %}
    code_paths << SIMD::SSE2.new if supports.sse2?
    code_paths << SIMD::SSE41.new if supports.sse41?
    code_paths << SIMD::AVX2.new if supports.avx2?
    code_paths << SIMD::AVX512.new if supports.avx512?
  {% elsif flag?(:aarch64) %}
    code_paths << SIMD::NEON.new if supports.neon?
    code_paths << SIMD::SVE.new if supports.sve?
    code_paths << SIMD::SVE2.new if supports.sve2?
  {% end %}

  code_paths
end

# Pre-allocate buffers
SIZE = 4096

# Float32 buffers
a_f32 = Slice(Float32).new(SIZE) { |i| (i % 100).to_f32 * 0.01_f32 }
b_f32 = Slice(Float32).new(SIZE) { |i| ((i + 50) % 100).to_f32 * 0.01_f32 }
c_f32 = Slice(Float32).new(SIZE) { |i| ((i + 25) % 100).to_f32 * 0.01_f32 }
dst_f32 = Slice(Float32).new(SIZE)

# Float64 buffers
a_f64 = Slice(Float64).new(SIZE) { |i| (i % 100).to_f64 * 0.01_f64 }
b_f64 = Slice(Float64).new(SIZE) { |i| ((i + 50) % 100).to_f64 * 0.01_f64 }
c_f64 = Slice(Float64).new(SIZE) { |i| ((i + 25) % 100).to_f64 * 0.01_f64 }
dst_f64 = Slice(Float64).new(SIZE)

# UInt64 buffers
a_u64 = Slice(UInt64).new(SIZE) { |i| i.to_u64 &* 0x123456789ABCDEF0_u64 }
b_u64 = Slice(UInt64).new(SIZE) { |i| i.to_u64 &* 0xFEDCBA9876543210_u64 }
dst_u64 = Slice(UInt64).new(SIZE)

# UInt32 buffers
a_u32 = Slice(UInt32).new(SIZE) { |i| i.to_u32 &* 0x12345678_u32 }
b_u32 = Slice(UInt32).new(SIZE) { |i| i.to_u32 &* 0x87654321_u32 }
dst_u32 = Slice(UInt32).new(SIZE)

# UInt16 buffers
a_u16 = Slice(UInt16).new(SIZE) { |i| ((i &* 0x1234) & 0xFFFF).to_u16 }
b_u16 = Slice(UInt16).new(SIZE) { |i| ((i &* 0x4321) & 0xFFFF).to_u16 }
dst_u16 = Slice(UInt16).new(SIZE)

# UInt8 buffers
a_u8 = Slice(UInt8).new(SIZE) { |i| (i % 256).to_u8 }
b_u8 = Slice(UInt8).new(SIZE) { |i| ((i + 128) % 256).to_u8 }
dst_u8 = Slice(UInt8).new(SIZE)

# Int32 buffers
a_i32 = Slice(Int32).new(SIZE) { |i| i.to_i32 &* 0x12345678_i32 }
b_i32 = Slice(Int32).new(SIZE) { |i| i.to_i32 &* 0x07654321_i32 }
dst_i32 = Slice(Int32).new(SIZE)

src_i16 = Slice(Int16).new(SIZE) { |i| ((i % 65536) - 32768).to_i16 }

key16 = StaticArray(UInt8, 16).new { |i| ((i * 17 + 5) % 256).to_u8 }

coeff = Slice[0.1_f32, 0.2_f32, 0.4_f32, 0.2_f32, 0.1_f32] # 5-tap filter
fir_src = Slice(Float32).new(SIZE + 4) { |i| (i % 100).to_f32 * 0.01_f32 }
fir_dst = Slice(Float32).new(SIZE)

scalar = SIMD.scalar
implementations = check_implementations

# Define all benchmarks
benchmarks = {
  "add" => {
    "dst = a + b",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.add(dst_f32, a_f32, b_f32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.add(dst_f32, a_f32, b_f32) }
      end
    },
  },
  "mul" => {
    "dst = a * b",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.mul(dst_f32, a_f32, b_f32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.mul(dst_f32, a_f32, b_f32) }
      end
    },
  },
  "fma" => {
    "dst = a * b + c",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.fma(dst_f32, a_f32, b_f32, c_f32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.fma(dst_f32, a_f32, b_f32, c_f32) }
      end
    },
  },
  "sum" => {
    "horizontal sum",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.sum(a_f32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.sum(a_f32) }
      end
    },
  },
  "dot" => {
    "dot product",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.dot(a_f32, b_f32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.dot(a_f32, b_f32) }
      end
    },
  },
  "max" => {
    "find maximum",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.max(a_f32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.max(a_f32) }
      end
    },
  },
  "clamp" => {
    "clamp to range",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.clamp(dst_f32, a_f32, 0.0_f32, 1.0_f32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.clamp(dst_f32, a_f32, 0.0_f32, 1.0_f32) }
      end
    },
  },
  "xor" => {
    "bitwise XOR",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.xor(dst_u32, a_u32, b_u32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.xor(dst_u32, a_u32, b_u32) }
      end
    },
  },
  "bswap" => {
    "byte swap",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.bswap(dst_u32, a_u32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.bswap(dst_u32, a_u32) }
      end
    },
  },
  "popcount" => {
    "population count",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.popcount(a_u8) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.popcount(a_u8) }
      end
    },
  },
  "copy" => {
    "memory copy",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.copy(dst_u8, a_u8) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.copy(dst_u8, a_u8) }
      end
    },
  },
  "fill" => {
    "memory fill",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.fill(dst_u8, 0xAB_u8) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.fill(dst_u8, 0xAB_u8) }
      end
    },
  },
  "convert" => {
    "int16 to float conversion",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.convert(dst_f32, src_i16) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.convert(dst_f32, src_i16) }
      end
    },
  },
  "xor_block16" => {
    "XOR with 16-byte key",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.xor_block16(dst_u8, a_u8, key16) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.xor_block16(dst_u8, a_u8, key16) }
      end
    },
  },
  "fir" => {
    "5-tap FIR filter",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.fir(fir_dst, fir_src, coeff) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.fir(fir_dst, fir_src, coeff) }
      end
    },
  },
  "min" => {
    "find minimum (f32)",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.min(a_f32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.min(a_f32) }
      end
    },
  },
  "add_f64" => {
    "Float64 addition",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.add(dst_f64, a_f64, b_f64) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.add(dst_f64, a_f64, b_f64) }
      end
    },
  },
  "mul_f64" => {
    "Float64 multiplication",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.mul(dst_f64, a_f64, b_f64) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.mul(dst_f64, a_f64, b_f64) }
      end
    },
  },
  "fma_f64" => {
    "Float64 fused multiply-add",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.fma(dst_f64, a_f64, b_f64, c_f64) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.fma(dst_f64, a_f64, b_f64, c_f64) }
      end
    },
  },
  "sum_f64" => {
    "Float64 horizontal sum",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.sum(a_f64) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.sum(a_f64) }
      end
    },
  },
  "dot_f64" => {
    "Float64 dot product",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.dot(a_f64, b_f64) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.dot(a_f64, b_f64) }
      end
    },
  },
  "add_u64" => {
    "UInt64 addition",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.add(dst_u64, a_u64, b_u64) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.add(dst_u64, a_u64, b_u64) }
      end
    },
  },
  "xor_u64" => {
    "UInt64 bitwise XOR",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.xor(dst_u64, a_u64, b_u64) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.xor(dst_u64, a_u64, b_u64) }
      end
    },
  },
  "add_u32" => {
    "UInt32 addition",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.add(dst_u32, a_u32, b_u32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.add(dst_u32, a_u32, b_u32) }
      end
    },
  },
  "add_u16" => {
    "UInt16 addition",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.add(dst_u16, a_u16, b_u16) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.add(dst_u16, a_u16, b_u16) }
      end
    },
  },
  "add_u8" => {
    "UInt8 addition",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.add(dst_u8, a_u8, b_u8) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.add(dst_u8, a_u8, b_u8) }
      end
    },
  },
  "add_i32" => {
    "Int32 addition",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.add(dst_i32, a_i32, b_i32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.add(dst_i32, a_i32, b_i32) }
      end
    },
  },
  "sum_u32" => {
    "UInt32 horizontal sum",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.sum(a_u32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.sum(a_u32) }
      end
    },
  },
  "max_i32" => {
    "Int32 find maximum (signed)",
    ->(x : Benchmark::IPS::Job) {
      x.report("Scalar") { scalar.max(a_i32) }
      implementations.each do |simd|
        x.report(simd.class.name.split("::").last) { simd.max(a_i32) }
      end
    },
  },
}

# Parse command line options
selected_tests = [] of String
list_only = false

OptionParser.parse do |parser|
  parser.banner = "Usage: benchmark [options]"

  parser.on("-l", "--list", "List available benchmarks") do
    list_only = true
  end

  parser.on("-t NAME", "--test=NAME", "Run specific benchmark (can be repeated)") do |name|
    selected_tests << name
  end

  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end

  parser.invalid_option do |flag|
    STDERR.puts "ERROR: #{flag} is not a valid option."
    STDERR.puts parser
    exit(1)
  end
end

# List benchmarks and exit if requested
if list_only
  puts "Available benchmarks:"
  benchmarks.each do |name, (desc, _)|
    puts "  #{name.ljust(15)} - #{desc}"
  end
  exit
end

# Validate selected tests
selected_tests.each do |name|
  unless benchmarks.has_key?(name)
    STDERR.puts "ERROR: Unknown benchmark '#{name}'"
    STDERR.puts "Use --list to see available benchmarks"
    exit(1)
  end
end

# If no tests selected, run all
tests_to_run = selected_tests.empty? ? benchmarks.keys.to_a : selected_tests

# Print header
puts "=" * 70
puts "SIMD Benchmark"
puts "=" * 70
puts ""
puts "System Information:"
puts "  SIMD Support: #{SIMD.supported_instruction_sets}"
puts "  Available Implementations: #{check_implementations.map(&.class.name).join(", ")}"
{% if flag?(:x86_64) %}
  if SIMD.supported_instruction_sets.avx512?
    puts "  AVX-512 Subsets: #{SIMD.avx512_subsets}"
  end
{% end %}
puts ""

if selected_tests.empty?
  puts "Running all #{tests_to_run.size} benchmarks..."
else
  puts "Running #{tests_to_run.size} selected benchmark(s): #{tests_to_run.join(", ")}"
end
puts ""

# Run benchmarks
tests_to_run.each_with_index do |name, idx|
  desc, bench_proc = benchmarks[name]

  puts "-" * 70
  puts "#{name} (#{desc}) - #{SIZE} elements"
  puts "-" * 70

  Benchmark.ips do |x|
    bench_proc.call(x)
  end

  puts "" if idx < tests_to_run.size - 1
end

puts ""
puts "=" * 70
puts "Benchmark complete!"
puts "=" * 70
