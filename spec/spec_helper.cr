require "spec"
require "../src/simd"

# ensure we're using trace logging
::Log.setup "*", :trace
Spec.before_suite do
  ::Log.setup "*", :trace
end

# Helper to check approximate equality for floats
def approx_eq(a : Float32, b : Float32, epsilon = 0.0001_f32)
  (a - b).abs < epsilon
end

def approx_eq(a : Float64, b : Float64, epsilon = 0.0001_f64)
  (a - b).abs < epsilon
end

def approx_eq_slice(a : Slice(Float32), b : Slice(Float32), epsilon = 0.0001_f32)
  return false unless a.size == b.size
  a.size.times { |i| return false unless approx_eq(a[i], b[i], epsilon) }
  true
end

def approx_eq_slice(a : Slice(Float64), b : Slice(Float64), epsilon = 0.0001_f64)
  return false unless a.size == b.size
  a.size.times { |i| return false unless approx_eq(a[i], b[i], epsilon) }
  true
end

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
