require "../src/simd"
require "benchmark"
require "option_parser"

module SIMD
  module Benchmarker
    record ImplEntry, name : String, impl : SIMD::Base

    record Case, name : String, desc : String, run : Proc(Benchmark::IPS::Job, Array(ImplEntry), Nil)

    class Context
      getter size : Int32

      getter a_f32 : Slice(Float32)
      getter b_f32 : Slice(Float32)
      getter c_f32 : Slice(Float32)
      getter dst_f32 : Slice(Float32)
      getter pos_f32 : Slice(Float32)

      getter a_f64 : Slice(Float64)
      getter b_f64 : Slice(Float64)
      getter c_f64 : Slice(Float64)
      getter dst_f64 : Slice(Float64)
      getter pos_f64 : Slice(Float64)

      getter a_u64 : Slice(UInt64)
      getter b_u64 : Slice(UInt64)
      getter dst_u64 : Slice(UInt64)
      getter a_i64 : Slice(Int64)
      getter b_i64 : Slice(Int64)
      getter dst_i64 : Slice(Int64)

      getter a_u32 : Slice(UInt32)
      getter b_u32 : Slice(UInt32)
      getter dst_u32 : Slice(UInt32)
      getter a_i32 : Slice(Int32)
      getter b_i32 : Slice(Int32)
      getter dst_i32 : Slice(Int32)

      getter a_u16 : Slice(UInt16)
      getter b_u16 : Slice(UInt16)
      getter dst_u16 : Slice(UInt16)
      getter a_i16 : Slice(Int16)
      getter b_i16 : Slice(Int16)
      getter dst_i16 : Slice(Int16)

      getter a_u8 : Slice(UInt8)
      getter b_u8 : Slice(UInt8)
      getter dst_u8 : Slice(UInt8)
      getter a_i8 : Slice(Int8)
      getter b_i8 : Slice(Int8)
      getter dst_i8 : Slice(Int8)

      getter mask_in : Slice(UInt8)
      getter mask_out : Slice(UInt8)

      getter key16 : StaticArray(UInt8, 16)

      getter fir_coeff_f32 : Slice(Float32)
      getter fir_src_f32 : Slice(Float32)
      getter fir_dst_f32 : Slice(Float32)

      getter fir_coeff_f64 : Slice(Float64)
      getter fir_src_f64 : Slice(Float64)
      getter fir_dst_f64 : Slice(Float64)

      # Sinks to prevent the compiler from discarding pure computations.
      property sink_u64 : UInt64 = 0_u64
      property sink_f32 : Float32 = 0.0_f32
      property sink_f64 : Float64 = 0.0_f64

      def initialize(@size : Int32)
        @a_f32 = Slice(Float32).new(size) { |i| (i % 100).to_f32 * 0.01_f32 }
        @b_f32 = Slice(Float32).new(size) { |i| ((i % 99) + 1).to_f32 * 0.01_f32 }
        @c_f32 = Slice(Float32).new(size) { |i| ((i + 25) % 100).to_f32 * 0.01_f32 }
        @dst_f32 = Slice(Float32).new(size)
        @pos_f32 = Slice(Float32).new(size) { |i| ((i % 100) + 1).to_f32 * 0.01_f32 }

        @a_f64 = Slice(Float64).new(size) { |i| (i % 100).to_f64 * 0.01_f64 }
        @b_f64 = Slice(Float64).new(size) { |i| ((i % 99) + 1).to_f64 * 0.01_f64 }
        @c_f64 = Slice(Float64).new(size) { |i| ((i + 25) % 100).to_f64 * 0.01_f64 }
        @dst_f64 = Slice(Float64).new(size)
        @pos_f64 = Slice(Float64).new(size) { |i| ((i % 100) + 1).to_f64 * 0.01_f64 }

        @a_u64 = Slice(UInt64).new(size) { |i| i.to_u64 &* 0x123456789ABCDEF0_u64 }
        @b_u64 = Slice(UInt64).new(size) { |i| i.to_u64 &* 0xFEDCBA9876543210_u64 }
        @dst_u64 = Slice(UInt64).new(size)
        @a_i64 = Slice(Int64).new(size) { |i| ((i % 2048) - 1024).to_i64 &* 0x12345_i64 }
        @b_i64 = Slice(Int64).new(size) { |i| ((i % 4096) - 2048).to_i64 &* 0x54321_i64 }
        @dst_i64 = Slice(Int64).new(size)

        @a_u32 = Slice(UInt32).new(size) { |i| i.to_u32 &* 0x12345678_u32 }
        @b_u32 = Slice(UInt32).new(size) { |i| i.to_u32 &* 0x87654321_u32 }
        @dst_u32 = Slice(UInt32).new(size)
        @a_i32 = Slice(Int32).new(size) { |i| ((i % 1_000_000) - 500_000).to_i32 }
        @b_i32 = Slice(Int32).new(size) { |i| ((i % 2_000_000) - 1_000_000).to_i32 }
        @dst_i32 = Slice(Int32).new(size)

        @a_u16 = Slice(UInt16).new(size) { |i| ((i &* 0x1234) & 0xFFFF).to_u16 }
        @b_u16 = Slice(UInt16).new(size) { |i| ((i &* 0x4321) & 0xFFFF).to_u16 }
        @dst_u16 = Slice(UInt16).new(size)
        @a_i16 = Slice(Int16).new(size) { |i| ((i % 65536) - 32768).to_i16 }
        @b_i16 = Slice(Int16).new(size) { |i| (((i + 12345) % 65536) - 32768).to_i16 }
        @dst_i16 = Slice(Int16).new(size)

        @a_u8 = Slice(UInt8).new(size) { |i| (i % 256).to_u8 }
        @b_u8 = Slice(UInt8).new(size) { |i| ((i + 128) % 256).to_u8 }
        @dst_u8 = Slice(UInt8).new(size)
        @a_i8 = Slice(Int8).new(size) { |i| ((i % 256) - 128).to_i8 }
        @b_i8 = Slice(Int8).new(size) { |i| (((i + 77) % 256) - 128).to_i8 }
        @dst_i8 = Slice(Int8).new(size)

        @mask_in = Slice(UInt8).new(size) { |i| (i & 3) == 0 ? 0_u8 : 1_u8 }
        @mask_out = Slice(UInt8).new(size)

        @key16 = StaticArray(UInt8, 16).new { |idx| ((idx * 17 + 5) % 256).to_u8 }

        @fir_coeff_f32 = Slice[0.1_f32, 0.2_f32, 0.4_f32, 0.2_f32, 0.1_f32]
        @fir_src_f32 = Slice(Float32).new(size + 4) { |idx| ((idx % 100) + 1).to_f32 * 0.01_f32 }
        @fir_dst_f32 = Slice(Float32).new(size)

        @fir_coeff_f64 = Slice[0.1_f64, 0.2_f64, 0.4_f64, 0.2_f64, 0.1_f64]
        @fir_src_f64 = Slice(Float64).new(size + 4) { |idx| ((idx % 100) + 1).to_f64 * 0.01_f64 }
        @fir_dst_f64 = Slice(Float64).new(size)
      end
    end

    struct Options
      property? list : Bool = false
      property only : Array(String) = [] of String
      property except : Array(String) = [] of String
      property impl_only : Array(String) = [] of String
      property impl_except : Array(String) = [] of String
      property size : Int32 = 4096
      property warmup_seconds : Float64 = 0.2
      property time_seconds : Float64 = 0.5
    end

    def self.check_implementations : Array(SIMD::Base)
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

    def self.implementations : Array(ImplEntry)
      impls = [ImplEntry.new("Scalar", SIMD.scalar)]
      check_implementations.each do |impl|
        impls << ImplEntry.new(impl.class.name.split("::").last, impl)
      end
      impls
    end

    def self.report(job : Benchmark::IPS::Job, impls : Array(ImplEntry), &block : SIMD::Base ->) : Nil
      impls.each do |entry|
        job.report(entry.name) { block.call(entry.impl) }
      end
    end

    private def self.glob_to_regex(pattern : String) : Regex
      source = String.build do |io|
        io << '^'
        pattern.each_char do |char|
          case char
          when '*'
            io << ".*"
          when '?'
            io << '.'
          else
            io << Regex.escape(char.to_s)
          end
        end
        io << '$'
      end
      Regex.new(source)
    end

    def self.matches?(pattern : String, name : String) : Bool
      return true if pattern == name
      return true if name.starts_with?(pattern)

      if pattern.starts_with?('/') && pattern.ends_with?('/') && pattern.size > 1
        return Regex.new(pattern[1, pattern.size - 2]).matches?(name)
      end

      if pattern.includes?('*') || pattern.includes?('?')
        return glob_to_regex(pattern).matches?(name)
      end

      false
    end

    def self.select_cases(cases : Array(Case), options : Options) : Array(Case)
      selected = cases
      unless options.only.empty?
        selected = selected.select { |bench_case| options.only.any? { |pattern| matches?(pattern, bench_case.name) } }
      end
      unless options.except.empty?
        selected = selected.reject { |bench_case| options.except.any? { |pattern| matches?(pattern, bench_case.name) } }
      end
      selected
    end

    def self.select_impls(impls : Array(ImplEntry), options : Options) : Array(ImplEntry)
      selected = impls
      unless options.impl_only.empty?
        selected = selected.select { |impl| options.impl_only.any? { |pattern| matches?(pattern, impl.name) } }
      end
      unless options.impl_except.empty?
        selected = selected.reject { |impl| options.impl_except.any? { |pattern| matches?(pattern, impl.name) } }
      end
      selected
    end

    def self.build_cases(ctx : Context) : Array(Case)
      cases = [] of Case

      # ------------------------------------------------------------
      # Float32
      # ------------------------------------------------------------
      cases << Case.new("f32/add", "dst = a + b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.add(ctx.dst_f32, ctx.a_f32, ctx.b_f32) } })
      cases << Case.new("f32/sub", "dst = a - b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.sub(ctx.dst_f32, ctx.a_f32, ctx.b_f32) } })
      cases << Case.new("f32/mul", "dst = a * b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.mul(ctx.dst_f32, ctx.a_f32, ctx.b_f32) } })
      cases << Case.new("f32/div", "dst = a / b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.div(ctx.dst_f32, ctx.a_f32, ctx.b_f32) } })
      cases << Case.new("f32/fma", "dst = a*b + c", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.fma(ctx.dst_f32, ctx.a_f32, ctx.b_f32, ctx.c_f32) } })
      cases << Case.new("f32/clamp", "dst = clamp(a, lo, hi)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.clamp(ctx.dst_f32, ctx.a_f32, 0.0_f32, 1.0_f32) } })
      cases << Case.new("f32/axpby", "dst = alpha*a + beta*b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.axpby(ctx.dst_f32, ctx.a_f32, ctx.b_f32, 0.75_f32, 0.25_f32) } })

      cases << Case.new("f32/sqrt", "dst = sqrt(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.sqrt(ctx.dst_f32, ctx.pos_f32) } })
      cases << Case.new("f32/rsqrt", "dst = rsqrt(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.rsqrt(ctx.dst_f32, ctx.pos_f32) } })
      cases << Case.new("f32/abs", "dst = abs(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls, &.abs(ctx.dst_f32, ctx.a_f32)) })
      cases << Case.new("f32/neg", "dst = -a", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls, &.neg(ctx.dst_f32, ctx.a_f32)) })
      cases << Case.new("f32/floor", "dst = floor(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.floor(ctx.dst_f32, ctx.a_f32) } })
      cases << Case.new("f32/ceil", "dst = ceil(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls, &.ceil(ctx.dst_f32, ctx.a_f32)) })
      cases << Case.new("f32/round", "dst = round(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.round(ctx.dst_f32, ctx.a_f32) } })

      cases << Case.new("f32/sum", "sum(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
        report(job, impls) do |simd|
          ctx.sink_f32 = simd.sum(ctx.a_f32)
        end
      end)
      cases << Case.new("f32/dot", "dot(a, b)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
        report(job, impls) do |simd|
          ctx.sink_f32 = simd.dot(ctx.a_f32, ctx.b_f32)
        end
      end)
      cases << Case.new("f32/max_reduce", "max(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
        report(job, impls) do |simd|
          ctx.sink_f32 = simd.max(ctx.a_f32)
        end
      end)
      cases << Case.new("f32/min_reduce", "min(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
        report(job, impls) do |simd|
          ctx.sink_f32 = simd.min(ctx.a_f32)
        end
      end)

      cases << Case.new("f32/cmp_gt_mask", "mask = a > b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_gt_mask(ctx.mask_out, ctx.a_f32, ctx.b_f32) } })

      cases << Case.new("f32/fir", "5-tap FIR filter", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.fir(ctx.fir_dst_f32, ctx.fir_src_f32, ctx.fir_coeff_f32) } })

      cases << Case.new("f32/min_elem", "dst[i] = min(a[i], b[i])", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.min(ctx.dst_f32, ctx.a_f32, ctx.b_f32) } })
      cases << Case.new("f32/max_elem", "dst[i] = max(a[i], b[i])", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.max(ctx.dst_f32, ctx.a_f32, ctx.b_f32) } })

      cases << Case.new("f32/cmp_eq", "mask = a == b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_eq(ctx.mask_out, ctx.a_f32, ctx.b_f32) } })
      cases << Case.new("f32/cmp_ne", "mask = a != b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_ne(ctx.mask_out, ctx.a_f32, ctx.b_f32) } })
      cases << Case.new("f32/cmp_lt", "mask = a < b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_lt(ctx.mask_out, ctx.a_f32, ctx.b_f32) } })
      cases << Case.new("f32/cmp_le", "mask = a <= b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_le(ctx.mask_out, ctx.a_f32, ctx.b_f32) } })
      cases << Case.new("f32/cmp_ge", "mask = a >= b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_ge(ctx.mask_out, ctx.a_f32, ctx.b_f32) } })

      # ------------------------------------------------------------
      # Float64
      # ------------------------------------------------------------
      cases << Case.new("f64/add", "dst = a + b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.add(ctx.dst_f64, ctx.a_f64, ctx.b_f64) } })
      cases << Case.new("f64/sub", "dst = a - b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.sub(ctx.dst_f64, ctx.a_f64, ctx.b_f64) } })
      cases << Case.new("f64/mul", "dst = a * b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.mul(ctx.dst_f64, ctx.a_f64, ctx.b_f64) } })
      cases << Case.new("f64/div", "dst = a / b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.div(ctx.dst_f64, ctx.a_f64, ctx.b_f64) } })
      cases << Case.new("f64/fma", "dst = a*b + c", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.fma(ctx.dst_f64, ctx.a_f64, ctx.b_f64, ctx.c_f64) } })
      cases << Case.new("f64/clamp", "dst = clamp(a, lo, hi)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.clamp(ctx.dst_f64, ctx.a_f64, 0.0_f64, 1.0_f64) } })
      cases << Case.new("f64/axpby", "dst = alpha*a + beta*b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.axpby(ctx.dst_f64, ctx.a_f64, ctx.b_f64, 0.75_f64, 0.25_f64) } })

      cases << Case.new("f64/sqrt", "dst = sqrt(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.sqrt(ctx.dst_f64, ctx.pos_f64) } })
      cases << Case.new("f64/rsqrt", "dst = rsqrt(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.rsqrt(ctx.dst_f64, ctx.pos_f64) } })
      cases << Case.new("f64/abs", "dst = abs(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls, &.abs(ctx.dst_f64, ctx.a_f64)) })
      cases << Case.new("f64/neg", "dst = -a", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls, &.neg(ctx.dst_f64, ctx.a_f64)) })
      cases << Case.new("f64/floor", "dst = floor(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.floor(ctx.dst_f64, ctx.a_f64) } })
      cases << Case.new("f64/ceil", "dst = ceil(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls, &.ceil(ctx.dst_f64, ctx.a_f64)) })
      cases << Case.new("f64/round", "dst = round(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.round(ctx.dst_f64, ctx.a_f64) } })

      cases << Case.new("f64/sum", "sum(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
        report(job, impls) do |simd|
          ctx.sink_f64 = simd.sum(ctx.a_f64)
        end
      end)
      cases << Case.new("f64/dot", "dot(a, b)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
        report(job, impls) do |simd|
          ctx.sink_f64 = simd.dot(ctx.a_f64, ctx.b_f64)
        end
      end)
      cases << Case.new("f64/max_reduce", "max(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
        report(job, impls) do |simd|
          ctx.sink_f64 = simd.max(ctx.a_f64)
        end
      end)
      cases << Case.new("f64/min_reduce", "min(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
        report(job, impls) do |simd|
          ctx.sink_f64 = simd.min(ctx.a_f64)
        end
      end)

      cases << Case.new("f64/cmp_gt_mask", "mask = a > b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_gt_mask(ctx.mask_out, ctx.a_f64, ctx.b_f64) } })

      cases << Case.new("f64/fir", "5-tap FIR filter", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.fir(ctx.fir_dst_f64, ctx.fir_src_f64, ctx.fir_coeff_f64) } })

      cases << Case.new("f64/min_elem", "dst[i] = min(a[i], b[i])", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.min(ctx.dst_f64, ctx.a_f64, ctx.b_f64) } })
      cases << Case.new("f64/max_elem", "dst[i] = max(a[i], b[i])", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.max(ctx.dst_f64, ctx.a_f64, ctx.b_f64) } })

      cases << Case.new("f64/cmp_eq", "mask = a == b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_eq(ctx.mask_out, ctx.a_f64, ctx.b_f64) } })
      cases << Case.new("f64/cmp_ne", "mask = a != b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_ne(ctx.mask_out, ctx.a_f64, ctx.b_f64) } })
      cases << Case.new("f64/cmp_lt", "mask = a < b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_lt(ctx.mask_out, ctx.a_f64, ctx.b_f64) } })
      cases << Case.new("f64/cmp_le", "mask = a <= b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_le(ctx.mask_out, ctx.a_f64, ctx.b_f64) } })
      cases << Case.new("f64/cmp_ge", "mask = a >= b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_ge(ctx.mask_out, ctx.a_f64, ctx.b_f64) } })

      # ------------------------------------------------------------
      # Integer arithmetic / bitwise / reductions
      # ------------------------------------------------------------
      {% begin %}
        {% int_sets = [
             {key: "u64", signed: false, a: "a_u64", b: "b_u64", dst: "dst_u64"},
             {key: "i64", signed: true, a: "a_i64", b: "b_i64", dst: "dst_i64"},
             {key: "u32", signed: false, a: "a_u32", b: "b_u32", dst: "dst_u32"},
             {key: "i32", signed: true, a: "a_i32", b: "b_i32", dst: "dst_i32"},
             {key: "u16", signed: false, a: "a_u16", b: "b_u16", dst: "dst_u16"},
             {key: "i16", signed: true, a: "a_i16", b: "b_i16", dst: "dst_i16"},
             {key: "u8", signed: false, a: "a_u8", b: "b_u8", dst: "dst_u8"},
             {key: "i8", signed: true, a: "a_i8", b: "b_i8", dst: "dst_i8"},
           ] %}

        {% for s in int_sets %}
        cases << Case.new({{ s[:key] }} + "/add", "dst = a + b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.add(ctx.{{s[:dst].id}}, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })
        cases << Case.new({{ s[:key] }} + "/sub", "dst = a - b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.sub(ctx.{{s[:dst].id}}, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })
        cases << Case.new({{ s[:key] }} + "/mul", "dst = a * b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.mul(ctx.{{s[:dst].id}}, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })

        {% if s[:key] == "u64" %}
          cases << Case.new("u64/clamp", "dst = clamp(a, lo, hi)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.clamp(ctx.dst_u64, ctx.a_u64, 0x100_u64, 0xFFFF_FFFF_FFFF_u64) } })
          cases << Case.new("u64/bswap", "byte swap", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.bswap(ctx.dst_u64, ctx.a_u64) } })
        {% elsif s[:key] == "i64" %}
          cases << Case.new("i64/clamp", "dst = clamp(a, lo, hi)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.clamp(ctx.dst_i64, ctx.a_i64, -10_000_i64, 10_000_i64) } })
          cases << Case.new("i64/bswap", "byte swap", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.bswap(ctx.dst_i64, ctx.a_i64) } })
        {% elsif s[:key] == "u32" %}
          cases << Case.new("u32/clamp", "dst = clamp(a, lo, hi)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.clamp(ctx.dst_u32, ctx.a_u32, 0x100_u32, 0xFFFF_FFFF_u32) } })
          cases << Case.new("u32/bswap", "byte swap", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.bswap(ctx.dst_u32, ctx.a_u32) } })
        {% elsif s[:key] == "i32" %}
          cases << Case.new("i32/clamp", "dst = clamp(a, lo, hi)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.clamp(ctx.dst_i32, ctx.a_i32, -10_000_i32, 10_000_i32) } })
          cases << Case.new("i32/bswap", "byte swap", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.bswap(ctx.dst_i32, ctx.a_i32) } })
        {% elsif s[:key] == "u16" %}
          cases << Case.new("u16/clamp", "dst = clamp(a, lo, hi)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.clamp(ctx.dst_u16, ctx.a_u16, 0x10_u16, 0xFF00_u16) } })
          cases << Case.new("u16/bswap", "byte swap", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.bswap(ctx.dst_u16, ctx.a_u16) } })
        {% elsif s[:key] == "i16" %}
          cases << Case.new("i16/clamp", "dst = clamp(a, lo, hi)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.clamp(ctx.dst_i16, ctx.a_i16, -10_000_i16, 10_000_i16) } })
          cases << Case.new("i16/bswap", "byte swap", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.bswap(ctx.dst_i16, ctx.a_i16) } })
        {% elsif s[:key] == "u8" %}
          cases << Case.new("u8/clamp", "dst = clamp(a, lo, hi)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.clamp(ctx.dst_u8, ctx.a_u8, 0x10_u8, 0xF0_u8) } })
        {% elsif s[:key] == "i8" %}
          cases << Case.new("i8/clamp", "dst = clamp(a, lo, hi)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.clamp(ctx.dst_i8, ctx.a_i8, -100_i8, 100_i8) } })
        {% end %}

        cases << Case.new({{ s[:key] }} + "/sum", "sum(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
          report(job, impls) do |simd|
            ctx.sink_u64 = ctx.sink_u64 &+ simd.sum(ctx.{{s[:a].id}}).hash
          end
        end)
        cases << Case.new({{ s[:key] }} + "/max_reduce", "max(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
          report(job, impls) do |simd|
            ctx.sink_u64 = ctx.sink_u64 &+ simd.max(ctx.{{s[:a].id}}).hash
          end
        end)
        cases << Case.new({{ s[:key] }} + "/min_reduce", "min(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
          report(job, impls) do |simd|
            ctx.sink_u64 = ctx.sink_u64 &+ simd.min(ctx.{{s[:a].id}}).hash
          end
        end)

        cases << Case.new({{ s[:key] }} + "/bitwise_and", "dst = a & b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.bitwise_and(ctx.{{s[:dst].id}}, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })
        cases << Case.new({{ s[:key] }} + "/bitwise_or", "dst = a | b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.bitwise_or(ctx.{{s[:dst].id}}, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })
        cases << Case.new({{ s[:key] }} + "/xor", "dst = a ^ b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.xor(ctx.{{s[:dst].id}}, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })

        cases << Case.new({{ s[:key] }} + "/cmp_gt_mask", "mask = a > b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_gt_mask(ctx.mask_out, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })

        cases << Case.new({{ s[:key] }} + "/abs", "dst = abs(a)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.abs(ctx.{{s[:dst].id}}, ctx.{{s[:a].id}}) } })
        {% if s[:signed] %}
          cases << Case.new({{ s[:key] }} + "/neg", "dst = -a", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.neg(ctx.{{s[:dst].id}}, ctx.{{s[:a].id}}) } })
        {% end %}

        cases << Case.new({{ s[:key] }} + "/min_elem", "dst[i] = min(a[i], b[i])", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.min(ctx.{{s[:dst].id}}, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })
        cases << Case.new({{ s[:key] }} + "/max_elem", "dst[i] = max(a[i], b[i])", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.max(ctx.{{s[:dst].id}}, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })

        cases << Case.new({{ s[:key] }} + "/cmp_eq", "mask = a == b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_eq(ctx.mask_out, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })
        cases << Case.new({{ s[:key] }} + "/cmp_ne", "mask = a != b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_ne(ctx.mask_out, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })
        cases << Case.new({{ s[:key] }} + "/cmp_lt", "mask = a < b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_lt(ctx.mask_out, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })
        cases << Case.new({{ s[:key] }} + "/cmp_le", "mask = a <= b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_le(ctx.mask_out, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })
        cases << Case.new({{ s[:key] }} + "/cmp_ge", "mask = a >= b", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.cmp_ge(ctx.mask_out, ctx.{{s[:a].id}}, ctx.{{s[:b].id}}) } })
      {% end %}
      {% end %}

      cases << Case.new("u8/popcount", "total bits set", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) do
        report(job, impls) do |simd|
          ctx.sink_u64 = simd.popcount(ctx.a_u8)
        end
      end)

      cases << Case.new("u8/blend", "dst = mask ? t : f", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.blend(ctx.dst_u8, ctx.a_u8, ctx.b_u8, ctx.mask_in) } })

      # ------------------------------------------------------------
      # Memory / conversion / block ops
      # ------------------------------------------------------------
      cases << Case.new("u8/copy", "copy bytes", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls, &.copy(ctx.dst_u8, ctx.a_u8)) })
      cases << Case.new("u8/fill", "fill bytes", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls, &.fill(ctx.dst_u8, 171_u8)) })

      i16_src = Slice(Int16).new(ctx.size) { |i| ((i % 65536) - 32768).to_i16 }
      i32_src = Slice(Int32).new(ctx.size) { |i| ((i % 1_000_000) - 500_000).to_i32 }

      cases << Case.new("f32/convert_i16", "Int16 -> Float32", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.convert(ctx.dst_f32, i16_src) } })
      cases << Case.new("f32/convert_u8_scale", "UInt8 -> Float32 (scale)", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.convert(ctx.dst_f32, ctx.a_u8, 0.5_f32) } })
      cases << Case.new("f64/convert_i32", "Int32 -> Float64", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.convert(ctx.dst_f64, i32_src) } })
      cases << Case.new("f64/convert_i16", "Int16 -> Float64", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.convert(ctx.dst_f64, i16_src) } })
      cases << Case.new("f64/convert_f32", "Float32 -> Float64", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.convert(ctx.dst_f64, ctx.a_f32) } })
      cases << Case.new("f32/convert_f64", "Float64 -> Float32", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.convert(ctx.dst_f32, ctx.a_f64) } })

      cases << Case.new("u8/xor_block16", "xor with 16-byte key", ->(job : Benchmark::IPS::Job, impls : Array(ImplEntry)) { report(job, impls) { |simd| simd.xor_block16(ctx.dst_u8, ctx.a_u8, ctx.key16) } })

      cases.sort_by!(&.name)
      cases
    end

    def self.parse_options(argv : Array(String)) : Options
      options = Options.new

      OptionParser.parse(argv) do |parser|
        parser.banner = "Usage: benchmarker [options]"

        parser.on("-l", "--list", "List available benchmarks") { options.list = true }
        parser.on("-t NAME", "--test=NAME", "Select benchmark (repeatable, matches by prefix/glob/regex)") { |name| options.only << name }
        parser.on("--only=NAME", "Select benchmark (repeatable, matches by prefix/glob/regex)") { |name| options.only << name }
        parser.on("--except=NAME", "Exclude benchmark (repeatable, matches by prefix/glob/regex)") { |name| options.except << name }

        parser.on("--impl=NAME", "Select implementation (repeatable, matches by prefix/glob/regex)") { |name| options.impl_only << name }
        parser.on("--except-impl=NAME", "Exclude implementation (repeatable, matches by prefix/glob/regex)") { |name| options.impl_except << name }

        parser.on("--size=N", "Vector size (default: #{options.size})") { |num| options.size = num.to_i }
        parser.on("--warmup=SECONDS", "Warmup seconds (default: #{options.warmup_seconds})") { |sec| options.warmup_seconds = sec.to_f }
        parser.on("--time=SECONDS", "Calculation seconds (default: #{options.time_seconds})") { |sec| options.time_seconds = sec.to_f }

        parser.on("-h", "--help", "Show help") do
          puts parser
          exit
        end
      end

      options
    end

    def self.run(argv = ARGV) : Nil
      options = parse_options(argv)
      ctx = Context.new(options.size)
      cases = build_cases(ctx)

      if options.list?
        begin
          cases.each do |bench_case|
            puts "#{bench_case.name}\t#{bench_case.desc}"
          end
        rescue IO::Error
          # ignore broken pipe when output is truncated (e.g. piped to `head`)
        end
        return
      end

      selected = select_cases(cases, options)
      if selected.empty?
        STDERR.puts "No benchmarks selected (use --list)"
        return
      end

      impls = select_impls(implementations, options)
      if impls.empty?
        STDERR.puts "No implementations selected"
        return
      end
      puts "Vector size: #{ctx.size}"
      puts "Implementations: #{impls.map(&.name).join(", ")}"

      selected.each do |bench_case|
        puts "\n#{bench_case.name}\t#{bench_case.desc}"
        Benchmark.ips(calculation: options.time_seconds.seconds, warmup: options.warmup_seconds.seconds, interactive: false) do |job|
          bench_case.run.call(job, impls)
        end
      end
    end
  end
end

SIMD::Benchmarker.run
