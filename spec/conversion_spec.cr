require "./spec_helper"

describe "SIMD conversion operations" do
  describe "convert Int16 to Float32" do
    it "scalar implementation" do
      simd = SIMD.scalar
      src = Slice[-100_i16, 0_i16, 100_i16, 32767_i16]
      dst = Slice(Float32).new(src.size)

      simd.convert(dst, src)

      dst[0].should eq(-100.0_f32)
      dst[1].should eq 0.0_f32
      dst[2].should eq 100.0_f32
      dst[3].should eq 32767.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        src = Slice[-100_i16, 0_i16, 100_i16, 32767_i16]
        dst = Slice(Float32).new(src.size)

        simd.convert(dst, src)

        dst[0].should eq(-100.0_f32)
        dst[1].should eq 0.0_f32
        dst[2].should eq 100.0_f32
        dst[3].should eq 32767.0_f32
      end
    end
  end

  describe "convert UInt8 to Float32 with scale" do
    it "scalar implementation" do
      simd = SIMD.scalar
      src = Slice[0_u8, 128_u8, 255_u8, 64_u8]
      dst = Slice(Float32).new(src.size)

      simd.convert(dst, src, 1.0_f32 / 255.0_f32)

      approx_eq(dst[0], 0.0_f32).should be_true
      approx_eq(dst[1], 128.0_f32 / 255.0_f32).should be_true
      approx_eq(dst[2], 1.0_f32).should be_true
      approx_eq(dst[3], 64.0_f32 / 255.0_f32).should be_true
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        src = Slice[0_u8, 128_u8, 255_u8, 64_u8]
        dst = Slice(Float32).new(src.size)

        simd.convert(dst, src, 1.0_f32 / 255.0_f32)

        approx_eq(dst[0], 0.0_f32).should be_true
        approx_eq(dst[1], 128.0_f32 / 255.0_f32).should be_true
        approx_eq(dst[2], 1.0_f32).should be_true
        approx_eq(dst[3], 64.0_f32 / 255.0_f32).should be_true
      end
    end
  end

  describe "convert Float64 to Float32 (narrowing)" do
    it "scalar implementation" do
      simd = SIMD.scalar
      src = Slice[1.5_f64, 2.5_f64, -3.5_f64, 4.5_f64]
      dst = Slice(Float32).new(src.size)

      simd.convert(dst, src)

      dst[0].should eq 1.5_f32
      dst[1].should eq 2.5_f32
      dst[2].should eq(-3.5_f32)
      dst[3].should eq 4.5_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        src = Slice[1.5_f64, 2.5_f64, -3.5_f64, 4.5_f64]
        dst = Slice(Float32).new(src.size)

        simd.convert(dst, src)

        dst[0].should eq 1.5_f32
        dst[1].should eq 2.5_f32
        dst[2].should eq(-3.5_f32)
        dst[3].should eq 4.5_f32
      end
    end
  end

  describe "convert Float32 to Float64 (widening)" do
    it "scalar implementation" do
      simd = SIMD.scalar
      src = Slice[1.5_f32, 2.5_f32, -3.5_f32, 4.5_f32]
      dst = Slice(Float64).new(src.size)

      simd.convert(dst, src)

      dst[0].should eq 1.5_f64
      dst[1].should eq 2.5_f64
      dst[2].should eq(-3.5_f64)
      dst[3].should eq 4.5_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        src = Slice[1.5_f32, 2.5_f32, -3.5_f32, 4.5_f32]
        dst = Slice(Float64).new(src.size)

        simd.convert(dst, src)

        dst[0].should eq 1.5_f64
        dst[1].should eq 2.5_f64
        dst[2].should eq(-3.5_f64)
        dst[3].should eq 4.5_f64
      end
    end
  end

  describe "convert Int32 to Float64" do
    it "scalar implementation" do
      simd = SIMD.scalar
      src = Slice[-100_i32, 0_i32, 100_i32, Int32::MAX]
      dst = Slice(Float64).new(src.size)

      simd.convert(dst, src)

      dst[0].should eq(-100.0_f64)
      dst[1].should eq 0.0_f64
      dst[2].should eq 100.0_f64
      dst[3].should eq Int32::MAX.to_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        src = Slice[-100_i32, 0_i32, 100_i32, Int32::MAX]
        dst = Slice(Float64).new(src.size)

        simd.convert(dst, src)

        dst[0].should eq(-100.0_f64)
        dst[1].should eq 0.0_f64
        dst[2].should eq 100.0_f64
        dst[3].should eq Int32::MAX.to_f64
      end
    end
  end

  describe "convert Int16 to Float64" do
    it "scalar implementation" do
      simd = SIMD.scalar
      src = Slice[-100_i16, 0_i16, 100_i16, Int16::MAX]
      dst = Slice(Float64).new(src.size)

      simd.convert(dst, src)

      dst[0].should eq(-100.0_f64)
      dst[1].should eq 0.0_f64
      dst[2].should eq 100.0_f64
      dst[3].should eq Int16::MAX.to_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        src = Slice[-100_i16, 0_i16, 100_i16, Int16::MAX]
        dst = Slice(Float64).new(src.size)

        simd.convert(dst, src)

        dst[0].should eq(-100.0_f64)
        dst[1].should eq 0.0_f64
        dst[2].should eq 100.0_f64
        dst[3].should eq Int16::MAX.to_f64
      end
    end
  end
end
