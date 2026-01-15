require "./spec_helper"

describe "SIMD additional operations" do
  describe "math (Float32)" do
    it "div" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[1.0_f32, 4.0_f32, -9.0_f32, 16.0_f32]
        b = Slice[2.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
        dst = Slice(Float32).new(a.size)
        simd.div(dst, a, b)
        dst.should eq Slice[0.5_f32, 2.0_f32, -3.0_f32, 4.0_f32]
      end
    end

    it "sqrt" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[0.0_f32, 1.0_f32, 4.0_f32, 9.0_f32]
        dst = Slice(Float32).new(a.size)
        simd.sqrt(dst, a)
        dst.should eq Slice[0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32]
      end
    end

    it "rsqrt" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[1.0_f32, 4.0_f32, 9.0_f32, 16.0_f32]
        dst = Slice(Float32).new(a.size)
        simd.rsqrt(dst, a)
        approx_eq_slice(dst, Slice[1.0_f32, 0.5_f32, (1.0_f32 / 3.0_f32), 0.25_f32]).should be_true
      end
    end

    it "abs (clears sign bit including -0.0)" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[-0.0_f32, -1.5_f32, 2.5_f32, -3.0_f32]
        dst = Slice(Float32).new(a.size)
        simd.abs(dst, a)
        dst[0].unsafe_as(UInt32).should eq 0_u32
        dst.should eq Slice[0.0_f32, 1.5_f32, 2.5_f32, 3.0_f32]
      end
    end

    it "neg" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[0.0_f32, -1.5_f32, 2.5_f32, -3.0_f32]
        dst = Slice(Float32).new(a.size)
        simd.neg(dst, a)
        dst.should eq Slice[-0.0_f32, 1.5_f32, -2.5_f32, 3.0_f32]
      end
    end

    it "min (element-wise)" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
        b = Slice[0.0_f32, 3.0_f32, 2.0_f32, 5.0_f32]
        dst = Slice(Float32).new(a.size)
        simd.min(dst, a, b)
        dst.should eq Slice[0.0_f32, 2.0_f32, 2.0_f32, 4.0_f32]
      end
    end

    it "max (element-wise)" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
        b = Slice[0.0_f32, 3.0_f32, 2.0_f32, 5.0_f32]
        dst = Slice(Float32).new(a.size)
        simd.max(dst, a, b)
        dst.should eq Slice[1.0_f32, 3.0_f32, 3.0_f32, 5.0_f32]
      end
    end

    it "floor" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[-1.2_f32, -1.5_f32, -1.6_f32, 1.2_f32, 2.5_f32, 3.5_f32]
        dst = Slice(Float32).new(a.size)
        simd.floor(dst, a)
        dst.should eq Slice[-2.0_f32, -2.0_f32, -2.0_f32, 1.0_f32, 2.0_f32, 3.0_f32]
      end
    end

    it "ceil" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[-1.2_f32, -1.5_f32, -1.6_f32, 1.2_f32, 2.5_f32, 3.5_f32]
        dst = Slice(Float32).new(a.size)
        simd.ceil(dst, a)
        dst.should eq Slice[-1.0_f32, -1.0_f32, -1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
      end
    end

    it "round" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[-1.2_f32, -1.5_f32, -1.6_f32, 1.2_f32, 2.5_f32, 3.5_f32]
        dst = Slice(Float32).new(a.size)
        simd.round(dst, a)
        dst.should eq Slice[-1.0_f32, -2.0_f32, -2.0_f32, 1.0_f32, 2.0_f32, 4.0_f32]
      end
    end
  end

  describe "math (Float64)" do
    it "div" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[1.0_f64, 4.0_f64, -9.0_f64, 16.0_f64]
        b = Slice[2.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        dst = Slice(Float64).new(a.size)
        simd.div(dst, a, b)
        dst.should eq Slice[0.5_f64, 2.0_f64, -3.0_f64, 4.0_f64]
      end
    end

    it "sqrt" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[0.0_f64, 1.0_f64, 4.0_f64, 9.0_f64]
        dst = Slice(Float64).new(a.size)
        simd.sqrt(dst, a)
        dst.should eq Slice[0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64]
      end
    end

    it "rsqrt" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[1.0_f64, 4.0_f64, 9.0_f64, 16.0_f64]
        dst = Slice(Float64).new(a.size)
        simd.rsqrt(dst, a)
        approx_eq_slice(dst, Slice[1.0_f64, 0.5_f64, (1.0_f64 / 3.0_f64), 0.25_f64]).should be_true
      end
    end

    it "abs" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[-0.0_f64, -1.5_f64, 2.5_f64, -3.0_f64]
        dst = Slice(Float64).new(a.size)
        simd.abs(dst, a)
        dst[0].unsafe_as(UInt64).should eq 0_u64
        dst.should eq Slice[0.0_f64, 1.5_f64, 2.5_f64, 3.0_f64]
      end
    end

    it "neg" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[0.0_f64, -1.5_f64, 2.5_f64, -3.0_f64]
        dst = Slice(Float64).new(a.size)
        simd.neg(dst, a)
        dst.should eq Slice[-0.0_f64, 1.5_f64, -2.5_f64, 3.0_f64]
      end
    end

    it "min (element-wise)" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        b = Slice[0.0_f64, 3.0_f64, 2.0_f64, 5.0_f64]
        dst = Slice(Float64).new(a.size)
        simd.min(dst, a, b)
        dst.should eq Slice[0.0_f64, 2.0_f64, 2.0_f64, 4.0_f64]
      end
    end

    it "max (element-wise)" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        b = Slice[0.0_f64, 3.0_f64, 2.0_f64, 5.0_f64]
        dst = Slice(Float64).new(a.size)
        simd.max(dst, a, b)
        dst.should eq Slice[1.0_f64, 3.0_f64, 3.0_f64, 5.0_f64]
      end
    end

    it "floor" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[-1.2_f64, -1.5_f64, -1.6_f64, 1.2_f64, 2.5_f64, 3.5_f64]
        dst = Slice(Float64).new(a.size)
        simd.floor(dst, a)
        dst.should eq Slice[-2.0_f64, -2.0_f64, -2.0_f64, 1.0_f64, 2.0_f64, 3.0_f64]
      end
    end

    it "ceil" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[-1.2_f64, -1.5_f64, -1.6_f64, 1.2_f64, 2.5_f64, 3.5_f64]
        dst = Slice(Float64).new(a.size)
        simd.ceil(dst, a)
        dst.should eq Slice[-1.0_f64, -1.0_f64, -1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
      end
    end

    it "round" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[-1.2_f64, -1.5_f64, -1.6_f64, 1.2_f64, 2.5_f64, 3.5_f64]
        dst = Slice(Float64).new(a.size)
        simd.round(dst, a)
        dst.should eq Slice[-1.0_f64, -2.0_f64, -2.0_f64, 1.0_f64, 2.0_f64, 4.0_f64]
      end
    end
  end

  describe "comparisons" do
    it "cmp_eq and cmp_ne" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[1_i16, -2_i16, 3_i16, -4_i16]
        b = Slice[1_i16, -3_i16, 3_i16, 0_i16]
        mask = Slice(UInt8).new(a.size)
        simd.cmp_eq(mask, a, b)
        mask.should eq Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
        simd.cmp_ne(mask, a, b)
        mask.should eq Slice[0x00_u8, 0xFF_u8, 0x00_u8, 0xFF_u8]
      end
    end

    it "cmp_lt/cmp_le/cmp_ge" do
      ([SIMD.scalar] + check_implementations).each do |simd|
        a = Slice[-1_i8, 5_i8, 0_i8, -10_i8]
        b = Slice[-1_i8, 3_i8, 1_i8, 5_i8]
        mask = Slice(UInt8).new(a.size)

        simd.cmp_lt(mask, a, b)
        mask.should eq Slice[0x00_u8, 0x00_u8, 0xFF_u8, 0xFF_u8]

        simd.cmp_le(mask, a, b)
        mask.should eq Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0xFF_u8]

        simd.cmp_ge(mask, a, b)
        mask.should eq Slice[0xFF_u8, 0xFF_u8, 0x00_u8, 0x00_u8]
      end
    end
  end
end
