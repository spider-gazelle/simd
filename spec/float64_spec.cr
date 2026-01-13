require "./spec_helper"

describe "SIMD Float64 operations" do
  describe "add" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
      b = Slice[0.5_f64, 1.0_f64, 1.5_f64, 2.0_f64]
      dst = Slice(Float64).new(a.size)

      simd.add(dst, a, b)

      dst[0].should eq 1.5_f64
      dst[1].should eq 3.0_f64
      dst[2].should eq 4.5_f64
      dst[3].should eq 6.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        b = Slice[0.5_f64, 1.0_f64, 1.5_f64, 2.0_f64]
        dst = Slice(Float64).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 1.5_f64
        dst[1].should eq 3.0_f64
        dst[2].should eq 4.5_f64
        dst[3].should eq 6.0_f64
      end
    end
  end

  describe "sub" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[10.0_f64, 20.0_f64, 30.0_f64, 40.0_f64]
      b = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
      dst = Slice(Float64).new(a.size)

      simd.sub(dst, a, b)

      dst[0].should eq 9.0_f64
      dst[1].should eq 18.0_f64
      dst[2].should eq 27.0_f64
      dst[3].should eq 36.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[10.0_f64, 20.0_f64, 30.0_f64, 40.0_f64]
        b = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        dst = Slice(Float64).new(a.size)

        simd.sub(dst, a, b)

        dst[0].should eq 9.0_f64
        dst[1].should eq 18.0_f64
        dst[2].should eq 27.0_f64
        dst[3].should eq 36.0_f64
      end
    end
  end

  describe "mul" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
      b = Slice[2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64]
      dst = Slice(Float64).new(a.size)

      simd.mul(dst, a, b)

      dst[0].should eq 2.0_f64
      dst[1].should eq 6.0_f64
      dst[2].should eq 12.0_f64
      dst[3].should eq 20.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        b = Slice[2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64]
        dst = Slice(Float64).new(a.size)

        simd.mul(dst, a, b)

        dst[0].should eq 2.0_f64
        dst[1].should eq 6.0_f64
        dst[2].should eq 12.0_f64
        dst[3].should eq 20.0_f64
      end
    end
  end

  describe "fma" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
      b = Slice[2.0_f64, 2.0_f64, 2.0_f64, 2.0_f64]
      c = Slice[0.5_f64, 1.0_f64, 1.5_f64, 2.0_f64]
      dst = Slice(Float64).new(a.size)

      simd.fma(dst, a, b, c)

      # a*b + c
      dst[0].should eq 2.5_f64
      dst[1].should eq 5.0_f64
      dst[2].should eq 7.5_f64
      dst[3].should eq 10.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        b = Slice[2.0_f64, 2.0_f64, 2.0_f64, 2.0_f64]
        c = Slice[0.5_f64, 1.0_f64, 1.5_f64, 2.0_f64]
        dst = Slice(Float64).new(a.size)

        simd.fma(dst, a, b, c)

        dst[0].should eq 2.5_f64
        dst[1].should eq 5.0_f64
        dst[2].should eq 7.5_f64
        dst[3].should eq 10.0_f64
      end
    end
  end

  describe "clamp" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-2.0_f64, 0.5_f64, 1.5_f64, 3.0_f64]
      dst = Slice(Float64).new(a.size)

      simd.clamp(dst, a, 0.0_f64, 1.0_f64)

      dst[0].should eq 0.0_f64
      dst[1].should eq 0.5_f64
      dst[2].should eq 1.0_f64
      dst[3].should eq 1.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-2.0_f64, 0.5_f64, 1.5_f64, 3.0_f64]
        dst = Slice(Float64).new(a.size)

        simd.clamp(dst, a, 0.0_f64, 1.0_f64)

        dst[0].should eq 0.0_f64
        dst[1].should eq 0.5_f64
        dst[2].should eq 1.0_f64
        dst[3].should eq 1.0_f64
      end
    end
  end

  describe "axpby" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
      b = Slice[0.5_f64, 1.0_f64, 1.5_f64, 2.0_f64]
      dst = Slice(Float64).new(a.size)

      simd.axpby(dst, a, b, 2.0_f64, 3.0_f64)

      # 2*a + 3*b
      dst[0].should eq 3.5_f64
      dst[1].should eq 7.0_f64
      dst[2].should eq 10.5_f64
      dst[3].should eq 14.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        b = Slice[0.5_f64, 1.0_f64, 1.5_f64, 2.0_f64]
        dst = Slice(Float64).new(a.size)

        simd.axpby(dst, a, b, 2.0_f64, 3.0_f64)

        dst[0].should eq 3.5_f64
        dst[1].should eq 7.0_f64
        dst[2].should eq 10.5_f64
        dst[3].should eq 14.0_f64
      end
    end
  end

  describe "sum" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
      result = simd.sum(a)
      result.should eq 10.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        result = simd.sum(a)
        result.should eq 10.0_f64
      end
    end
  end

  describe "dot" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
      b = Slice[2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64]
      result = simd.dot(a, b)
      # 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
      result.should eq 40.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        b = Slice[2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64]
        result = simd.dot(a, b)
        result.should eq 40.0_f64
      end
    end
  end

  describe "max" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[3.0_f64, 1.0_f64, 9.0_f64, 2.0_f64]
      result = simd.max(a)
      result.should eq 9.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[3.0_f64, 1.0_f64, 9.0_f64, 2.0_f64]
        result = simd.max(a)
        result.should eq 9.0_f64
      end
    end
  end

  describe "min" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[3.0_f64, 1.0_f64, 9.0_f64, 2.0_f64]
      result = simd.min(a)
      result.should eq 1.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[3.0_f64, 1.0_f64, 9.0_f64, 2.0_f64]
        result = simd.min(a)
        result.should eq 1.0_f64
      end
    end

    it "finds min with negative values" do
      check_implementations.each do |simd|
        a = Slice[3.0_f64, -10.0_f64, 4.0_f64, 1.0_f64]
        result = simd.min(a)
        result.should eq(-10.0_f64)
      end
    end
  end

  describe "cmp_gt_mask" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f64, 3.0_f64, 2.0_f64, 4.0_f64]
      b = Slice[2.0_f64, 2.0_f64, 2.0_f64, 2.0_f64]
      mask = Slice(UInt8).new(a.size)

      simd.cmp_gt_mask(mask, a, b)

      mask[0].should eq 0x00_u8
      mask[1].should eq 0xFF_u8
      mask[2].should eq 0x00_u8
      mask[3].should eq 0xFF_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f64, 3.0_f64, 2.0_f64, 4.0_f64]
        b = Slice[2.0_f64, 2.0_f64, 2.0_f64, 2.0_f64]
        mask = Slice(UInt8).new(a.size)

        simd.cmp_gt_mask(mask, a, b)

        mask[0].should eq 0x00_u8
        mask[1].should eq 0xFF_u8
        mask[2].should eq 0x00_u8
        mask[3].should eq 0xFF_u8
      end
    end
  end

  describe "blend" do
    it "scalar implementation" do
      simd = SIMD.scalar
      t = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
      f = Slice[10.0_f64, 20.0_f64, 30.0_f64, 40.0_f64]
      mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
      dst = Slice(Float64).new(t.size)

      simd.blend(dst, t, f, mask)

      dst[0].should eq 1.0_f64
      dst[1].should eq 20.0_f64
      dst[2].should eq 3.0_f64
      dst[3].should eq 40.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        t = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        f = Slice[10.0_f64, 20.0_f64, 30.0_f64, 40.0_f64]
        mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
        dst = Slice(Float64).new(t.size)

        simd.blend(dst, t, f, mask)

        dst[0].should eq 1.0_f64
        dst[1].should eq 20.0_f64
        dst[2].should eq 3.0_f64
        dst[3].should eq 40.0_f64
      end
    end
  end

  describe "compress" do
    it "scalar implementation" do
      simd = SIMD.scalar
      src = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
      mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
      dst = Slice(Float64).new(src.size)

      count = simd.compress(dst, src, mask)

      count.should eq 2
      dst[0].should eq 1.0_f64
      dst[1].should eq 3.0_f64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        src = Slice[1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
        mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
        dst = Slice(Float64).new(src.size)

        count = simd.compress(dst, src, mask)

        count.should eq 2
        dst[0].should eq 1.0_f64
        dst[1].should eq 3.0_f64
      end
    end
  end

  describe "consistency across sizes" do
    it "add produces same results across all implementations" do
      scalar = SIMD.scalar

      [2, 4, 7, 8, 15, 16, 32].each do |size|
        a = Slice(Float64).new(size) { |i| (i + 1).to_f64 }
        b = Slice(Float64).new(size) { |i| (i * 0.5_f64) }
        dst_scalar = Slice(Float64).new(size)

        scalar.add(dst_scalar, a, b)

        check_implementations.each do |simd|
          dst_hw = Slice(Float64).new(size)
          simd.add(dst_hw, a, b)
          approx_eq_slice(dst_scalar, dst_hw).should be_true
        end
      end
    end

    it "sum produces same results across all implementations" do
      scalar = SIMD.scalar

      [2, 4, 7, 8, 15, 16, 32].each do |size|
        a = Slice(Float64).new(size) { |i| (i + 1).to_f64 }

        result_scalar = scalar.sum(a)

        check_implementations.each do |simd|
          result_hw = simd.sum(a)
          approx_eq(result_scalar, result_hw).should be_true
        end
      end
    end
  end
end
