require "./spec_helper"

describe "SIMD Float32 operations" do
  describe "add" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32]
      b = Slice[0.5_f32, 1.0_f32, 1.5_f32, 2.0_f32, 2.5_f32, 3.0_f32, 3.5_f32, 4.0_f32]
      dst = Slice(Float32).new(a.size)

      simd.add(dst, a, b)

      dst[0].should eq 1.5_f32
      dst[1].should eq 3.0_f32
      dst[2].should eq 4.5_f32
      dst[3].should eq 6.0_f32
      dst[4].should eq 7.5_f32
      dst[5].should eq 9.0_f32
      dst[6].should eq 10.5_f32
      dst[7].should eq 12.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32]
        b = Slice[0.5_f32, 1.0_f32, 1.5_f32, 2.0_f32, 2.5_f32, 3.0_f32, 3.5_f32, 4.0_f32]
        dst = Slice(Float32).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 1.5_f32
        dst[1].should eq 3.0_f32
        dst[2].should eq 4.5_f32
        dst[3].should eq 6.0_f32
        dst[4].should eq 7.5_f32
        dst[5].should eq 9.0_f32
        dst[6].should eq 10.5_f32
        dst[7].should eq 12.0_f32
      end
    end
  end

  describe "sub" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32]
      b = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
      dst = Slice(Float32).new(a.size)

      simd.sub(dst, a, b)

      dst[0].should eq 9.0_f32
      dst[1].should eq 18.0_f32
      dst[2].should eq 27.0_f32
      dst[3].should eq 36.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32]
        b = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
        dst = Slice(Float32).new(a.size)

        simd.sub(dst, a, b)

        dst[0].should eq 9.0_f32
        dst[1].should eq 18.0_f32
        dst[2].should eq 27.0_f32
        dst[3].should eq 36.0_f32
      end
    end
  end

  describe "mul" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
      b = Slice[2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32]
      dst = Slice(Float32).new(a.size)

      simd.mul(dst, a, b)

      dst[0].should eq 2.0_f32
      dst[1].should eq 6.0_f32
      dst[2].should eq 12.0_f32
      dst[3].should eq 20.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
        b = Slice[2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32]
        dst = Slice(Float32).new(a.size)

        simd.mul(dst, a, b)

        dst[0].should eq 2.0_f32
        dst[1].should eq 6.0_f32
        dst[2].should eq 12.0_f32
        dst[3].should eq 20.0_f32
      end
    end
  end

  describe "fma" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
      b = Slice[2.0_f32, 2.0_f32, 2.0_f32, 2.0_f32]
      c = Slice[0.5_f32, 1.0_f32, 1.5_f32, 2.0_f32]
      dst = Slice(Float32).new(a.size)

      simd.fma(dst, a, b, c)

      # a*b + c
      dst[0].should eq 2.5_f32
      dst[1].should eq 5.0_f32
      dst[2].should eq 7.5_f32
      dst[3].should eq 10.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
        b = Slice[2.0_f32, 2.0_f32, 2.0_f32, 2.0_f32]
        c = Slice[0.5_f32, 1.0_f32, 1.5_f32, 2.0_f32]
        dst = Slice(Float32).new(a.size)

        simd.fma(dst, a, b, c)

        # a*b + c
        dst[0].should eq 2.5_f32
        dst[1].should eq 5.0_f32
        dst[2].should eq 7.5_f32
        dst[3].should eq 10.0_f32
      end
    end
  end

  describe "clamp" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-2.0_f32, 0.5_f32, 1.5_f32, 3.0_f32]
      dst = Slice(Float32).new(a.size)

      simd.clamp(dst, a, 0.0_f32, 1.0_f32)

      dst[0].should eq 0.0_f32
      dst[1].should eq 0.5_f32
      dst[2].should eq 1.0_f32
      dst[3].should eq 1.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-2.0_f32, 0.5_f32, 1.5_f32, 3.0_f32]
        dst = Slice(Float32).new(a.size)

        simd.clamp(dst, a, 0.0_f32, 1.0_f32)

        dst[0].should eq 0.0_f32
        dst[1].should eq 0.5_f32
        dst[2].should eq 1.0_f32
        dst[3].should eq 1.0_f32
      end
    end
  end

  describe "axpby" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
      b = Slice[0.5_f32, 1.0_f32, 1.5_f32, 2.0_f32]
      dst = Slice(Float32).new(a.size)

      simd.axpby(dst, a, b, 2.0_f32, 3.0_f32)

      # 2*a + 3*b
      dst[0].should eq 3.5_f32
      dst[1].should eq 7.0_f32
      dst[2].should eq 10.5_f32
      dst[3].should eq 14.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
        b = Slice[0.5_f32, 1.0_f32, 1.5_f32, 2.0_f32]
        dst = Slice(Float32).new(a.size)

        simd.axpby(dst, a, b, 2.0_f32, 3.0_f32)

        # 2*a + 3*b
        dst[0].should eq 3.5_f32
        dst[1].should eq 7.0_f32
        dst[2].should eq 10.5_f32
        dst[3].should eq 14.0_f32
      end
    end
  end

  describe "sum" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32]
      result = simd.sum(a)
      result.should eq 36.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32]
        result = simd.sum(a)
        result.should eq 36.0_f32
      end
    end
  end

  describe "dot" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
      b = Slice[2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32]
      result = simd.dot(a, b)
      # 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
      result.should eq 40.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
        b = Slice[2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32]
        result = simd.dot(a, b)
        # 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        result.should eq 40.0_f32
      end
    end
  end

  describe "max" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[3.0_f32, 1.0_f32, 4.0_f32, 1.0_f32, 5.0_f32, 9.0_f32, 2.0_f32, 6.0_f32]
      result = simd.max(a)
      result.should eq 9.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[3.0_f32, 1.0_f32, 4.0_f32, 1.0_f32, 5.0_f32, 9.0_f32, 2.0_f32, 6.0_f32]
        result = simd.max(a)
        result.should eq 9.0_f32
      end
    end
  end

  describe "min" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[3.0_f32, 1.0_f32, 4.0_f32, 1.0_f32, 5.0_f32, 9.0_f32, 2.0_f32, 6.0_f32]
      result = simd.min(a)
      result.should eq 1.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[3.0_f32, 1.0_f32, 4.0_f32, 1.0_f32, 5.0_f32, 9.0_f32, 2.0_f32, 6.0_f32]
        result = simd.min(a)
        result.should eq 1.0_f32
      end
    end

    it "finds min with negative values" do
      check_implementations.each do |simd|
        a = Slice[3.0_f32, -10.0_f32, 4.0_f32, 1.0_f32]
        result = simd.min(a)
        result.should eq(-10.0_f32)
      end
    end
  end

  describe "cmp_gt_mask" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1.0_f32, 3.0_f32, 2.0_f32, 4.0_f32]
      b = Slice[2.0_f32, 2.0_f32, 2.0_f32, 2.0_f32]
      mask = Slice(UInt8).new(a.size)

      simd.cmp_gt_mask(mask, a, b)

      mask[0].should eq 0x00_u8
      mask[1].should eq 0xFF_u8
      mask[2].should eq 0x00_u8
      mask[3].should eq 0xFF_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1.0_f32, 3.0_f32, 2.0_f32, 4.0_f32]
        b = Slice[2.0_f32, 2.0_f32, 2.0_f32, 2.0_f32]
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
      t = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
      f = Slice[10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32]
      mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
      dst = Slice(Float32).new(t.size)

      simd.blend(dst, t, f, mask)

      dst[0].should eq 1.0_f32
      dst[1].should eq 20.0_f32
      dst[2].should eq 3.0_f32
      dst[3].should eq 40.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        t = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
        f = Slice[10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32]
        mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
        dst = Slice(Float32).new(t.size)

        simd.blend(dst, t, f, mask)

        dst[0].should eq 1.0_f32
        dst[1].should eq 20.0_f32
        dst[2].should eq 3.0_f32
        dst[3].should eq 40.0_f32
      end
    end
  end

  describe "compress" do
    it "scalar implementation" do
      simd = SIMD.scalar
      src = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
      mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
      dst = Slice(Float32).new(src.size)

      count = simd.compress(dst, src, mask)

      count.should eq 2
      dst[0].should eq 1.0_f32
      dst[1].should eq 3.0_f32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        src = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
        mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
        dst = Slice(Float32).new(src.size)

        count = simd.compress(dst, src, mask)

        count.should eq 2
        dst[0].should eq 1.0_f32
        dst[1].should eq 3.0_f32
      end
    end
  end

  describe "fir" do
    it "scalar implementation" do
      simd = SIMD.scalar
      # Simple 3-tap moving average
      coeff = Slice[1.0_f32 / 3.0_f32, 1.0_f32 / 3.0_f32, 1.0_f32 / 3.0_f32]
      src = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32]
      dst = Slice(Float32).new(4) # src.size - coeff.size + 1

      simd.fir(dst, src, coeff)

      # (1+2+3)/3 = 2, (2+3+4)/3 = 3, (3+4+5)/3 = 4, (4+5+6)/3 = 5
      approx_eq(dst[0], 2.0_f32).should be_true
      approx_eq(dst[1], 3.0_f32).should be_true
      approx_eq(dst[2], 4.0_f32).should be_true
      approx_eq(dst[3], 5.0_f32).should be_true
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        # Simple 3-tap moving average
        coeff = Slice[1.0_f32 / 3.0_f32, 1.0_f32 / 3.0_f32, 1.0_f32 / 3.0_f32]
        src = Slice[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32]
        dst = Slice(Float32).new(4) # src.size - coeff.size + 1

        simd.fir(dst, src, coeff)

        # (1+2+3)/3 = 2, (2+3+4)/3 = 3, (3+4+5)/3 = 4, (4+5+6)/3 = 5
        approx_eq(dst[0], 2.0_f32).should be_true
        approx_eq(dst[1], 3.0_f32).should be_true
        approx_eq(dst[2], 4.0_f32).should be_true
        approx_eq(dst[3], 5.0_f32).should be_true
      end
    end
  end

  describe "consistency across sizes" do
    it "add produces same results across all implementations" do
      scalar = SIMD.scalar

      [4, 7, 8, 15, 16, 17, 32, 100].each do |size|
        a = Slice(Float32).new(size) { |i| (i + 1).to_f32 }
        b = Slice(Float32).new(size) { |i| (i * 0.5_f32) }
        dst_scalar = Slice(Float32).new(size)

        scalar.add(dst_scalar, a, b)

        check_implementations.each do |simd|
          dst_hw = Slice(Float32).new(size)
          simd.add(dst_hw, a, b)
          approx_eq_slice(dst_scalar, dst_hw).should be_true
        end
      end
    end

    it "dot produces same results across all implementations" do
      scalar = SIMD.scalar

      [4, 7, 8, 15, 16, 17, 32, 100].each do |size|
        a = Slice(Float32).new(size) { |i| (i + 1).to_f32 }
        b = Slice(Float32).new(size) { |i| (i * 0.5_f32) }

        result_scalar = scalar.dot(a, b)

        check_implementations.each do |simd|
          result_hw = simd.dot(a, b)
          approx_eq(result_scalar, result_hw).should be_true
        end
      end
    end

    it "sum produces same results across all implementations" do
      scalar = SIMD.scalar

      [4, 7, 8, 15, 16, 17, 32, 100].each do |size|
        a = Slice(Float32).new(size) { |i| (i + 1).to_f32 }

        result_scalar = scalar.sum(a)

        check_implementations.each do |simd|
          result_hw = simd.sum(a)
          approx_eq(result_scalar, result_hw).should be_true
        end
      end
    end

    it "max produces same results across all implementations" do
      scalar = SIMD.scalar

      [4, 7, 8, 15, 16, 17, 32, 100].each do |size|
        a = Slice(Float32).new(size) { |i| ((i * 7) % 13).to_f32 - 5.0_f32 }

        result_scalar = scalar.max(a)

        check_implementations.each do |simd|
          result_hw = simd.max(a)
          result_scalar.should eq result_hw
        end
      end
    end
  end

  describe "edge cases" do
    it "handles empty slices for max" do
      simd = SIMD.scalar
      expect_raises(ArgumentError) do
        simd.max(Slice(Float32).new(0))
      end
    end

    it "handles mismatched slice lengths" do
      simd = SIMD.scalar
      a = Slice[1.0_f32, 2.0_f32]
      b = Slice[1.0_f32]
      dst = Slice(Float32).new(2)

      expect_raises(ArgumentError) do
        simd.add(dst, a, b)
      end
    end

    it "handles single element slices" do
      check_implementations.each do |simd|
        a = Slice[42.0_f32]
        simd.sum(a).should eq 42.0_f32
        simd.max(a).should eq 42.0_f32
        simd.min(a).should eq 42.0_f32
      end
    end
  end

  describe "realistic usage" do
    it "normalizes audio buffer and computes RMS-ish metric" do
      check_implementations.each do |simd|
        buf = Slice[0.1_f32, 0.4_f32, -0.2_f32, 0.9_f32, -0.5_f32, 0.3_f32, -0.8_f32, 0.6_f32]
        tmp = Slice(Float32).new(buf.size)

        maxv = simd.max(buf).abs
        scale = maxv.zero? ? 1.0_f32 : (1.0_f32 / maxv).to_f32

        scale_arr = Slice(Float32).new(buf.size, scale)
        simd.mul(tmp, buf, scale_arr)
        simd.clamp(tmp, tmp, -1.0_f32, 1.0_f32)

        score = simd.dot(tmp, tmp) / buf.size.to_f32
        score.should be > 0.0_f32
        score.should be <= 1.0_f32
      end
    end
  end
end
