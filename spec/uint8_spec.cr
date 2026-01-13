require "./spec_helper"

describe "SIMD UInt8 operations" do
  describe "add" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8,
        9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8, 15_u8, 16_u8]
      b = Slice[10_u8, 20_u8, 30_u8, 40_u8, 50_u8, 60_u8, 70_u8, 80_u8,
        90_u8, 100_u8, 110_u8, 120_u8, 130_u8, 140_u8, 150_u8, 160_u8]
      dst = Slice(UInt8).new(a.size)

      simd.add(dst, a, b)

      dst[0].should eq 11_u8
      dst[1].should eq 22_u8
      dst[2].should eq 33_u8
      dst[3].should eq 44_u8
      dst[15].should eq 176_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8,
          9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8, 15_u8, 16_u8]
        b = Slice[10_u8, 20_u8, 30_u8, 40_u8, 50_u8, 60_u8, 70_u8, 80_u8,
          90_u8, 100_u8, 110_u8, 120_u8, 130_u8, 140_u8, 150_u8, 160_u8]
        dst = Slice(UInt8).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 11_u8
        dst[1].should eq 22_u8
        dst[2].should eq 33_u8
        dst[3].should eq 44_u8
        dst[15].should eq 176_u8
      end
    end

    it "handles wrapping overflow" do
      check_implementations.each do |simd|
        a = Slice[UInt8::MAX, UInt8::MAX - 1]
        b = Slice[1_u8, 2_u8]
        dst = Slice(UInt8).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 0_u8
        dst[1].should eq 0_u8
      end
    end
  end

  describe "sub" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[100_u8, 200_u8, 150_u8, 250_u8]
      b = Slice[1_u8, 2_u8, 3_u8, 4_u8]
      dst = Slice(UInt8).new(a.size)

      simd.sub(dst, a, b)

      dst[0].should eq 99_u8
      dst[1].should eq 198_u8
      dst[2].should eq 147_u8
      dst[3].should eq 246_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[100_u8, 200_u8, 150_u8, 250_u8]
        b = Slice[1_u8, 2_u8, 3_u8, 4_u8]
        dst = Slice(UInt8).new(a.size)

        simd.sub(dst, a, b)

        dst[0].should eq 99_u8
        dst[1].should eq 198_u8
        dst[2].should eq 147_u8
        dst[3].should eq 246_u8
      end
    end
  end

  describe "clamp" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0_u8, 50_u8, 150_u8, 200_u8]
      dst = Slice(UInt8).new(a.size)

      simd.clamp(dst, a, 10_u8, 100_u8)

      dst[0].should eq 10_u8
      dst[1].should eq 50_u8
      dst[2].should eq 100_u8
      dst[3].should eq 100_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0_u8, 50_u8, 150_u8, 200_u8]
        dst = Slice(UInt8).new(a.size)

        simd.clamp(dst, a, 10_u8, 100_u8)

        dst[0].should eq 10_u8
        dst[1].should eq 50_u8
        dst[2].should eq 100_u8
        dst[3].should eq 100_u8
      end
    end
  end

  describe "max" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[3_u8, 1_u8, 9_u8, 2_u8, 5_u8, 8_u8, 4_u8, 7_u8,
        10_u8, 20_u8, 15_u8, 25_u8, 12_u8, 18_u8, 14_u8, 22_u8]
      result = simd.max(a)
      result.should eq 25_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[3_u8, 1_u8, 9_u8, 2_u8, 5_u8, 8_u8, 4_u8, 7_u8,
          10_u8, 20_u8, 15_u8, 25_u8, 12_u8, 18_u8, 14_u8, 22_u8]
        result = simd.max(a)
        result.should eq 25_u8
      end
    end
  end

  describe "min" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[3_u8, 1_u8, 9_u8, 2_u8, 5_u8, 8_u8, 4_u8, 7_u8,
        10_u8, 20_u8, 15_u8, 25_u8, 12_u8, 18_u8, 14_u8, 22_u8]
      result = simd.min(a)
      result.should eq 1_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[3_u8, 1_u8, 9_u8, 2_u8, 5_u8, 8_u8, 4_u8, 7_u8,
          10_u8, 20_u8, 15_u8, 25_u8, 12_u8, 18_u8, 14_u8, 22_u8]
        result = simd.min(a)
        result.should eq 1_u8
      end
    end
  end

  describe "sum" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8,
        9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8, 15_u8, 16_u8]
      result = simd.sum(a)
      result.should eq 136_u64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8,
          9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8, 15_u8, 16_u8]
        result = simd.sum(a)
        result.should eq 136_u64
      end
    end
  end

  describe "bitwise_and" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF_u8, 0x12_u8, 0xAA_u8, 0x55_u8]
      b = Slice[0x0F_u8, 0xF0_u8, 0xFF_u8, 0x00_u8]
      dst = Slice(UInt8).new(a.size)

      simd.bitwise_and(dst, a, b)

      dst[0].should eq 0x0F_u8
      dst[1].should eq 0x10_u8
      dst[2].should eq 0xAA_u8
      dst[3].should eq 0x00_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF_u8, 0x12_u8, 0xAA_u8, 0x55_u8]
        b = Slice[0x0F_u8, 0xF0_u8, 0xFF_u8, 0x00_u8]
        dst = Slice(UInt8).new(a.size)

        simd.bitwise_and(dst, a, b)

        dst[0].should eq 0x0F_u8
        dst[1].should eq 0x10_u8
        dst[2].should eq 0xAA_u8
        dst[3].should eq 0x00_u8
      end
    end
  end

  describe "bitwise_or" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xF0_u8, 0x12_u8, 0xAA_u8, 0x55_u8]
      b = Slice[0x0F_u8, 0xF0_u8, 0x00_u8, 0xAA_u8]
      dst = Slice(UInt8).new(a.size)

      simd.bitwise_or(dst, a, b)

      dst[0].should eq 0xFF_u8
      dst[1].should eq 0xF2_u8
      dst[2].should eq 0xAA_u8
      dst[3].should eq 0xFF_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xF0_u8, 0x12_u8, 0xAA_u8, 0x55_u8]
        b = Slice[0x0F_u8, 0xF0_u8, 0x00_u8, 0xAA_u8]
        dst = Slice(UInt8).new(a.size)

        simd.bitwise_or(dst, a, b)

        dst[0].should eq 0xFF_u8
        dst[1].should eq 0xF2_u8
        dst[2].should eq 0xAA_u8
        dst[3].should eq 0xFF_u8
      end
    end
  end

  describe "xor" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF_u8, 0x12_u8, 0xAA_u8, 0xFF_u8]
      b = Slice[0x0F_u8, 0x12_u8, 0x55_u8, 0x00_u8]
      dst = Slice(UInt8).new(a.size)

      simd.xor(dst, a, b)

      dst[0].should eq 0xF0_u8
      dst[1].should eq 0x00_u8
      dst[2].should eq 0xFF_u8
      dst[3].should eq 0xFF_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF_u8, 0x12_u8, 0xAA_u8, 0xFF_u8]
        b = Slice[0x0F_u8, 0x12_u8, 0x55_u8, 0x00_u8]
        dst = Slice(UInt8).new(a.size)

        simd.xor(dst, a, b)

        dst[0].should eq 0xF0_u8
        dst[1].should eq 0x00_u8
        dst[2].should eq 0xFF_u8
        dst[3].should eq 0xFF_u8
      end
    end
  end

  describe "popcount" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0x00_u8, 0xFF_u8, 0x0F_u8, 0xF0_u8, 0xAA_u8, 0x55_u8, 0x01_u8, 0x80_u8]
      result = simd.popcount(a)
      # 0 + 8 + 4 + 4 + 4 + 4 + 1 + 1 = 26
      result.should eq 26_u64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0x00_u8, 0xFF_u8, 0x0F_u8, 0xF0_u8, 0xAA_u8, 0x55_u8, 0x01_u8, 0x80_u8]
        result = simd.popcount(a)
        # 0 + 8 + 4 + 4 + 4 + 4 + 1 + 1 = 26
        result.should eq 26_u64
      end
    end
  end

  describe "cmp_gt_mask" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u8, 5_u8, 3_u8, 10_u8]
      b = Slice[2_u8, 3_u8, 3_u8, 5_u8]
      mask = Slice(UInt8).new(a.size)

      simd.cmp_gt_mask(mask, a, b)

      mask[0].should eq 0x00_u8
      mask[1].should eq 0xFF_u8
      mask[2].should eq 0x00_u8
      mask[3].should eq 0xFF_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u8, 5_u8, 3_u8, 10_u8]
        b = Slice[2_u8, 3_u8, 3_u8, 5_u8]
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
      t = Slice[1_u8, 2_u8, 3_u8, 4_u8]
      f = Slice[10_u8, 20_u8, 30_u8, 40_u8]
      mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
      dst = Slice(UInt8).new(t.size)

      simd.blend(dst, t, f, mask)

      dst[0].should eq 1_u8
      dst[1].should eq 20_u8
      dst[2].should eq 3_u8
      dst[3].should eq 40_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        t = Slice[1_u8, 2_u8, 3_u8, 4_u8]
        f = Slice[10_u8, 20_u8, 30_u8, 40_u8]
        mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
        dst = Slice(UInt8).new(t.size)

        simd.blend(dst, t, f, mask)

        dst[0].should eq 1_u8
        dst[1].should eq 20_u8
        dst[2].should eq 3_u8
        dst[3].should eq 40_u8
      end
    end
  end
end
