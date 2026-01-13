require "./spec_helper"

describe "SIMD Int8 operations" do
  describe "add" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_i8, -2_i8, 3_i8, -4_i8, 5_i8, -6_i8, 7_i8, -8_i8,
        9_i8, -10_i8, 11_i8, -12_i8, 13_i8, -14_i8, 15_i8, -16_i8]
      b = Slice[10_i8, 20_i8, -30_i8, -40_i8, 50_i8, 60_i8, -70_i8, -80_i8,
        9_i8, 10_i8, 11_i8, 12_i8, 13_i8, 14_i8, 15_i8, 16_i8]
      dst = Slice(Int8).new(a.size)

      simd.add(dst, a, b)

      dst[0].should eq 11_i8
      dst[1].should eq 18_i8
      dst[2].should eq(-27_i8)
      dst[3].should eq(-44_i8)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_i8, -2_i8, 3_i8, -4_i8, 5_i8, -6_i8, 7_i8, -8_i8,
          9_i8, -10_i8, 11_i8, -12_i8, 13_i8, -14_i8, 15_i8, -16_i8]
        b = Slice[10_i8, 20_i8, -30_i8, -40_i8, 50_i8, 60_i8, -70_i8, -80_i8,
          9_i8, 10_i8, 11_i8, 12_i8, 13_i8, 14_i8, 15_i8, 16_i8]
        dst = Slice(Int8).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 11_i8
        dst[1].should eq 18_i8
        dst[2].should eq(-27_i8)
        dst[3].should eq(-44_i8)
      end
    end
  end

  describe "sub" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[100_i8, -50_i8, 75_i8, -25_i8]
      b = Slice[1_i8, 2_i8, -3_i8, -4_i8]
      dst = Slice(Int8).new(a.size)

      simd.sub(dst, a, b)

      dst[0].should eq 99_i8
      dst[1].should eq(-52_i8)
      dst[2].should eq 78_i8
      dst[3].should eq(-21_i8)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[100_i8, -50_i8, 75_i8, -25_i8]
        b = Slice[1_i8, 2_i8, -3_i8, -4_i8]
        dst = Slice(Int8).new(a.size)

        simd.sub(dst, a, b)

        dst[0].should eq 99_i8
        dst[1].should eq(-52_i8)
        dst[2].should eq 78_i8
        dst[3].should eq(-21_i8)
      end
    end
  end

  describe "clamp" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-50_i8, 50_i8, 100_i8, -100_i8]
      dst = Slice(Int8).new(a.size)

      simd.clamp(dst, a, -25_i8, 75_i8)

      dst[0].should eq(-25_i8)
      dst[1].should eq 50_i8
      dst[2].should eq 75_i8
      dst[3].should eq(-25_i8)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-50_i8, 50_i8, 100_i8, -100_i8]
        dst = Slice(Int8).new(a.size)

        simd.clamp(dst, a, -25_i8, 75_i8)

        dst[0].should eq(-25_i8)
        dst[1].should eq 50_i8
        dst[2].should eq 75_i8
        dst[3].should eq(-25_i8)
      end
    end
  end

  describe "max" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-3_i8, 1_i8, -9_i8, 2_i8, -5_i8, 8_i8, -4_i8, 7_i8,
        10_i8, -20_i8, 15_i8, -25_i8, 12_i8, -18_i8, 14_i8, -22_i8]
      result = simd.max(a)
      result.should eq 15_i8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-3_i8, 1_i8, -9_i8, 2_i8, -5_i8, 8_i8, -4_i8, 7_i8,
          10_i8, -20_i8, 15_i8, -25_i8, 12_i8, -18_i8, 14_i8, -22_i8]
        result = simd.max(a)
        result.should eq 15_i8
      end
    end

    it "finds max correctly with all negatives" do
      check_implementations.each do |simd|
        a = Slice[-3_i8, -1_i8, -9_i8, -2_i8]
        result = simd.max(a)
        result.should eq(-1_i8)
      end
    end
  end

  describe "min" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-3_i8, 1_i8, -9_i8, 2_i8, -5_i8, 8_i8, -4_i8, 7_i8,
        10_i8, -20_i8, 15_i8, -25_i8, 12_i8, -18_i8, 14_i8, -22_i8]
      result = simd.min(a)
      result.should eq(-25_i8)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-3_i8, 1_i8, -9_i8, 2_i8, -5_i8, 8_i8, -4_i8, 7_i8,
          10_i8, -20_i8, 15_i8, -25_i8, 12_i8, -18_i8, 14_i8, -22_i8]
        result = simd.min(a)
        result.should eq(-25_i8)
      end
    end
  end

  describe "cmp_gt_mask" do
    it "scalar implementation with signed comparison" do
      simd = SIMD.scalar
      a = Slice[-1_i8, 5_i8, 0_i8, -10_i8]
      b = Slice[-2_i8, 3_i8, 0_i8, 5_i8]
      mask = Slice(UInt8).new(a.size)

      simd.cmp_gt_mask(mask, a, b)

      mask[0].should eq 0xFF_u8 # -1 > -2
      mask[1].should eq 0xFF_u8 # 5 > 3
      mask[2].should eq 0x00_u8 # 0 == 0
      mask[3].should eq 0x00_u8 # -10 < 5
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-1_i8, 5_i8, 0_i8, -10_i8]
        b = Slice[-2_i8, 3_i8, 0_i8, 5_i8]
        mask = Slice(UInt8).new(a.size)

        simd.cmp_gt_mask(mask, a, b)

        mask[0].should eq 0xFF_u8
        mask[1].should eq 0xFF_u8
        mask[2].should eq 0x00_u8
        mask[3].should eq 0x00_u8
      end
    end
  end
end
