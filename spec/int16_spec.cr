require "./spec_helper"

describe "SIMD Int16 operations" do
  describe "add" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_i16, -2_i16, 3_i16, -4_i16, 5_i16, -6_i16, 7_i16, -8_i16]
      b = Slice[10_i16, 20_i16, -30_i16, -40_i16, 50_i16, 60_i16, -70_i16, -80_i16]
      dst = Slice(Int16).new(a.size)

      simd.add(dst, a, b)

      dst[0].should eq 11_i16
      dst[1].should eq 18_i16
      dst[2].should eq(-27_i16)
      dst[3].should eq(-44_i16)
      dst[4].should eq 55_i16
      dst[5].should eq 54_i16
      dst[6].should eq(-63_i16)
      dst[7].should eq(-88_i16)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_i16, -2_i16, 3_i16, -4_i16, 5_i16, -6_i16, 7_i16, -8_i16]
        b = Slice[10_i16, 20_i16, -30_i16, -40_i16, 50_i16, 60_i16, -70_i16, -80_i16]
        dst = Slice(Int16).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 11_i16
        dst[1].should eq 18_i16
        dst[2].should eq(-27_i16)
        dst[3].should eq(-44_i16)
        dst[4].should eq 55_i16
        dst[5].should eq 54_i16
        dst[6].should eq(-63_i16)
        dst[7].should eq(-88_i16)
      end
    end
  end

  describe "sub" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[100_i16, -200_i16, 300_i16, -400_i16]
      b = Slice[1_i16, 2_i16, -3_i16, -4_i16]
      dst = Slice(Int16).new(a.size)

      simd.sub(dst, a, b)

      dst[0].should eq 99_i16
      dst[1].should eq(-202_i16)
      dst[2].should eq 303_i16
      dst[3].should eq(-396_i16)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[100_i16, -200_i16, 300_i16, -400_i16]
        b = Slice[1_i16, 2_i16, -3_i16, -4_i16]
        dst = Slice(Int16).new(a.size)

        simd.sub(dst, a, b)

        dst[0].should eq 99_i16
        dst[1].should eq(-202_i16)
        dst[2].should eq 303_i16
        dst[3].should eq(-396_i16)
      end
    end
  end

  describe "clamp" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-50_i16, 50_i16, 150_i16, -100_i16]
      dst = Slice(Int16).new(a.size)

      simd.clamp(dst, a, -25_i16, 100_i16)

      dst[0].should eq(-25_i16)
      dst[1].should eq 50_i16
      dst[2].should eq 100_i16
      dst[3].should eq(-25_i16)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-50_i16, 50_i16, 150_i16, -100_i16]
        dst = Slice(Int16).new(a.size)

        simd.clamp(dst, a, -25_i16, 100_i16)

        dst[0].should eq(-25_i16)
        dst[1].should eq 50_i16
        dst[2].should eq 100_i16
        dst[3].should eq(-25_i16)
      end
    end
  end

  describe "max" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-3_i16, 1_i16, -9_i16, 2_i16, -5_i16, 8_i16, -4_i16, 7_i16]
      result = simd.max(a)
      result.should eq 8_i16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-3_i16, 1_i16, -9_i16, 2_i16, -5_i16, 8_i16, -4_i16, 7_i16]
        result = simd.max(a)
        result.should eq 8_i16
      end
    end

    it "finds max correctly with all negatives" do
      check_implementations.each do |simd|
        a = Slice[-3_i16, -1_i16, -9_i16, -2_i16]
        result = simd.max(a)
        result.should eq(-1_i16)
      end
    end
  end

  describe "min" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-3_i16, 1_i16, -9_i16, 2_i16, -5_i16, 8_i16, -4_i16, 7_i16]
      result = simd.min(a)
      result.should eq(-9_i16)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-3_i16, 1_i16, -9_i16, 2_i16, -5_i16, 8_i16, -4_i16, 7_i16]
        result = simd.min(a)
        result.should eq(-9_i16)
      end
    end
  end

  describe "cmp_gt_mask" do
    it "scalar implementation with signed comparison" do
      simd = SIMD.scalar
      a = Slice[-1_i16, 5_i16, 0_i16, -10_i16]
      b = Slice[-2_i16, 3_i16, 0_i16, 5_i16]
      mask = Slice(UInt8).new(a.size)

      simd.cmp_gt_mask(mask, a, b)

      mask[0].should eq 0xFF_u8 # -1 > -2
      mask[1].should eq 0xFF_u8 # 5 > 3
      mask[2].should eq 0x00_u8 # 0 == 0
      mask[3].should eq 0x00_u8 # -10 < 5
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-1_i16, 5_i16, 0_i16, -10_i16]
        b = Slice[-2_i16, 3_i16, 0_i16, 5_i16]
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
