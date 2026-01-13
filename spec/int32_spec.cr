require "./spec_helper"

describe "SIMD Int32 operations" do
  describe "add" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_i32, -2_i32, 3_i32, -4_i32]
      b = Slice[10_i32, 20_i32, -30_i32, -40_i32]
      dst = Slice(Int32).new(a.size)

      simd.add(dst, a, b)

      dst[0].should eq 11_i32
      dst[1].should eq 18_i32
      dst[2].should eq(-27_i32)
      dst[3].should eq(-44_i32)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_i32, -2_i32, 3_i32, -4_i32]
        b = Slice[10_i32, 20_i32, -30_i32, -40_i32]
        dst = Slice(Int32).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 11_i32
        dst[1].should eq 18_i32
        dst[2].should eq(-27_i32)
        dst[3].should eq(-44_i32)
      end
    end
  end

  describe "sub" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[100_i32, -200_i32, 300_i32, -400_i32]
      b = Slice[1_i32, 2_i32, -3_i32, -4_i32]
      dst = Slice(Int32).new(a.size)

      simd.sub(dst, a, b)

      dst[0].should eq 99_i32
      dst[1].should eq(-202_i32)
      dst[2].should eq 303_i32
      dst[3].should eq(-396_i32)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[100_i32, -200_i32, 300_i32, -400_i32]
        b = Slice[1_i32, 2_i32, -3_i32, -4_i32]
        dst = Slice(Int32).new(a.size)

        simd.sub(dst, a, b)

        dst[0].should eq 99_i32
        dst[1].should eq(-202_i32)
        dst[2].should eq 303_i32
        dst[3].should eq(-396_i32)
      end
    end
  end

  describe "clamp" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-50_i32, 50_i32, 150_i32, -100_i32]
      dst = Slice(Int32).new(a.size)

      simd.clamp(dst, a, -25_i32, 100_i32)

      dst[0].should eq(-25_i32)
      dst[1].should eq 50_i32
      dst[2].should eq 100_i32
      dst[3].should eq(-25_i32)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-50_i32, 50_i32, 150_i32, -100_i32]
        dst = Slice(Int32).new(a.size)

        simd.clamp(dst, a, -25_i32, 100_i32)

        dst[0].should eq(-25_i32)
        dst[1].should eq 50_i32
        dst[2].should eq 100_i32
        dst[3].should eq(-25_i32)
      end
    end
  end

  describe "max" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-3_i32, 1_i32, -9_i32, 2_i32]
      result = simd.max(a)
      result.should eq 2_i32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-3_i32, 1_i32, -9_i32, 2_i32]
        result = simd.max(a)
        result.should eq 2_i32
      end
    end

    it "finds max correctly with all negatives" do
      check_implementations.each do |simd|
        a = Slice[-3_i32, -1_i32, -9_i32, -2_i32]
        result = simd.max(a)
        result.should eq(-1_i32)
      end
    end
  end

  describe "min" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[-3_i32, 1_i32, -9_i32, 2_i32]
      result = simd.min(a)
      result.should eq(-9_i32)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-3_i32, 1_i32, -9_i32, 2_i32]
        result = simd.min(a)
        result.should eq(-9_i32)
      end
    end
  end

  describe "cmp_gt_mask" do
    it "scalar implementation with signed comparison" do
      simd = SIMD.scalar
      a = Slice[-1_i32, 5_i32, 0_i32, -10_i32]
      b = Slice[-2_i32, 3_i32, 0_i32, 5_i32]
      mask = Slice(UInt8).new(a.size)

      simd.cmp_gt_mask(mask, a, b)

      mask[0].should eq 0xFF_u8 # -1 > -2
      mask[1].should eq 0xFF_u8 # 5 > 3
      mask[2].should eq 0x00_u8 # 0 == 0
      mask[3].should eq 0x00_u8 # -10 < 5
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-1_i32, 5_i32, 0_i32, -10_i32]
        b = Slice[-2_i32, 3_i32, 0_i32, 5_i32]
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
