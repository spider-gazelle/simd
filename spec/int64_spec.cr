require "./spec_helper"

describe "SIMD Int64 operations" do
  describe "add" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_i64, -2_i64, 3_i64, -4_i64]
      b = Slice[10_i64, 20_i64, -30_i64, -40_i64]
      dst = Slice(Int64).new(a.size)

      simd.add(dst, a, b)

      dst[0].should eq 11_i64
      dst[1].should eq 18_i64
      dst[2].should eq(-27_i64)
      dst[3].should eq(-44_i64)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_i64, -2_i64, 3_i64, -4_i64]
        b = Slice[10_i64, 20_i64, -30_i64, -40_i64]
        dst = Slice(Int64).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 11_i64
        dst[1].should eq 18_i64
        dst[2].should eq(-27_i64)
        dst[3].should eq(-44_i64)
      end
    end
  end

  describe "sub" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[100_i64, -200_i64, 300_i64, -400_i64]
      b = Slice[1_i64, 2_i64, -3_i64, -4_i64]
      dst = Slice(Int64).new(a.size)

      simd.sub(dst, a, b)

      dst[0].should eq 99_i64
      dst[1].should eq(-202_i64)
      dst[2].should eq 303_i64
      dst[3].should eq(-396_i64)
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[100_i64, -200_i64, 300_i64, -400_i64]
        b = Slice[1_i64, 2_i64, -3_i64, -4_i64]
        dst = Slice(Int64).new(a.size)

        simd.sub(dst, a, b)

        dst[0].should eq 99_i64
        dst[1].should eq(-202_i64)
        dst[2].should eq 303_i64
        dst[3].should eq(-396_i64)
      end
    end
  end

  describe "cmp_gt_mask" do
    it "scalar implementation with signed comparison" do
      simd = SIMD.scalar
      a = Slice[-1_i64, 5_i64, 0_i64, -10_i64]
      b = Slice[-2_i64, 3_i64, 0_i64, 5_i64]
      mask = Slice(UInt8).new(a.size)

      simd.cmp_gt_mask(mask, a, b)

      mask[0].should eq 0xFF_u8 # -1 > -2
      mask[1].should eq 0xFF_u8 # 5 > 3
      mask[2].should eq 0x00_u8 # 0 == 0
      mask[3].should eq 0x00_u8 # -10 < 5
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[-1_i64, 5_i64, 0_i64, -10_i64]
        b = Slice[-2_i64, 3_i64, 0_i64, 5_i64]
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
