require "./spec_helper"

describe "SIMD UInt16 operations" do
  describe "add" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u16, 2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16]
      b = Slice[10_u16, 20_u16, 30_u16, 40_u16, 50_u16, 60_u16, 70_u16, 80_u16]
      dst = Slice(UInt16).new(a.size)

      simd.add(dst, a, b)

      dst[0].should eq 11_u16
      dst[1].should eq 22_u16
      dst[2].should eq 33_u16
      dst[3].should eq 44_u16
      dst[4].should eq 55_u16
      dst[5].should eq 66_u16
      dst[6].should eq 77_u16
      dst[7].should eq 88_u16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u16, 2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16]
        b = Slice[10_u16, 20_u16, 30_u16, 40_u16, 50_u16, 60_u16, 70_u16, 80_u16]
        dst = Slice(UInt16).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 11_u16
        dst[1].should eq 22_u16
        dst[2].should eq 33_u16
        dst[3].should eq 44_u16
        dst[4].should eq 55_u16
        dst[5].should eq 66_u16
        dst[6].should eq 77_u16
        dst[7].should eq 88_u16
      end
    end

    it "handles wrapping overflow" do
      check_implementations.each do |simd|
        a = Slice[UInt16::MAX, UInt16::MAX - 1]
        b = Slice[1_u16, 2_u16]
        dst = Slice(UInt16).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 0_u16
        dst[1].should eq 0_u16
      end
    end
  end

  describe "sub" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[100_u16, 200_u16, 300_u16, 400_u16]
      b = Slice[1_u16, 2_u16, 3_u16, 4_u16]
      dst = Slice(UInt16).new(a.size)

      simd.sub(dst, a, b)

      dst[0].should eq 99_u16
      dst[1].should eq 198_u16
      dst[2].should eq 297_u16
      dst[3].should eq 396_u16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[100_u16, 200_u16, 300_u16, 400_u16]
        b = Slice[1_u16, 2_u16, 3_u16, 4_u16]
        dst = Slice(UInt16).new(a.size)

        simd.sub(dst, a, b)

        dst[0].should eq 99_u16
        dst[1].should eq 198_u16
        dst[2].should eq 297_u16
        dst[3].should eq 396_u16
      end
    end
  end

  describe "mul" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u16, 2_u16, 3_u16, 4_u16]
      b = Slice[10_u16, 20_u16, 30_u16, 40_u16]
      dst = Slice(UInt16).new(a.size)

      simd.mul(dst, a, b)

      dst[0].should eq 10_u16
      dst[1].should eq 40_u16
      dst[2].should eq 90_u16
      dst[3].should eq 160_u16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u16, 2_u16, 3_u16, 4_u16]
        b = Slice[10_u16, 20_u16, 30_u16, 40_u16]
        dst = Slice(UInt16).new(a.size)

        simd.mul(dst, a, b)

        dst[0].should eq 10_u16
        dst[1].should eq 40_u16
        dst[2].should eq 90_u16
        dst[3].should eq 160_u16
      end
    end
  end

  describe "clamp" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0_u16, 50_u16, 150_u16, 200_u16]
      dst = Slice(UInt16).new(a.size)

      simd.clamp(dst, a, 10_u16, 100_u16)

      dst[0].should eq 10_u16
      dst[1].should eq 50_u16
      dst[2].should eq 100_u16
      dst[3].should eq 100_u16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0_u16, 50_u16, 150_u16, 200_u16]
        dst = Slice(UInt16).new(a.size)

        simd.clamp(dst, a, 10_u16, 100_u16)

        dst[0].should eq 10_u16
        dst[1].should eq 50_u16
        dst[2].should eq 100_u16
        dst[3].should eq 100_u16
      end
    end
  end

  describe "max" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[3_u16, 1_u16, 9_u16, 2_u16, 5_u16, 8_u16, 4_u16, 7_u16]
      result = simd.max(a)
      result.should eq 9_u16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[3_u16, 1_u16, 9_u16, 2_u16, 5_u16, 8_u16, 4_u16, 7_u16]
        result = simd.max(a)
        result.should eq 9_u16
      end
    end
  end

  describe "min" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[3_u16, 1_u16, 9_u16, 2_u16, 5_u16, 8_u16, 4_u16, 7_u16]
      result = simd.min(a)
      result.should eq 1_u16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[3_u16, 1_u16, 9_u16, 2_u16, 5_u16, 8_u16, 4_u16, 7_u16]
        result = simd.min(a)
        result.should eq 1_u16
      end
    end
  end

  describe "sum" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u16, 2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16]
      result = simd.sum(a)
      result.should eq 36_u64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u16, 2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16]
        result = simd.sum(a)
        result.should eq 36_u64
      end
    end
  end

  describe "bitwise_and" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF00_u16, 0x1234_u16, 0xAAAA_u16, 0x5555_u16]
      b = Slice[0x0F0F_u16, 0xF0F0_u16, 0xFFFF_u16, 0x0000_u16]
      dst = Slice(UInt16).new(a.size)

      simd.bitwise_and(dst, a, b)

      dst[0].should eq 0x0F00_u16
      dst[1].should eq 0x1030_u16
      dst[2].should eq 0xAAAA_u16
      dst[3].should eq 0x0000_u16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF00_u16, 0x1234_u16, 0xAAAA_u16, 0x5555_u16]
        b = Slice[0x0F0F_u16, 0xF0F0_u16, 0xFFFF_u16, 0x0000_u16]
        dst = Slice(UInt16).new(a.size)

        simd.bitwise_and(dst, a, b)

        dst[0].should eq 0x0F00_u16
        dst[1].should eq 0x1030_u16
        dst[2].should eq 0xAAAA_u16
        dst[3].should eq 0x0000_u16
      end
    end
  end

  describe "bitwise_or" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF00_u16, 0x1234_u16, 0xAAAA_u16, 0x5555_u16]
      b = Slice[0x0F0F_u16, 0xF0F0_u16, 0x0000_u16, 0xAAAA_u16]
      dst = Slice(UInt16).new(a.size)

      simd.bitwise_or(dst, a, b)

      dst[0].should eq 0xFF0F_u16
      dst[1].should eq 0xF2F4_u16
      dst[2].should eq 0xAAAA_u16
      dst[3].should eq 0xFFFF_u16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF00_u16, 0x1234_u16, 0xAAAA_u16, 0x5555_u16]
        b = Slice[0x0F0F_u16, 0xF0F0_u16, 0x0000_u16, 0xAAAA_u16]
        dst = Slice(UInt16).new(a.size)

        simd.bitwise_or(dst, a, b)

        dst[0].should eq 0xFF0F_u16
        dst[1].should eq 0xF2F4_u16
        dst[2].should eq 0xAAAA_u16
        dst[3].should eq 0xFFFF_u16
      end
    end
  end

  describe "xor" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF00_u16, 0x1234_u16, 0xAAAA_u16, 0xFFFF_u16]
      b = Slice[0x0F0F_u16, 0x1234_u16, 0x5555_u16, 0x0000_u16]
      dst = Slice(UInt16).new(a.size)

      simd.xor(dst, a, b)

      dst[0].should eq 0xF00F_u16
      dst[1].should eq 0x0000_u16
      dst[2].should eq 0xFFFF_u16
      dst[3].should eq 0xFFFF_u16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF00_u16, 0x1234_u16, 0xAAAA_u16, 0xFFFF_u16]
        b = Slice[0x0F0F_u16, 0x1234_u16, 0x5555_u16, 0x0000_u16]
        dst = Slice(UInt16).new(a.size)

        simd.xor(dst, a, b)

        dst[0].should eq 0xF00F_u16
        dst[1].should eq 0x0000_u16
        dst[2].should eq 0xFFFF_u16
        dst[3].should eq 0xFFFF_u16
      end
    end
  end

  describe "bswap" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0x1234_u16, 0xAABB_u16, 0x00FF_u16, 0xFF00_u16]
      dst = Slice(UInt16).new(a.size)

      simd.bswap(dst, a)

      dst[0].should eq 0x3412_u16
      dst[1].should eq 0xBBAA_u16
      dst[2].should eq 0xFF00_u16
      dst[3].should eq 0x00FF_u16
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0x1234_u16, 0xAABB_u16, 0x00FF_u16, 0xFF00_u16]
        dst = Slice(UInt16).new(a.size)

        simd.bswap(dst, a)

        dst[0].should eq 0x3412_u16
        dst[1].should eq 0xBBAA_u16
        dst[2].should eq 0xFF00_u16
        dst[3].should eq 0x00FF_u16
      end
    end
  end

  describe "cmp_gt_mask" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u16, 5_u16, 3_u16, 10_u16]
      b = Slice[2_u16, 3_u16, 3_u16, 5_u16]
      mask = Slice(UInt8).new(a.size)

      simd.cmp_gt_mask(mask, a, b)

      mask[0].should eq 0x00_u8
      mask[1].should eq 0xFF_u8
      mask[2].should eq 0x00_u8
      mask[3].should eq 0xFF_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u16, 5_u16, 3_u16, 10_u16]
        b = Slice[2_u16, 3_u16, 3_u16, 5_u16]
        mask = Slice(UInt8).new(a.size)

        simd.cmp_gt_mask(mask, a, b)

        mask[0].should eq 0x00_u8
        mask[1].should eq 0xFF_u8
        mask[2].should eq 0x00_u8
        mask[3].should eq 0xFF_u8
      end
    end
  end
end
