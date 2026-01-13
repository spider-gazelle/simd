require "./spec_helper"

describe "SIMD UInt64 operations" do
  describe "add" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u64, 2_u64, 3_u64, 4_u64]
      b = Slice[10_u64, 20_u64, 30_u64, 40_u64]
      dst = Slice(UInt64).new(a.size)

      simd.add(dst, a, b)

      dst[0].should eq 11_u64
      dst[1].should eq 22_u64
      dst[2].should eq 33_u64
      dst[3].should eq 44_u64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u64, 2_u64, 3_u64, 4_u64]
        b = Slice[10_u64, 20_u64, 30_u64, 40_u64]
        dst = Slice(UInt64).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 11_u64
        dst[1].should eq 22_u64
        dst[2].should eq 33_u64
        dst[3].should eq 44_u64
      end
    end
  end

  describe "sub" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[100_u64, 200_u64, 300_u64, 400_u64]
      b = Slice[1_u64, 2_u64, 3_u64, 4_u64]
      dst = Slice(UInt64).new(a.size)

      simd.sub(dst, a, b)

      dst[0].should eq 99_u64
      dst[1].should eq 198_u64
      dst[2].should eq 297_u64
      dst[3].should eq 396_u64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[100_u64, 200_u64, 300_u64, 400_u64]
        b = Slice[1_u64, 2_u64, 3_u64, 4_u64]
        dst = Slice(UInt64).new(a.size)

        simd.sub(dst, a, b)

        dst[0].should eq 99_u64
        dst[1].should eq 198_u64
        dst[2].should eq 297_u64
        dst[3].should eq 396_u64
      end
    end
  end

  describe "bitwise_and" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF00FF00FF00FF00_u64, 0x123456789ABCDEF0_u64]
      b = Slice[0x0F0F0F0F0F0F0F0F_u64, 0xF0F0F0F0F0F0F0F0_u64]
      dst = Slice(UInt64).new(a.size)

      simd.bitwise_and(dst, a, b)

      dst[0].should eq 0x0F000F000F000F00_u64
      dst[1].should eq 0x1030507090B0D0F0_u64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF00FF00FF00FF00_u64, 0x123456789ABCDEF0_u64]
        b = Slice[0x0F0F0F0F0F0F0F0F_u64, 0xF0F0F0F0F0F0F0F0_u64]
        dst = Slice(UInt64).new(a.size)

        simd.bitwise_and(dst, a, b)

        dst[0].should eq 0x0F000F000F000F00_u64
        dst[1].should eq 0x1030507090B0D0F0_u64
      end
    end
  end

  describe "bitwise_or" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF00FF00FF00FF00_u64, 0x123456789ABCDEF0_u64]
      b = Slice[0x0F0F0F0F0F0F0F0F_u64, 0x000000000000000F_u64]
      dst = Slice(UInt64).new(a.size)

      simd.bitwise_or(dst, a, b)

      dst[0].should eq 0xFF0FFF0FFF0FFF0F_u64
      dst[1].should eq 0x123456789ABCDEFF_u64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF00FF00FF00FF00_u64, 0x123456789ABCDEF0_u64]
        b = Slice[0x0F0F0F0F0F0F0F0F_u64, 0x000000000000000F_u64]
        dst = Slice(UInt64).new(a.size)

        simd.bitwise_or(dst, a, b)

        dst[0].should eq 0xFF0FFF0FFF0FFF0F_u64
        dst[1].should eq 0x123456789ABCDEFF_u64
      end
    end
  end

  describe "xor" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF00FF00FF00FF00_u64, 0xAAAAAAAAAAAAAAAA_u64]
      b = Slice[0x0F0F0F0F0F0F0F0F_u64, 0x5555555555555555_u64]
      dst = Slice(UInt64).new(a.size)

      simd.xor(dst, a, b)

      dst[0].should eq 0xF00FF00FF00FF00F_u64
      dst[1].should eq 0xFFFFFFFFFFFFFFFF_u64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF00FF00FF00FF00_u64, 0xAAAAAAAAAAAAAAAA_u64]
        b = Slice[0x0F0F0F0F0F0F0F0F_u64, 0x5555555555555555_u64]
        dst = Slice(UInt64).new(a.size)

        simd.xor(dst, a, b)

        dst[0].should eq 0xF00FF00FF00FF00F_u64
        dst[1].should eq 0xFFFFFFFFFFFFFFFF_u64
      end
    end
  end

  describe "bswap" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0x0102030405060708_u64, 0xAABBCCDDEEFF0011_u64]
      dst = Slice(UInt64).new(a.size)

      simd.bswap(dst, a)

      dst[0].should eq 0x0807060504030201_u64
      dst[1].should eq 0x1100FFEEDDCCBBAA_u64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0x0102030405060708_u64, 0xAABBCCDDEEFF0011_u64]
        dst = Slice(UInt64).new(a.size)

        simd.bswap(dst, a)

        dst[0].should eq 0x0807060504030201_u64
        dst[1].should eq 0x1100FFEEDDCCBBAA_u64
      end
    end
  end

  describe "cmp_gt_mask" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u64, 5_u64, 3_u64, 10_u64]
      b = Slice[2_u64, 3_u64, 3_u64, 5_u64]
      mask = Slice(UInt8).new(a.size)

      simd.cmp_gt_mask(mask, a, b)

      mask[0].should eq 0x00_u8
      mask[1].should eq 0xFF_u8
      mask[2].should eq 0x00_u8
      mask[3].should eq 0xFF_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u64, 5_u64, 3_u64, 10_u64]
        b = Slice[2_u64, 3_u64, 3_u64, 5_u64]
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
