require "./spec_helper"

describe "SIMD UInt32 operations" do
  describe "add" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u32, 2_u32, 3_u32, 4_u32]
      b = Slice[10_u32, 20_u32, 30_u32, 40_u32]
      dst = Slice(UInt32).new(a.size)

      simd.add(dst, a, b)

      dst[0].should eq 11_u32
      dst[1].should eq 22_u32
      dst[2].should eq 33_u32
      dst[3].should eq 44_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u32, 2_u32, 3_u32, 4_u32]
        b = Slice[10_u32, 20_u32, 30_u32, 40_u32]
        dst = Slice(UInt32).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 11_u32
        dst[1].should eq 22_u32
        dst[2].should eq 33_u32
        dst[3].should eq 44_u32
      end
    end

    it "handles wrapping overflow" do
      check_implementations.each do |simd|
        a = Slice[UInt32::MAX, UInt32::MAX - 1]
        b = Slice[1_u32, 2_u32]
        dst = Slice(UInt32).new(a.size)

        simd.add(dst, a, b)

        dst[0].should eq 0_u32
        dst[1].should eq 0_u32
      end
    end
  end

  describe "sub" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[100_u32, 200_u32, 300_u32, 400_u32]
      b = Slice[1_u32, 2_u32, 3_u32, 4_u32]
      dst = Slice(UInt32).new(a.size)

      simd.sub(dst, a, b)

      dst[0].should eq 99_u32
      dst[1].should eq 198_u32
      dst[2].should eq 297_u32
      dst[3].should eq 396_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[100_u32, 200_u32, 300_u32, 400_u32]
        b = Slice[1_u32, 2_u32, 3_u32, 4_u32]
        dst = Slice(UInt32).new(a.size)

        simd.sub(dst, a, b)

        dst[0].should eq 99_u32
        dst[1].should eq 198_u32
        dst[2].should eq 297_u32
        dst[3].should eq 396_u32
      end
    end
  end

  describe "mul" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u32, 2_u32, 3_u32, 4_u32]
      b = Slice[10_u32, 20_u32, 30_u32, 40_u32]
      dst = Slice(UInt32).new(a.size)

      simd.mul(dst, a, b)

      dst[0].should eq 10_u32
      dst[1].should eq 40_u32
      dst[2].should eq 90_u32
      dst[3].should eq 160_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u32, 2_u32, 3_u32, 4_u32]
        b = Slice[10_u32, 20_u32, 30_u32, 40_u32]
        dst = Slice(UInt32).new(a.size)

        simd.mul(dst, a, b)

        dst[0].should eq 10_u32
        dst[1].should eq 40_u32
        dst[2].should eq 90_u32
        dst[3].should eq 160_u32
      end
    end
  end

  describe "clamp" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0_u32, 50_u32, 150_u32, 200_u32]
      dst = Slice(UInt32).new(a.size)

      simd.clamp(dst, a, 10_u32, 100_u32)

      dst[0].should eq 10_u32
      dst[1].should eq 50_u32
      dst[2].should eq 100_u32
      dst[3].should eq 100_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0_u32, 50_u32, 150_u32, 200_u32]
        dst = Slice(UInt32).new(a.size)

        simd.clamp(dst, a, 10_u32, 100_u32)

        dst[0].should eq 10_u32
        dst[1].should eq 50_u32
        dst[2].should eq 100_u32
        dst[3].should eq 100_u32
      end
    end
  end

  describe "max" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[3_u32, 1_u32, 9_u32, 2_u32]
      result = simd.max(a)
      result.should eq 9_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[3_u32, 1_u32, 9_u32, 2_u32]
        result = simd.max(a)
        result.should eq 9_u32
      end
    end
  end

  describe "min" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[3_u32, 1_u32, 9_u32, 2_u32]
      result = simd.min(a)
      result.should eq 1_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[3_u32, 1_u32, 9_u32, 2_u32]
        result = simd.min(a)
        result.should eq 1_u32
      end
    end
  end

  describe "sum" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u32, 2_u32, 3_u32, 4_u32]
      result = simd.sum(a)
      result.should eq 10_u64
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u32, 2_u32, 3_u32, 4_u32]
        result = simd.sum(a)
        result.should eq 10_u64
      end
    end
  end

  describe "bitwise_and" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF00FF00_u32, 0x12345678_u32, 0xAAAAAAAA_u32, 0x55555555_u32]
      b = Slice[0x0F0F0F0F_u32, 0xF0F0F0F0_u32, 0xFFFFFFFF_u32, 0x00000000_u32]
      dst = Slice(UInt32).new(a.size)

      simd.bitwise_and(dst, a, b)

      dst[0].should eq 0x0F000F00_u32
      dst[1].should eq 0x10305070_u32
      dst[2].should eq 0xAAAAAAAA_u32
      dst[3].should eq 0x00000000_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF00FF00_u32, 0x12345678_u32, 0xAAAAAAAA_u32, 0x55555555_u32]
        b = Slice[0x0F0F0F0F_u32, 0xF0F0F0F0_u32, 0xFFFFFFFF_u32, 0x00000000_u32]
        dst = Slice(UInt32).new(a.size)

        simd.bitwise_and(dst, a, b)

        dst[0].should eq 0x0F000F00_u32
        dst[1].should eq 0x10305070_u32
        dst[2].should eq 0xAAAAAAAA_u32
        dst[3].should eq 0x00000000_u32
      end
    end
  end

  describe "bitwise_or" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF00FF00_u32, 0x12345678_u32, 0xAAAAAAAA_u32, 0x55555555_u32]
      b = Slice[0x0F0F0F0F_u32, 0xF0F0F0F0_u32, 0x00000000_u32, 0xAAAAAAAA_u32]
      dst = Slice(UInt32).new(a.size)

      simd.bitwise_or(dst, a, b)

      dst[0].should eq 0xFF0FFF0F_u32
      dst[1].should eq 0xF2F4F6F8_u32
      dst[2].should eq 0xAAAAAAAA_u32
      dst[3].should eq 0xFFFFFFFF_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF00FF00_u32, 0x12345678_u32, 0xAAAAAAAA_u32, 0x55555555_u32]
        b = Slice[0x0F0F0F0F_u32, 0xF0F0F0F0_u32, 0x00000000_u32, 0xAAAAAAAA_u32]
        dst = Slice(UInt32).new(a.size)

        simd.bitwise_or(dst, a, b)

        dst[0].should eq 0xFF0FFF0F_u32
        dst[1].should eq 0xF2F4F6F8_u32
        dst[2].should eq 0xAAAAAAAA_u32
        dst[3].should eq 0xFFFFFFFF_u32
      end
    end
  end

  describe "xor" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0xFF00FF00_u32, 0x12345678_u32, 0xAAAAAAAA_u32, 0xFFFFFFFF_u32]
      b = Slice[0x0F0F0F0F_u32, 0x12345678_u32, 0x55555555_u32, 0x00000000_u32]
      dst = Slice(UInt32).new(a.size)

      simd.xor(dst, a, b)

      dst[0].should eq 0xF00FF00F_u32
      dst[1].should eq 0x00000000_u32
      dst[2].should eq 0xFFFFFFFF_u32
      dst[3].should eq 0xFFFFFFFF_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0xFF00FF00_u32, 0x12345678_u32, 0xAAAAAAAA_u32, 0xFFFFFFFF_u32]
        b = Slice[0x0F0F0F0F_u32, 0x12345678_u32, 0x55555555_u32, 0x00000000_u32]
        dst = Slice(UInt32).new(a.size)

        simd.xor(dst, a, b)

        dst[0].should eq 0xF00FF00F_u32
        dst[1].should eq 0x00000000_u32
        dst[2].should eq 0xFFFFFFFF_u32
        dst[3].should eq 0xFFFFFFFF_u32
      end
    end
  end

  describe "bswap" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[0x12345678_u32, 0xAABBCCDD_u32, 0x00FF00FF_u32, 0xFF00FF00_u32]
      dst = Slice(UInt32).new(a.size)

      simd.bswap(dst, a)

      dst[0].should eq 0x78563412_u32
      dst[1].should eq 0xDDCCBBAA_u32
      dst[2].should eq 0xFF00FF00_u32
      dst[3].should eq 0x00FF00FF_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[0x12345678_u32, 0xAABBCCDD_u32, 0x00FF00FF_u32, 0xFF00FF00_u32]
        dst = Slice(UInt32).new(a.size)

        simd.bswap(dst, a)

        dst[0].should eq 0x78563412_u32
        dst[1].should eq 0xDDCCBBAA_u32
        dst[2].should eq 0xFF00FF00_u32
        dst[3].should eq 0x00FF00FF_u32
      end
    end
  end

  describe "cmp_gt_mask" do
    it "scalar implementation" do
      simd = SIMD.scalar
      a = Slice[1_u32, 5_u32, 3_u32, 10_u32]
      b = Slice[2_u32, 3_u32, 3_u32, 5_u32]
      mask = Slice(UInt8).new(a.size)

      simd.cmp_gt_mask(mask, a, b)

      mask[0].should eq 0x00_u8
      mask[1].should eq 0xFF_u8
      mask[2].should eq 0x00_u8
      mask[3].should eq 0xFF_u8
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        a = Slice[1_u32, 5_u32, 3_u32, 10_u32]
        b = Slice[2_u32, 3_u32, 3_u32, 5_u32]
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
      t = Slice[1_u32, 2_u32, 3_u32, 4_u32]
      f = Slice[10_u32, 20_u32, 30_u32, 40_u32]
      mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
      dst = Slice(UInt32).new(t.size)

      simd.blend(dst, t, f, mask)

      dst[0].should eq 1_u32
      dst[1].should eq 20_u32
      dst[2].should eq 3_u32
      dst[3].should eq 40_u32
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        t = Slice[1_u32, 2_u32, 3_u32, 4_u32]
        f = Slice[10_u32, 20_u32, 30_u32, 40_u32]
        mask = Slice[0xFF_u8, 0x00_u8, 0xFF_u8, 0x00_u8]
        dst = Slice(UInt32).new(t.size)

        simd.blend(dst, t, f, mask)

        dst[0].should eq 1_u32
        dst[1].should eq 20_u32
        dst[2].should eq 3_u32
        dst[3].should eq 40_u32
      end
    end
  end
end
