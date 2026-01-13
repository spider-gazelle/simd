require "./spec_helper"

describe "SIMD memory operations" do
  describe "copy" do
    it "scalar implementation" do
      simd = SIMD.scalar
      src = Slice[1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8,
        9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8, 15_u8, 16_u8]
      dst = Slice(UInt8).new(src.size)

      simd.copy(dst, src)

      src.size.times { |i| dst[i].should eq src[i] }
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        src = Slice[1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8,
          9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8, 15_u8, 16_u8]
        dst = Slice(UInt8).new(src.size)

        simd.copy(dst, src)

        src.size.times { |i| dst[i].should eq src[i] }
      end
    end
  end

  describe "fill" do
    it "scalar implementation" do
      simd = SIMD.scalar
      dst = Slice(UInt8).new(20)
      simd.fill(dst, 0xAB_u8)
      dst.all? { |v| v == 0xAB_u8 }.should be_true
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        dst = Slice(UInt8).new(20)
        simd.fill(dst, 0xAB_u8)
        dst.all? { |v| v == 0xAB_u8 }.should be_true
      end
    end
  end

  describe "xor_block16" do
    it "scalar implementation" do
      simd = SIMD.scalar
      src = Slice[0_u8, 1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8,
        8_u8, 9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8, 15_u8,
        16_u8, 17_u8, 18_u8, 19_u8, 20_u8, 21_u8, 22_u8, 23_u8]
      key = StaticArray[0xFF_u8, 0xFE_u8, 0xFD_u8, 0xFC_u8, 0xFB_u8, 0xFA_u8, 0xF9_u8, 0xF8_u8,
        0xF7_u8, 0xF6_u8, 0xF5_u8, 0xF4_u8, 0xF3_u8, 0xF2_u8, 0xF1_u8, 0xF0_u8]
      dst = Slice(UInt8).new(src.size)

      simd.xor_block16(dst, src, key)

      # Verify XOR operation
      src.size.times do |i|
        expected = src[i] ^ key[i & 15]
        dst[i].should eq expected
      end
    end

    it "hardware implementations match scalar" do
      check_implementations.each do |simd|
        src = Slice[0_u8, 1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8,
          8_u8, 9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8, 15_u8,
          16_u8, 17_u8, 18_u8, 19_u8, 20_u8, 21_u8, 22_u8, 23_u8]
        key = StaticArray[0xFF_u8, 0xFE_u8, 0xFD_u8, 0xFC_u8, 0xFB_u8, 0xFA_u8, 0xF9_u8, 0xF8_u8,
          0xF7_u8, 0xF6_u8, 0xF5_u8, 0xF4_u8, 0xF3_u8, 0xF2_u8, 0xF1_u8, 0xF0_u8]
        dst = Slice(UInt8).new(src.size)

        simd.xor_block16(dst, src, key)

        # Verify XOR operation
        src.size.times do |i|
          expected = src[i] ^ key[i & 15]
          dst[i].should eq expected
        end
      end
    end
  end
end
