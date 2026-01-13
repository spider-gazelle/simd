require "./spec_helper"

describe SIMD do
  describe "detection" do
    it "detects CPU SIMD support" do
      supports = SIMD.supported_instruction_sets
      supports.none?.should be_false
      puts "SIMD support: #{supports}"

      {% if flag?(:x86_64) %}
        if supports.avx512?
          avx512_subsets = SIMD.avx512_subsets
          avx512_subsets.none?.should be_false
          puts "AVX512 Subset support: #{avx512_subsets}"
        end
      {% end %}
    end

    it "provides a hardware-accelerated instance" do
      simd = SIMD.instance
      simd.should_not be_nil
      puts "SIMD instance: #{simd.class}"
    end

    it "provides all available implementations" do
      impls = check_implementations
      impls.size.should be > 0
      puts "Available implementations: #{impls.map(&.class.name).join(", ")}"
    end
  end
end
