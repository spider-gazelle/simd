module SIMD
  # ---------------------------
  # Windows IsProcessorFeaturePresent
  # ---------------------------
  {% if flag?(:win32) %}
    @[Link("kernel32")]
    lib Kernel32
      fun IsProcessorFeaturePresent(feature : UInt32) : Bool
    end

    # PROCESSOR_FEATURE_ID values (Windows)
    PF_AVX2_INSTRUCTIONS_AVAILABLE     = 40_u32
    PF_AVX512F_INSTRUCTIONS_AVAILABLE  = 41_u32
    PF_ARM_SVE_INSTRUCTIONS_AVAILABLE  = 46_u32
    PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE = 47_u32
  {% end %}

  # ---------------------------
  # x86 CPUID + XGETBV (inline asm)
  # Works on Windows/macOS/Linux/BSD for x86_64
  # ---------------------------
  {% if flag?(:x86_64) %}
    private def self.cpuid(eax_in : UInt32, ecx_in : UInt32) : Tuple(UInt32, UInt32, UInt32, UInt32)
      a = 0_u32
      b = 0_u32
      c = 0_u32
      d = 0_u32

      asm(
        "cpuid"
              : "={eax}"(a), "={ebx}"(b), "={ecx}"(c), "={edx}"(d)
              : "{eax}"(eax_in), "{ecx}"(ecx_in)
              : "memory"
      )

      {a, b, c, d}
    end

    private def self.xgetbv(index : UInt32) : UInt64
      lo = 0_u32
      hi = 0_u32

      asm(
        "xgetbv"
              : "={eax}"(lo), "={edx}"(hi)
              : "{ecx}"(index)
              : "memory"
      )

      (hi.to_u64 << 32) | lo.to_u64
    end

    private def self.detect_x86_basics(support : SupportedSIMD) : SupportedSIMD
      # On x86_64, SSE2 is baseline (ABI guarantee) → just set it.
      support |= SupportedSIMD::SSE2

      # Detect SSE4.1 via CPUID leaf 1 ECX bit 19
      # (No OSXSAVE/XGETBV required for SSE).
      _, _, ecx1, _ = cpuid(1_u32, 0_u32)
      support |= SupportedSIMD::SSE41 if (ecx1 & (1_u32 << 19)) != 0

      support
    end

    private def self.detect_x86_avx_nonwindows(support : SupportedSIMD) : SupportedSIMD
      max_leaf, _, _, _ = cpuid(0_u32, 0_u32)
      return support unless max_leaf >= 1

      # Leaf 1 ECX:
      # - OSXSAVE bit 27
      # - AVX bit 28
      _, _, ecx1, _ = cpuid(1_u32, 0_u32)
      has_osxsave = (ecx1 & (1_u32 << 27)) != 0
      has_avx = (ecx1 & (1_u32 << 28)) != 0
      return support unless has_osxsave && has_avx

      xcr0 = xgetbv(0_u32)

      # Need XMM (bit 1) + YMM (bit 2) enabled by OS for AVX/AVX2
      return support unless (xcr0 & 0x6) == 0x6

      if max_leaf >= 7
        _, ebx7, _, _ = cpuid(7_u32, 0_u32)

        # AVX2: EBX bit 5
        support |= SupportedSIMD::AVX2 if (ebx7 & (1_u32 << 5)) != 0

        # AVX-512F: EBX bit 16
        has_avx512f = (ebx7 & (1_u32 << 16)) != 0

        # AVX-512 state requires XCR0 bits 5..7 (opmask + ZMM)
        avx512_state_ok = (xcr0 & 0xE0) == 0xE0

        support |= SupportedSIMD::AVX512 if has_avx512f && avx512_state_ok
      end

      support
    end

    private def self.detect_avx512_subsets_cpuid : AVX512Subsets
      feats = AVX512Subsets::None
      max_leaf, _, _, _ = cpuid(0_u32, 0_u32)
      return feats unless max_leaf >= 7

      _, ebx, ecx, edx = cpuid(7_u32, 0_u32)

      feats |= AVX512Subsets::F if (ebx & (1_u32 << 16)) != 0
      feats |= AVX512Subsets::DQ if (ebx & (1_u32 << 17)) != 0
      feats |= AVX512Subsets::CD if (ebx & (1_u32 << 28)) != 0
      feats |= AVX512Subsets::BW if (ebx & (1_u32 << 30)) != 0
      feats |= AVX512Subsets::VL if (ebx & (1_u32 << 31)) != 0

      feats |= AVX512Subsets::VBMI if (ecx & (1_u32 << 1)) != 0
      feats |= AVX512Subsets::VBMI2 if (ecx & (1_u32 << 6)) != 0
      feats |= AVX512Subsets::VNNI if (ecx & (1_u32 << 11)) != 0
      feats |= AVX512Subsets::BITALG if (ecx & (1_u32 << 12)) != 0
      feats |= AVX512Subsets::VPOPCNTDQ if (ecx & (1_u32 << 14)) != 0

      feats |= AVX512Subsets::VP2INTERSECT if (edx & (1_u32 << 8)) != 0
      feats |= AVX512Subsets::FP16 if (edx & (1_u32 << 23)) != 0

      feats
    end
  {% end %}

  # ---------------------------
  # Linux AArch64 + (some) BSD AArch64:
  # Best-effort SVE via auxv file parsing (no getauxval)
  # ---------------------------
  {% if flag?(:aarch64) %}
    private AT_HWCAP  = 16_u64
    private HWCAP_SVE = 1_u64 << 22

    private def self.read_auxv_u64_pairs(path : String) : Hash(UInt64, UInt64)
      h = Hash(UInt64, UInt64).new
      File.open(path, "rb") do |io|
        buf = Bytes.new(16)
        while (n = io.read(buf)) == 16
          a_type = IO::ByteFormat::SystemEndian.decode(UInt64, buf[0, 8])
          a_val = IO::ByteFormat::SystemEndian.decode(UInt64, buf[8, 8])
          break if a_type == 0_u64 # AT_NULL
          h[a_type] = a_val
        end
      end
      h
    end

    private def self.try_read_auxv_hwcap(paths : Array(String)) : UInt64?
      paths.each do |path|
        next unless File.exists?(path)
        begin
          auxv = read_auxv_u64_pairs(path)
          return auxv[AT_HWCAP]? if auxv.has_key?(AT_HWCAP)
        rescue
        end
      end
      nil
    end
  {% end %}

  # ---------------------------
  # Public API
  # ---------------------------
  def self.supported_instruction_sets : SupportedSIMD
    support = SupportedSIMD::None

    {% if flag?(:x86_64) %}
      support = detect_x86_basics(support)

      {% if flag?(:win32) %}
        # Windows: prefer OS-reported usability for AVX2/AVX-512F
        support |= SupportedSIMD::AVX2   if Kernel32.IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE)
        support |= SupportedSIMD::AVX512 if Kernel32.IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE)
      {% else %}
        # macOS/Linux/BSD: CPUID + XGETBV for safe AVX/AVX-512 execution
        support = detect_x86_avx_nonwindows(support)
      {% end %}

    {% elsif flag?(:aarch64) %}
      # NEON is baseline on AArch64
      support |= SupportedSIMD::NEON

      {% if flag?(:win32) %}
        support |= SupportedSIMD::SVE  if Kernel32.IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
        support |= SupportedSIMD::SVE2 if Kernel32.IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE)

      {% elsif flag?(:linux) %}
        begin
          auxv = read_auxv_u64_pairs("/proc/self/auxv")
          hwcap = auxv[AT_HWCAP]? || 0_u64
          support |= SupportedSIMD::SVE if (hwcap & HWCAP_SVE) != 0
        rescue
        end

      {% elsif flag?(:bsd) %}
        # Best-effort: FreeBSD procfs commonly exposes /proc/curproc/auxv (if mounted)
        paths = [
          "/proc/curproc/auxv",
          "/proc/self/auxv",
          "/compat/linux/proc/self/auxv",
          "/compat/linux/proc/curproc/auxv",
        ]
        hwcap = try_read_auxv_hwcap(paths) || 0_u64
        support |= SupportedSIMD::SVE if (hwcap & HWCAP_SVE) != 0

      {% elsif flag?(:darwin) %}
        # Apple Silicon: NEON only in practice
      {% end %}
    {% end %}

    support
  end

  # Only meaningful if supported_instruction_sets includes AVX512
  def self.avx512_subsets : AVX512Subsets
    {% if flag?(:x86_64) %}
      {% if flag?(:win32) %}
        return AVX512Subsets::None unless Kernel32.IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE)
      {% end %}
      detect_avx512_subsets_cpuid
    {% else %}
      AVX512Subsets::None
    {% end %}
  end
end
