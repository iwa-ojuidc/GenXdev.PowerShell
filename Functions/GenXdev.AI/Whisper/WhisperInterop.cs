using System.Runtime.InteropServices;

namespace GenXdev.AI.Whisper
{
    /// <summary>
    /// Sampling strategy, mirrors enum whisper_sampling_strategy in whisper.h
    /// </summary>
    internal enum WhisperSamplingStrategy
    {
        Greedy = 0,
        BeamSearch = 1
    }

    /// <summary>
    /// Mirrors whisper_vad_params (24 bytes) from whisper.h
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct WhisperVadParams
    {
        public float Threshold;
        public int MinSpeechDurationMs;
        public int MinSilenceDurationMs;
        public float MaxSpeechDurationS;
        public int SpeechPadMs;
        public float SamplesOverlap;
    }

    /// <summary>
    /// Mirrors whisper_aheads (16 bytes) from whisper.h
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct WhisperAheads
    {
        public nuint NHeads;
        public IntPtr Heads;
    }

    /// <summary>
    /// Mirrors whisper_context_params (48 bytes) from whisper.h.
    /// Never construct this by hand - obtain it from
    /// whisper_context_default_params_by_ref and mutate.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct WhisperContextParams
    {
        public byte UseGpu;
        public byte FlashAttn;
        public int GpuDevice;
        public byte DtwTokenTimestamps;
        public int DtwAheadsPreset;
        public int DtwNTop;
        public WhisperAheads DtwAheads;
        public nuint DtwMemSize;
    }

    /// <summary>
    /// Mirrors whisper_full_params (304 bytes on win-x64) from whisper.h,
    /// whisper.cpp v1.9.1+ (commit 080bbbe8).
    ///
    /// Field order and types must match the C header exactly. All C bools are
    /// one byte, so they are declared as byte here - using C# bool would default
    /// to a four byte UnmanagedType.Bool and shift every subsequent field.
    /// C enums are four byte ints, size_t is eight bytes on win-x64.
    ///
    /// Never construct this by hand - obtain it from
    /// whisper_full_default_params_by_ref and mutate. WhisperInterop.EnsureLayoutValid
    /// verifies both the size and a set of sentinel defaults before first use.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct WhisperFullParams
    {
        public int Strategy;

        public int NThreads;
        public int NMaxTextCtx;
        public int OffsetMs;
        public int DurationMs;

        public byte Translate;
        public byte NoContext;
        public byte NoTimestamps;
        public byte SingleSegment;
        public byte PrintSpecial;
        public byte PrintProgress;
        public byte PrintRealtime;
        public byte PrintTimestamps;

        public byte TokenTimestamps;
        public float TholdPt;
        public float TholdPtsum;
        public int MaxLen;
        public byte SplitOnWord;
        public int MaxTokens;

        public byte DebugMode;
        public int AudioCtx;

        public byte TdrzEnable;

        public IntPtr SuppressRegex;

        public IntPtr InitialPrompt;
        public byte CarryInitialPrompt;
        public IntPtr PromptTokens;
        public int PromptNTokens;

        public IntPtr Language;
        public byte DetectLanguage;

        public byte SuppressBlank;
        public byte SuppressNst;

        public float Temperature;
        public float MaxInitialTs;
        public float LengthPenalty;

        public float TemperatureInc;
        public float EntropyThold;
        public float LogprobThold;
        public float NoSpeechThold;

        // struct { int best_of; } greedy;
        public int GreedyBestOf;

        // struct { int beam_size; float patience; } beam_search;
        public int BeamSearchBeamSize;
        public float BeamSearchPatience;

        public IntPtr NewSegmentCallback;
        public IntPtr NewSegmentCallbackUserData;

        public IntPtr ProgressCallback;
        public IntPtr ProgressCallbackUserData;

        public IntPtr EncoderBeginCallback;
        public IntPtr EncoderBeginCallbackUserData;

        public IntPtr AbortCallback;
        public IntPtr AbortCallbackUserData;

        public IntPtr LogitsFilterCallback;
        public IntPtr LogitsFilterCallbackUserData;

        public IntPtr GrammarRules;
        public nuint NGrammarRules;
        public nuint IStartRule;
        public float GrammarPenalty;

        public byte Vad;
        public IntPtr VadModelPath;
        public WhisperVadParams VadParams;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void WhisperNewSegmentCallback(
        IntPtr ctx, IntPtr state, int nNew, IntPtr userData);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void WhisperProgressCallback(
        IntPtr ctx, IntPtr state, int progress, IntPtr userData);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate byte GgmlAbortCallback(IntPtr userData);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void GgmlLogCallback(int level, IntPtr text, IntPtr userData);

    /// <summary>
    /// Raw P/Invoke surface over whisper.dll, plus native library resolution and
    /// a layout guard.
    ///
    /// The natives are the stock whisper.cpp/ggml shared libraries built with
    /// AVX/AVX2/AVX512/SSE42 disabled, so they run on any x64 CPU. They keep their
    /// original filenames on purpose: whisper.dll imports ggml.dll and ggml-base.dll
    /// by name, and ggml.dll imports ggml-cpu.dll, so renaming any of them breaks
    /// the import chain at load time.
    /// </summary>
    internal static class WhisperInterop
    {
        private const string LIB = "whisper";

        /// <summary>
        /// Extra directories to probe for the native libraries. Cmdlets populate
        /// this from their own module base, because PowerShell loads GenXdev.dll
        /// without a file-backed Assembly.Location - the assembly directory probe
        /// alone finds nothing.
        /// </summary>
        private static readonly List<string> _probeDirectories = new();

        private static readonly object _initLock = new object();

        private static bool _resolverRegistered;

        private static bool _layoutVerified;

        // Held for the process lifetime so the GC cannot collect the delegate
        // while whisper.dll still holds the function pointer.
        private static GgmlLogCallback _logCallback;

        // The sink SetLogHandler was given, reused so this layer's own
        // diagnostics reach the same destination as whisper's native logging.
        private static Action<string> _diagnosticSink;

        /// <summary>
        /// Registers a directory to probe when loading the native libraries.
        /// Read-only: it never creates or deletes anything.
        /// </summary>
        public static void AddProbeDirectory(string directory)
        {
            if (string.IsNullOrWhiteSpace(directory)) return;

            lock (_initLock)
            {
                if (!Directory.Exists(directory)) return;

                if (!_probeDirectories.Contains(directory,
                        StringComparer.OrdinalIgnoreCase))
                {
                    _probeDirectories.Add(directory);
                }
            }
        }

        /// <summary>
        /// Installs the DllImport resolver. Safe to call repeatedly.
        /// </summary>
        public static void EnsureResolverRegistered()
        {
            lock (_initLock)
            {
                if (_resolverRegistered) return;

                NativeLibrary.SetDllImportResolver(
                    typeof(WhisperInterop).Assembly, ResolveNativeLibrary);

                _resolverRegistered = true;
            }
        }

        /// <summary>
        /// Candidate directories in probe order: explicitly registered directories
        /// first, then the directory holding this assembly and its lib subfolder
        /// (which is empty when PowerShell loaded the assembly without a location).
        /// </summary>
        private static IEnumerable<string> GetCandidateDirectories()
        {
            List<string> dirs;

            lock (_initLock)
            {
                dirs = new List<string>(_probeDirectories);
            }

            string assemblyDir = null;

            try
            {
                var location = typeof(WhisperInterop).Assembly.Location;

                if (!string.IsNullOrEmpty(location))
                {
                    assemblyDir = Path.GetDirectoryName(location);
                }
            }
            catch
            {
                // Some load contexts throw rather than returning an empty location
            }

            if (!string.IsNullOrEmpty(assemblyDir))
            {
                dirs.Add(assemblyDir);
                dirs.Add(Path.Combine(assemblyDir, "lib"));
            }

            return dirs;
        }

        private static IntPtr ResolveNativeLibrary(
            string libraryName,
            System.Reflection.Assembly assembly,
            DllImportSearchPath? searchPath)
        {
            if (!string.Equals(libraryName, LIB, StringComparison.OrdinalIgnoreCase))
            {
                return IntPtr.Zero;
            }

            foreach (var dir in GetCandidateDirectories())
            {
                var candidate = Path.Combine(dir, "whisper.dll");

                if (!File.Exists(candidate)) continue;

                // Load the dependencies first, from the same directory, so the
                // loader never has to fall back to the process search path.
                // ggml-base has no ggml dependencies, ggml-cpu needs ggml-base,
                // and ggml needs both - so this is dependency order.
                foreach (var dep in new[]
                    { "ggml-base.dll", "ggml-cpu.dll", "ggml.dll" })
                {
                    var depPath = Path.Combine(dir, dep);

                    if (File.Exists(depPath))
                    {
                        NativeLibrary.TryLoad(depPath, out _);
                    }
                }

                if (NativeLibrary.TryLoad(candidate, out var handle))
                {
                    return handle;
                }
            }

            // Nothing matched - fall through to the default resolution so the
            // runtime produces its own DllNotFoundException.
            return IntPtr.Zero;
        }

        /// <summary>
        /// Describes where the resolver looked, for diagnostics when loading fails.
        /// </summary>
        public static string DescribeProbePaths()
        {
            return string.Join("; ", GetCandidateDirectories());
        }

        /// <summary>
        /// Verifies that the managed struct layout matches the native one before any
        /// call that would marshal it. A mismatch here would silently corrupt memory,
        /// so this checks the total size and a set of sentinel defaults that
        /// whisper_full_default_params is documented to produce.
        /// </summary>
        public static void EnsureLayoutValid()
        {
            lock (_initLock)
            {
                if (_layoutVerified) return;
            }

            EnsureResolverRegistered();

            var managedSize = Marshal.SizeOf<WhisperFullParams>();

            if (managedSize != 304)
            {
                throw new InvalidOperationException(
                    $"whisper_full_params layout mismatch: managed struct is " +
                    $"{managedSize} bytes, expected 304. The vendored whisper.dll " +
                    $"was built from an incompatible whisper.cpp revision.");
            }

            var contextSize = Marshal.SizeOf<WhisperContextParams>();

            if (contextSize != 48)
            {
                throw new InvalidOperationException(
                    $"whisper_context_params layout mismatch: managed struct is " +
                    $"{contextSize} bytes, expected 48.");
            }

            var ptr = whisper_full_default_params_by_ref(
                (int)WhisperSamplingStrategy.Greedy);

            if (ptr == IntPtr.Zero)
            {
                throw new InvalidOperationException(
                    "whisper_full_default_params_by_ref returned null.");
            }

            try
            {
                var p = Marshal.PtrToStructure<WhisperFullParams>(ptr);

                // Sentinels straight out of whisper.cpp whisper_full_default_params.
                // If the field order drifted, at least one of these lands wrong.
                var problems = new List<string>();

                if (p.NMaxTextCtx != 16384)
                    problems.Add($"n_max_text_ctx={p.NMaxTextCtx} (expected 16384)");

                if (Math.Abs(p.NoSpeechThold - 0.6f) > 0.0001f)
                    problems.Add($"no_speech_thold={p.NoSpeechThold} (expected 0.6)");

                if (Math.Abs(p.LengthPenalty - (-1.0f)) > 0.0001f)
                    problems.Add($"length_penalty={p.LengthPenalty} (expected -1)");

                if (Math.Abs(p.TemperatureInc - 0.2f) > 0.0001f)
                    problems.Add($"temperature_inc={p.TemperatureInc} (expected 0.2)");

                if (Math.Abs(p.EntropyThold - 2.4f) > 0.0001f)
                    problems.Add($"entropy_thold={p.EntropyThold} (expected 2.4)");

                if (p.GreedyBestOf != 5)
                    problems.Add($"greedy.best_of={p.GreedyBestOf} (expected 5)");

                if (Math.Abs(p.GrammarPenalty - 100.0f) > 0.0001f)
                    problems.Add($"grammar_penalty={p.GrammarPenalty} (expected 100)");

                if (problems.Count > 0)
                {
                    throw new InvalidOperationException(
                        "whisper_full_params layout mismatch - the managed struct " +
                        "does not line up with the native one: " +
                        string.Join(", ", problems) +
                        ". The vendored whisper.dll was built from an incompatible " +
                        "whisper.cpp revision; update WhisperFullParams in " +
                        "WhisperInterop.cs to match include/whisper.h.");
                }
            }
            finally
            {
                whisper_free_params(ptr);
            }

            lock (_initLock)
            {
                _layoutVerified = true;
            }
        }

        /// <summary>
        /// Routes whisper/ggml native logging into the supplied sink instead of
        /// stderr, so it cannot corrupt the PowerShell output stream.
        /// </summary>
        public static void SetLogHandler(Action<string> sink)
        {
            EnsureResolverRegistered();

            var callback = new GgmlLogCallback((level, text, userData) =>
            {
                if (sink == null || text == IntPtr.Zero) return;

                try
                {
                    var message = Marshal.PtrToStringUTF8(text);

                    if (!string.IsNullOrWhiteSpace(message))
                    {
                        sink(message.TrimEnd('\r', '\n'));
                    }
                }
                catch
                {
                    // Never let a logging failure escape into native code.
                }
            });

            lock (_initLock)
            {
                _logCallback = callback;

                _diagnosticSink = sink;
            }

            whisper_log_set(callback, IntPtr.Zero);
        }

        /// <summary>
        /// Emits a message from this binding layer into the same sink that
        /// receives whisper's native log lines. Safe to call from any thread -
        /// the cmdlets' sinks enqueue rather than write.
        /// </summary>
        public static void Log(string message)
        {
            Action<string> sink;

            lock (_initLock)
            {
                sink = _diagnosticSink;
            }

            if (sink == null || string.IsNullOrWhiteSpace(message)) return;

            try
            {
                sink(message);
            }
            catch
            {
                // Diagnostics must never break transcription
            }
        }

        #region whisper.dll imports

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr whisper_full_default_params_by_ref(int strategy);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern void whisper_free_params(IntPtr param);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr whisper_context_default_params_by_ref();

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern void whisper_free_context_params(IntPtr param);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr whisper_init_from_file_with_params(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string pathModel,
            WhisperContextParams param);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern void whisper_free(IntPtr ctx);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_full(
            IntPtr ctx, WhisperFullParams param, IntPtr samples, int nSamples);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_full_n_segments(IntPtr ctx);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr whisper_full_get_segment_text(
            IntPtr ctx, int iSegment);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern long whisper_full_get_segment_t0(
            IntPtr ctx, int iSegment);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern long whisper_full_get_segment_t1(
            IntPtr ctx, int iSegment);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern float whisper_full_get_segment_no_speech_prob(
            IntPtr ctx, int iSegment);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_full_n_tokens(IntPtr ctx, int iSegment);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern float whisper_full_get_token_p(
            IntPtr ctx, int iSegment, int iToken);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr whisper_full_get_token_text(
            IntPtr ctx, int iSegment, int iToken);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_is_multilingual(IntPtr ctx);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_full_lang_id(IntPtr ctx);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr whisper_lang_str(int id);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_lang_id(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string lang);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_lang_max_id();

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern void whisper_log_set(
            GgmlLogCallback logCallback, IntPtr userData);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr whisper_print_system_info();

        #endregion

        /// <summary>
        /// Marshals a native UTF-8 string that whisper owns. The returned pointer
        /// must never be freed by us.
        /// </summary>
        public static string PtrToString(IntPtr ptr)
        {
            return ptr == IntPtr.Zero ? string.Empty
                : (Marshal.PtrToStringUTF8(ptr) ?? string.Empty);
        }

        /// <summary>
        /// Reports the CPU feature set the loaded natives were built for.
        /// </summary>
        public static string GetSystemInfo()
        {
            EnsureResolverRegistered();

            return PtrToString(whisper_print_system_info());
        }
    }
}
