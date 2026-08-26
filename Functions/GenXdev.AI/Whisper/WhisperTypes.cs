namespace GenXdev.AI.Whisper
{
    /// <summary>
    /// Whisper model variants available for download.
    ///
    /// The member names are kept exactly as they were so that -ModelType keeps
    /// binding from the strings the PowerShell wrappers pass ('tiny',
    /// 'LargeV3Turbo', ...) and existing scripts keep working unchanged.
    /// </summary>
    public enum GgmlType
    {
        Tiny = 0,
        TinyEn = 1,
        Base = 2,
        BaseEn = 3,
        Small = 4,
        SmallEn = 5,
        Medium = 6,
        MediumEn = 7,
        LargeV1 = 8,
        LargeV2 = 9,
        LargeV3 = 10,
        LargeV3Turbo = 11,

        // Quantized variants. On a CPU without AVX the work is largely
        // memory-bandwidth bound, so these are noticeably faster than their
        // full-precision counterparts at roughly a third of the size - the
        // practical choice on low-end hardware.
        // Appended deliberately: the existing values above must never be
        // renumbered, since the member names are the public contract for
        // -ModelType and the numbers are recorded in generated metadata.
        TinyQ5_1 = 12,
        TinyEnQ5_1 = 13,
        BaseQ5_1 = 14,
        BaseEnQ5_1 = 15,
        SmallQ5_1 = 16,
        SmallEnQ5_1 = 17,
        MediumQ5_0 = 18,
        MediumEnQ5_0 = 19,
        LargeV2Q5_0 = 20,
        LargeV3Q5_0 = 21,
        LargeV3TurboQ5_0 = 22
    }

    /// <summary>
    /// Maps model variants onto the canonical whisper.cpp model names and the
    /// files that hold them.
    /// </summary>
    public static class GgmlTypeExtensions
    {
        /// <summary>
        /// The canonical whisper.cpp name, e.g. GgmlType.LargeV3Turbo becomes
        /// "large-v3-turbo". This is what the ggerganov/whisper.cpp HuggingFace
        /// repository publishes.
        /// </summary>
        public static string ToCanonicalName(this GgmlType modelType)
        {
            switch (modelType)
            {
                case GgmlType.Tiny: return "tiny";
                case GgmlType.TinyEn: return "tiny.en";
                case GgmlType.Base: return "base";
                case GgmlType.BaseEn: return "base.en";
                case GgmlType.Small: return "small";
                case GgmlType.SmallEn: return "small.en";
                case GgmlType.Medium: return "medium";
                case GgmlType.MediumEn: return "medium.en";
                case GgmlType.LargeV1: return "large-v1";
                case GgmlType.LargeV2: return "large-v2";
                case GgmlType.LargeV3: return "large-v3";
                case GgmlType.LargeV3Turbo: return "large-v3-turbo";
                case GgmlType.TinyQ5_1: return "tiny-q5_1";
                case GgmlType.TinyEnQ5_1: return "tiny.en-q5_1";
                case GgmlType.BaseQ5_1: return "base-q5_1";
                case GgmlType.BaseEnQ5_1: return "base.en-q5_1";
                case GgmlType.SmallQ5_1: return "small-q5_1";
                case GgmlType.SmallEnQ5_1: return "small.en-q5_1";
                case GgmlType.MediumQ5_0: return "medium-q5_0";
                case GgmlType.MediumEnQ5_0: return "medium.en-q5_0";
                case GgmlType.LargeV2Q5_0: return "large-v2-q5_0";
                case GgmlType.LargeV3Q5_0: return "large-v3-q5_0";
                case GgmlType.LargeV3TurboQ5_0: return "large-v3-turbo-q5_0";
                default:
                    throw new ArgumentOutOfRangeException(
                        nameof(modelType), modelType, "Unknown Whisper model type");
            }
        }

        /// <summary>
        /// The canonical model filename, e.g. "ggml-large-v3-turbo.bin".
        /// </summary>
        public static string ToModelFileName(this GgmlType modelType)
        {
            return $"ggml-{modelType.ToCanonicalName()}.bin";
        }

        /// <summary>
        /// Candidate filenames for a model, most preferred first. The canonical
        /// whisper.cpp name comes first so models downloaded by hand or by
        /// whisper.cpp's own download script are picked up; the legacy
        /// "ggml-{EnumName}.bin" form is still honoured so previously downloaded
        /// models are not orphaned.
        /// </summary>
        public static IEnumerable<string> GetModelFileNameCandidates(
            this GgmlType modelType)
        {
            yield return modelType.ToModelFileName();

            var legacy = $"ggml-{modelType}.bin";

            if (!string.Equals(legacy, modelType.ToModelFileName(),
                    StringComparison.OrdinalIgnoreCase))
            {
                yield return legacy;
            }
        }
    }

    /// <summary>
    /// A single token produced by the model.
    /// </summary>
    public class WhisperToken
    {
        public string Text { get; set; }

        public float Probability { get; set; }
    }

    /// <summary>
    /// One transcribed segment of audio.
    ///
    /// Consumers read Text, Start and End (Start-AudioTranscription's SRT output
    /// depends on the two timestamps); the remaining properties round out the
    /// shape that -Passthru used to emit.
    /// </summary>
    public class SegmentData
    {
        public SegmentData(
            string text,
            TimeSpan start,
            TimeSpan end,
            float minProbability,
            float maxProbability,
            float probability,
            float noSpeechProbability,
            string language,
            WhisperToken[] tokens)
        {
            Text = text;
            Start = start;
            End = end;
            MinProbability = minProbability;
            MaxProbability = maxProbability;
            Probability = probability;
            NoSpeechProbability = noSpeechProbability;
            Language = language;
            Tokens = tokens;
        }

        /// <summary>The transcribed text of this segment.</summary>
        public string Text { get; }

        /// <summary>Offset of the segment start within the audio.</summary>
        public TimeSpan Start { get; }

        /// <summary>Offset of the segment end within the audio.</summary>
        public TimeSpan End { get; }

        /// <summary>Lowest token probability in this segment.</summary>
        public float MinProbability { get; }

        /// <summary>Highest token probability in this segment.</summary>
        public float MaxProbability { get; }

        /// <summary>Mean token probability across the segment.</summary>
        public float Probability { get; }

        /// <summary>Probability that this segment contains no speech.</summary>
        public float NoSpeechProbability { get; }

        /// <summary>Detected or configured language code.</summary>
        public string Language { get; }

        /// <summary>Individual tokens making up this segment.</summary>
        public WhisperToken[] Tokens { get; }

        public override string ToString() => Text ?? string.Empty;
    }
}
