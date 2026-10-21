using System;
using System.IO;
using System.Text;
using Undercurrent.Core.Progression;
using UnityEngine;

namespace Undercurrent.Runtime.Services
{
    public sealed class JsonUtilitySaveCodec : ISaveCodec
    {
        public string Encode(SaveData data) => JsonUtility.ToJson(data, false);

        public SaveData Decode(string text)
        {
            var data = JsonUtility.FromJson<SaveData>(text);
            if (data == null) throw new FormatException("Save file did not contain save data");
            return data;
        }
    }

    /// <summary>
    /// Stores the save as JSON in Application.persistentDataPath. Writes go to a temp file first and the previous
    /// save is kept as .bak, so an interrupted write cannot destroy progress.
    /// </summary>
    public sealed class FileSaveStore : ISaveStore
    {
        private readonly string _path;
        private readonly string _tmp;
        private readonly string _bak;
        private readonly string _dir;

        public FileSaveStore(string fileName = "undercurrent_save.json")
        {
            _dir = Application.persistentDataPath;
            _path = Path.Combine(_dir, fileName);
            _tmp = _path + ".tmp";
            _bak = _path + ".bak";
        }

        public string FilePath => _path;

        public bool TryRead(out string text)
        {
            text = null;
            foreach (var candidate in new[] { _path, _bak })
            {
                try
                {
                    if (!File.Exists(candidate)) continue;
                    text = File.ReadAllText(candidate, Encoding.UTF8);
                    if (!string.IsNullOrWhiteSpace(text)) return true;
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[Undercurrent] Could not read {candidate}: {e.Message}");
                }
            }
            return false;
        }

        public void Write(string text)
        {
            try
            {
                Directory.CreateDirectory(_dir);
                File.WriteAllText(_tmp, text, Encoding.UTF8);
                if (File.Exists(_path))
                {
                    File.Copy(_path, _bak, true);
                    File.Delete(_path);
                }
                File.Move(_tmp, _path);
            }
            catch (Exception e)
            {
                Debug.LogError("[Undercurrent] Save failed: " + e);
            }
        }

        public void WriteBackup(string text, string reason)
        {
            try
            {
                var name = $"{Path.GetFileNameWithoutExtension(_path)}.{reason}.{DateTime.UtcNow:yyyyMMddHHmmss}.json";
                File.WriteAllText(Path.Combine(_dir, name), text ?? string.Empty, Encoding.UTF8);
            }
            catch (Exception e)
            {
                Debug.LogWarning("[Undercurrent] Backup failed: " + e.Message);
            }
        }
    }

    /// <summary>
    /// Records unhandled exceptions to a local, size-capped log so testers can attach it to bug reports.
    /// This is local crash logging only — no crash reporting service is integrated.
    /// </summary>
    public static class CrashLog
    {
        private const long MaxBytes = 256 * 1024;
        private static string _path;
        private static readonly object Lock = new object();

        public static int ExceptionsThisSession { get; private set; }

        public static void Install()
        {
            _path = Path.Combine(Application.persistentDataPath, "undercurrent_errors.log");
            Application.logMessageReceivedThreaded -= OnLog;
            Application.logMessageReceivedThreaded += OnLog;
        }

        private static void OnLog(string condition, string stackTrace, LogType type)
        {
            if (type != LogType.Exception && type != LogType.Assert) return;
            lock (Lock)
            {
                ExceptionsThisSession++;
                try
                {
                    if (File.Exists(_path) && new FileInfo(_path).Length > MaxBytes) File.Delete(_path);
                    File.AppendAllText(_path, $"[{DateTime.UtcNow:O}] {type}: {condition}\n{stackTrace}\n");
                }
                catch
                {
                    // Never throw from a log callback.
                }
            }
        }
    }
}
