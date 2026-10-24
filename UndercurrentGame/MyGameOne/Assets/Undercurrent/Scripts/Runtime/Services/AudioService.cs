using System.Collections.Generic;
using Undercurrent.Core.Progression;
using UnityEngine;

namespace Undercurrent.Runtime.Services
{
    /// <summary>
    /// Music, ambience, SFX and voice playback with independent volume channels. Clips are looked up first in
    /// Resources/Undercurrent/Audio/&lt;id&gt; (licensed replacements) and fall back to the procedural library.
    /// </summary>
    public sealed class AudioService : MonoBehaviour
    {
        private const int SfxVoices = 14;

        private Dictionary<string, AudioClip> _library;
        private readonly Dictionary<string, AudioClip> _overrides = new Dictionary<string, AudioClip>();
        private AudioSource[] _music;
        private int _activeMusic;
        private AudioSource _ambience;
        private AudioSource[] _sfx;
        private int _nextSfx;
        private AudioSource _voice;
        private SettingsData _settings;
        private float _musicFade = 1f;
        private string _currentMusic;

        public void Init(SettingsData settings)
        {
            _settings = settings;
            _library = ProceduralAudio.BuildLibrary();
            _music = new[] { NewSource("Music A", true), NewSource("Music B", true) };
            _ambience = NewSource("Ambience", true);
            _voice = NewSource("Voice", false);
            _sfx = new AudioSource[SfxVoices];
            for (int i = 0; i < SfxVoices; i++) _sfx[i] = NewSource("Sfx " + i, false);
            ApplyVolumes();
        }

        private AudioSource NewSource(string name, bool loop)
        {
            var go = new GameObject(name);
            go.transform.SetParent(transform, false);
            var s = go.AddComponent<AudioSource>();
            s.playOnAwake = false;
            s.loop = loop;
            s.spatialBlend = 0f;
            return s;
        }

        public void ApplyVolumes()
        {
            if (_settings == null || _music == null) return;
            _ambience.volume = 0.6f * _settings.SfxVolume;
            _voice.volume = _settings.VoiceVolume;
        }

        private AudioClip Get(string id)
        {
            if (string.IsNullOrEmpty(id)) return null;
            if (!_overrides.TryGetValue(id, out var licensed))
            {
                // Looked up once per id; a null entry means "no licensed replacement, use procedural".
                licensed = Resources.Load<AudioClip>("Undercurrent/Audio/" + id);
                _overrides[id] = licensed;
            }
            if (licensed != null) return licensed;
            return _library != null && _library.TryGetValue(id, out var c) ? c : null;
        }

        public void PlaySfx(string id, float volume = 1f, float pitchJitter = 0.06f, float pan = 0f)
        {
            var clip = Get(id);
            if (clip == null || _settings == null) return;
            var s = _sfx[_nextSfx];
            _nextSfx = (_nextSfx + 1) % _sfx.Length;
            s.Stop();
            s.clip = clip;
            s.volume = Mathf.Clamp01(volume) * _settings.SfxVolume;
            s.pitch = 1f + Random.Range(-pitchJitter, pitchJitter);
            s.panStereo = Mathf.Clamp(pan, -1f, 1f);
            s.Play();
        }

        public void PlayVoice(string id, float volume = 1f)
        {
            var clip = Get(id);
            if (clip == null || _settings == null) return;
            _voice.pitch = 1f + Random.Range(-0.08f, 0.08f);
            _voice.PlayOneShot(clip, volume * _settings.VoiceVolume);
        }

        public void PlayUI(string id) => PlaySfx(id, 0.8f, 0f);

        public void PlayMusic(string id)
        {
            if (_music == null || id == _currentMusic) return;
            var clip = Get(id);
            if (clip == null) return;
            _currentMusic = id;
            _activeMusic = 1 - _activeMusic;
            var src = _music[_activeMusic];
            src.clip = clip;
            src.volume = 0f;
            src.Play();
            _musicFade = 0f;
        }

        public void PlayAmbience(string id)
        {
            if (_ambience == null) return;
            var clip = Get(id);
            if (clip == null) { _ambience.Stop(); return; }
            if (_ambience.clip == clip && _ambience.isPlaying) return;
            _ambience.clip = clip;
            _ambience.Play();
        }

        public void StopAmbience()
        {
            if (_ambience != null) _ambience.Stop();
        }

        private void Update()
        {
            if (_music == null || _settings == null) return;
            _musicFade = Mathf.MoveTowards(_musicFade, 1f, Time.unscaledDeltaTime / 1.2f);
            float target = 0.55f * _settings.MusicVolume;
            _music[_activeMusic].volume = target * _musicFade;
            var other = _music[1 - _activeMusic];
            other.volume = target * (1f - _musicFade);
            if (_musicFade >= 1f && other.isPlaying) other.Stop();
            _ambience.volume = 0.6f * _settings.SfxVolume;
        }
    }
}
