using UnityEngine;

namespace Undercurrent.Runtime.Presentation
{
    /// <summary>
    /// Side-on 2.5D fight camera: frames both fighters, zooms with their separation, applies settings-scaled
    /// trauma-based shake and a cinematic push-in for final hits.
    /// </summary>
    public sealed class FightCamera : MonoBehaviour
    {
        public Camera Camera { get; private set; }

        private Transform _a;
        private Transform _b;
        private float _trauma;
        private float _shakeScale = 1f;
        private Vector3 _focusOverride;
        private float _focusWeight;
        private float _focusTimer;
        private float _seed;
        private Vector3 _smoothedCenter;
        private float _smoothedDistance = 5f;

        public void Init(Transform a, Transform b, float shakeScale)
        {
            _a = a;
            _b = b;
            _shakeScale = shakeScale;
            Camera = gameObject.GetComponent<Camera>();
            if (Camera == null) Camera = gameObject.AddComponent<Camera>();
            Camera.fieldOfView = 38f;
            Camera.nearClipPlane = 0.1f;
            Camera.farClipPlane = 80f;
            Camera.clearFlags = CameraClearFlags.SolidColor;
            Camera.backgroundColor = new Color(0.02f, 0.02f, 0.03f);
            gameObject.tag = "MainCamera";
            _seed = Random.value * 100f;
            _smoothedCenter = Center();
            Snap();
        }

        public void SetShakeScale(float s) => _shakeScale = Mathf.Clamp01(s);

        /// <summary>Adds screen shake. Amount 0..1; scaled by the accessibility setting.</summary>
        public void Shake(float amount) => _trauma = Mathf.Clamp01(_trauma + amount * _shakeScale);

        /// <summary>Pushes the camera towards a point for a cinematic moment.</summary>
        public void Focus(Vector3 point, float seconds)
        {
            _focusOverride = point;
            _focusTimer = seconds;
        }

        private Vector3 Center()
        {
            if (_a == null || _b == null) return new Vector3(0, 1.1f, 0);
            var c = (_a.position + _b.position) * 0.5f;
            c.y = 1.15f + Mathf.Max(_a.position.y, _b.position.y) * 0.35f;
            return c;
        }

        private float Separation() => _a == null || _b == null ? 2f : Mathf.Abs(_a.position.x - _b.position.x);

        public void Snap()
        {
            _smoothedCenter = Center();
            _smoothedDistance = DistanceFor(Separation());
            Apply(0f);
        }

        private float DistanceFor(float separation)
        {
            // Keep both fighters inside the frame with margin at any aspect ratio.
            float aspect = Camera != null ? Mathf.Max(1.2f, Camera.aspect) : 1.78f;
            float halfFovH = Mathf.Atan(Mathf.Tan(Camera != null ? Camera.fieldOfView * 0.5f * Mathf.Deg2Rad : 0.33f) * aspect);
            float needed = (separation * 0.5f + 1.2f) / Mathf.Tan(halfFovH);
            return Mathf.Clamp(needed, 4.2f, 10f);
        }

        private void LateUpdate()
        {
            Apply(Time.unscaledDeltaTime);
        }

        private void Apply(float dt)
        {
            var target = Center();
            float distance = DistanceFor(Separation());

            if (_focusTimer > 0f)
            {
                _focusTimer -= dt;
                _focusWeight = Mathf.MoveTowards(_focusWeight, 1f, dt * 4f);
            }
            else _focusWeight = Mathf.MoveTowards(_focusWeight, 0f, dt * 2f);

            target = Vector3.Lerp(target, _focusOverride, _focusWeight * 0.6f);
            distance = Mathf.Lerp(distance, 3.2f, _focusWeight * 0.7f);

            float k = dt <= 0f ? 1f : 1f - Mathf.Exp(-6f * dt);
            _smoothedCenter = Vector3.Lerp(_smoothedCenter, target, k);
            _smoothedDistance = Mathf.Lerp(_smoothedDistance, distance, k);

            var pos = _smoothedCenter + new Vector3(0f, 0.55f, -_smoothedDistance);
            var rot = Quaternion.LookRotation(_smoothedCenter + Vector3.up * 0.05f - pos, Vector3.up);

            if (_trauma > 0f)
            {
                float shake = _trauma * _trauma;
                float t = Time.unscaledTime * 28f;
                pos += new Vector3(Mathf.PerlinNoise(_seed, t) - 0.5f, Mathf.PerlinNoise(_seed + 7f, t) - 0.5f, 0f) * 0.22f * shake;
                rot *= Quaternion.Euler(0f, 0f, (Mathf.PerlinNoise(_seed + 13f, t) - 0.5f) * 3f * shake);
                _trauma = Mathf.Max(0f, _trauma - dt * 2.4f);
            }
            transform.SetPositionAndRotation(pos, rot);
        }
    }
}
