using System.Collections.Generic;
using Undercurrent.Runtime.Services;
using UnityEngine;

namespace Undercurrent.Runtime.Presentation
{
    /// <summary>
    /// One pre-built particle system per effect type, emitted on demand via EmitParams — no per-hit instantiation
    /// or garbage during combat. Density follows the graphics tier.
    /// </summary>
    public sealed class VfxPool : MonoBehaviour
    {
        private readonly Dictionary<string, ParticleSystem> _systems = new Dictionary<string, ParticleSystem>();
        private readonly Dictionary<string, int> _counts = new Dictionary<string, int>();

        public void Init(bool reduceFlashes)
        {
            float flash = reduceFlashes ? 0.5f : 1f;
            Create("spark_light", new Color(1f, 0.9f, 0.6f, 1f * flash), 0.18f, 3.5f, 0.05f, 10, 0.4f);
            Create("spark_heavy", new Color(1f, 0.75f, 0.35f, 1f * flash), 0.28f, 5f, 0.09f, 18, 0.5f);
            Create("spark_special", new Color(0.55f, 0.65f, 1f, 1f * flash), 0.45f, 6.5f, 0.12f, 30, 0.2f);
            Create("block_spark", new Color(0.55f, 0.85f, 1f, 0.9f * flash), 0.16f, 3f, 0.06f, 10, 0.1f);
            Create("parry_flash", new Color(0.9f, 1f, 1f, 1f * flash), 0.3f, 2.2f, 0.18f, 16, 0f);
            Create("dust", new Color(0.55f, 0.48f, 0.4f, 0.55f), 0.7f, 1.2f, 0.25f, 14, -0.05f);
            Create("sweat", new Color(0.85f, 0.9f, 1f, 0.7f), 0.35f, 2.5f, 0.025f, 8, 1f);
        }

        private void Create(string id, Color color, float lifetime, float speed, float size, int count, float gravity)
        {
            var go = new GameObject("VFX_" + id);
            go.transform.SetParent(transform, false);
            var ps = go.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);

            var main = ps.main;
            main.playOnAwake = false;
            main.loop = false;
            main.duration = 1f;
            main.startLifetime = new ParticleSystem.MinMaxCurve(lifetime * 0.6f, lifetime);
            main.startSpeed = new ParticleSystem.MinMaxCurve(speed * 0.4f, speed);
            main.startSize = new ParticleSystem.MinMaxCurve(size * 0.6f, size);
            main.startColor = color;
            main.gravityModifier = gravity;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.maxParticles = count * 6;

            var emission = ps.emission;
            emission.enabled = false;

            var shape = ps.shape;
            shape.enabled = true;
            shape.shapeType = ParticleSystemShapeType.Sphere;
            shape.radius = 0.05f;

            var col = ps.colorOverLifetime;
            col.enabled = true;
            var g = new Gradient();
            g.SetKeys(new[] { new GradientColorKey(Color.white, 0f), new GradientColorKey(Color.white, 1f) },
                new[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(0f, 1f) });
            col.color = new ParticleSystem.MinMaxGradient(g);

            var sizeOverLife = ps.sizeOverLifetime;
            sizeOverLife.enabled = true;
            sizeOverLife.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 1f, 1f, 0.2f));

            var renderer = go.GetComponent<ParticleSystemRenderer>();
            renderer.sharedMaterial = VisualAssets.ParticleBase;
            renderer.renderMode = ParticleSystemRenderMode.Billboard;
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;

            _systems[id] = ps;
            _counts[id] = count;
        }

        public void Play(string id, Vector3 position, int facing = 1)
        {
            if (string.IsNullOrEmpty(id) || !_systems.TryGetValue(id, out var ps)) return;
            var emit = new ParticleSystem.EmitParams
            {
                position = position,
                applyShapeToPosition = true,
            };
            int n = Mathf.Max(1, Mathf.RoundToInt(_counts[id] * QualityService.VfxDensity));
            ps.Emit(emit, n);
        }
    }
}
