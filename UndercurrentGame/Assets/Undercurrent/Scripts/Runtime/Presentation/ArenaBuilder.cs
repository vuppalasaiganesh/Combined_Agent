using Undercurrent.Core.Content;
using Undercurrent.Runtime.Services;
using UnityEngine;
using UnityEngine.Rendering;

namespace Undercurrent.Runtime.Presentation
{
    /// <summary>
    /// Builds the three original arenas from simple geometry and lights. This is greybox/placeholder art that
    /// establishes layout, lighting mood and stage bounds; final environment art is listed in ASSET_LICENSES.md as missing.
    /// Light counts are kept within URP's default per-object limits for mobile.
    /// </summary>
    public static class ArenaBuilder
    {
        public static GameObject Build(ArenaDefinition arena)
        {
            var root = new GameObject("Arena_" + arena.Id);
            switch (arena.VisualPreset)
            {
                case "freightyard": BuildFreightYard(root.transform, arena); break;
                case "cistern": BuildCistern(root.transform, arena); break;
                case "market": BuildMarket(root.transform, arena); break;
                case "rooftop": BuildRooftop(root.transform, arena); break;
                case "forge": BuildForge(root.transform, arena); break;
                case "finals": BuildFinals(root.transform, arena); break;
                default: BuildNetLoft(root.transform, arena); break;
            }
            return root;
        }

        private static GameObject Box(Transform parent, string name, Vector3 pos, Vector3 size, Material mat, bool castShadows = true)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            Object.Destroy(go.GetComponent<Collider>());
            go.transform.SetParent(parent, false);
            go.transform.localPosition = pos;
            go.transform.localScale = size;
            var r = go.GetComponent<MeshRenderer>();
            r.sharedMaterial = mat;
            r.shadowCastingMode = castShadows ? ShadowCastingMode.On : ShadowCastingMode.Off;
            go.isStatic = true;
            return go;
        }

        private static GameObject Cyl(Transform parent, string name, Vector3 pos, Vector3 size, Material mat, Vector3? euler = null)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            go.name = name;
            Object.Destroy(go.GetComponent<Collider>());
            go.transform.SetParent(parent, false);
            go.transform.localPosition = pos;
            go.transform.localScale = size;
            if (euler.HasValue) go.transform.localEulerAngles = euler.Value;
            go.GetComponent<MeshRenderer>().sharedMaterial = mat;
            go.isStatic = true;
            return go;
        }

        private static Light AddLight(Transform parent, LightType type, Vector3 pos, Vector3 euler, Color color, float intensity, float range = 10f, float spot = 60f, bool shadows = false)
        {
            var go = new GameObject(type + " Light");
            go.transform.SetParent(parent, false);
            go.transform.localPosition = pos;
            go.transform.localEulerAngles = euler;
            var l = go.AddComponent<Light>();
            l.type = type;
            l.color = color;
            l.intensity = intensity;
            l.range = range;
            if (type == LightType.Spot) l.spotAngle = spot;
            l.shadows = shadows && QualityService.ShadowsEnabled ? LightShadows.Soft : LightShadows.None;
            return l;
        }

        private static void Ambient(Color sky, Color equator, Color ground, Color fog, float fogDensity)
        {
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = sky;
            RenderSettings.ambientEquatorColor = equator;
            RenderSettings.ambientGroundColor = ground;
            RenderSettings.fog = fogDensity > 0f;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogColor = fog;
            RenderSettings.fogDensity = fogDensity;
        }

        private static void Lamp(Transform parent, Vector3 pos, Color color, float intensity)
        {
            var bulb = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            Object.Destroy(bulb.GetComponent<Collider>());
            bulb.transform.SetParent(parent, false);
            bulb.transform.localPosition = pos;
            bulb.transform.localScale = Vector3.one * 0.22f;
            var m = VisualAssets.Unlit(color * 1.5f);
            bulb.GetComponent<MeshRenderer>().sharedMaterial = m;
            bulb.GetComponent<MeshRenderer>().shadowCastingMode = ShadowCastingMode.Off;
            Cyl(parent, "Cable", pos + Vector3.up * 1.2f, new Vector3(0.02f, 1.2f, 0.02f), VisualAssets.Lit(Color.black));
            AddLight(parent, LightType.Point, pos + Vector3.down * 0.2f, Vector3.zero, color, intensity, 9f);
        }

        // ------------------------------------------------------------------ Pier 9 Net-Loft

        private static void BuildNetLoft(Transform root, ArenaDefinition a)
        {
            Ambient(new Color(0.28f, 0.3f, 0.36f), new Color(0.22f, 0.18f, 0.14f), new Color(0.1f, 0.08f, 0.06f), new Color(0.08f, 0.07f, 0.06f), 0.012f);
            var plank = VisualAssets.Lit(new Color(0.36f, 0.25f, 0.16f), 0.3f);
            var plankDark = VisualAssets.Lit(new Color(0.24f, 0.16f, 0.1f), 0.25f);
            var beam = VisualAssets.Lit(new Color(0.18f, 0.12f, 0.08f), 0.2f);
            var rope = VisualAssets.Lit(new Color(0.62f, 0.55f, 0.38f), 0.1f);
            var net = VisualAssets.Lit(new Color(0.16f, 0.28f, 0.3f), 0.1f);

            float w = a.StageHalfWidth + 3f;
            for (int i = -10; i <= 10; i++)
                Box(root, "Plank", new Vector3(i * 0.9f, -0.05f, 1.5f), new Vector3(0.86f, 0.1f, 9f), (i & 1) == 0 ? plank : plankDark, false);
            Box(root, "BackWall", new Vector3(0, 3f, 4.5f), new Vector3(w * 2.5f, 6f, 0.3f), plankDark);
            for (int i = -4; i <= 4; i++)
                Box(root, "Post", new Vector3(i * 2.6f, 3f, 4.2f), new Vector3(0.3f, 6f, 0.3f), beam);
            Box(root, "Rafter", new Vector3(0, 5.2f, 2f), new Vector3(w * 2.5f, 0.25f, 0.3f), beam);
            Box(root, "NetHang", new Vector3(-3.5f, 3.4f, 4.2f), new Vector3(3.2f, 2.2f, 0.05f), net, false);
            Box(root, "NetHang2", new Vector3(4.2f, 3.0f, 4.2f), new Vector3(2.4f, 1.8f, 0.05f), net, false);
            for (int i = 0; i < 3; i++)
                Cyl(root, "RopeCoil", new Vector3(-a.StageHalfWidth - 0.8f + i * 0.3f, 0.12f + i * 0.02f, 2.6f - i * 0.6f), new Vector3(0.8f, 0.12f, 0.8f), rope);
            Box(root, "Crate", new Vector3(a.StageHalfWidth + 1.1f, 0.5f, 2.5f), new Vector3(1f, 1f, 1f), plank);
            Box(root, "Crate2", new Vector3(a.StageHalfWidth + 1.0f, 1.4f, 2.7f), new Vector3(0.8f, 0.8f, 0.8f), plankDark);
            Box(root, "WindowGlow", new Vector3(0.8f, 3.4f, 4.33f), new Vector3(2.2f, 1.3f, 0.02f), VisualAssets.Unlit(new Color(0.35f, 0.45f, 0.65f)), false);

            AddLight(root, LightType.Directional, new Vector3(0, 6, -2), new Vector3(42f, -28f, 0f), new Color(0.62f, 0.7f, 0.9f), 0.55f, shadows: true);
            Lamp(root, new Vector3(-2.2f, 3.6f, 0.6f), new Color(1f, 0.66f, 0.32f), 2.4f);
            Lamp(root, new Vector3(2.2f, 3.6f, 0.6f), new Color(1f, 0.66f, 0.32f), 2.4f);
        }

        // ------------------------------------------------------------------ Calloway Freight Yard

        private static void BuildFreightYard(Transform root, ArenaDefinition a)
        {
            Ambient(new Color(0.18f, 0.22f, 0.3f), new Color(0.12f, 0.14f, 0.18f), new Color(0.05f, 0.05f, 0.06f), new Color(0.1f, 0.12f, 0.16f), 0.02f);
            var concrete = VisualAssets.Lit(new Color(0.22f, 0.23f, 0.25f), 0.78f); // wet, reflective
            Color[] containerColors = { new Color(0.55f, 0.16f, 0.12f), new Color(0.12f, 0.3f, 0.45f), new Color(0.6f, 0.45f, 0.12f), new Color(0.2f, 0.36f, 0.2f) };
            Box(root, "Ground", new Vector3(0, -0.05f, 2f), new Vector3(40f, 0.1f, 14f), concrete, false);
            for (int i = 0; i < 6; i++)
                Box(root, "Puddle", new Vector3(-5f + i * 2.1f, 0.002f, 0.6f + (i % 3) * 0.7f), new Vector3(1.2f, 0.004f, 0.6f), VisualAssets.Lit(new Color(0.08f, 0.09f, 0.11f), 0.95f), false);

            float edge = a.StageHalfWidth + 1.8f;
            for (int row = 0; row < 2; row++)
            for (int i = -3; i <= 3; i++)
            {
                var mat = VisualAssets.Lit(containerColors[(i + 3 + row) % containerColors.Length], 0.35f, 0.4f);
                var c = Box(root, "Container", new Vector3(i * 6.2f, 1.3f + row * 2.6f, 5f), new Vector3(6f, 2.55f, 2.4f), mat);
                for (int r = -5; r <= 5; r++)
                    Box(c.transform, "Rib", new Vector3(r * 0.09f, 0f, -0.51f), new Vector3(0.012f, 0.95f, 0.02f), mat, false);
            }
            Box(root, "SideStackL", new Vector3(-edge - 1.2f, 1.3f, 1.5f), new Vector3(2.4f, 2.55f, 6f), VisualAssets.Lit(containerColors[1], 0.35f, 0.4f));
            Box(root, "SideStackR", new Vector3(edge + 1.2f, 1.3f, 1.5f), new Vector3(2.4f, 2.55f, 6f), VisualAssets.Lit(containerColors[0], 0.35f, 0.4f));

            var pole = VisualAssets.Lit(new Color(0.3f, 0.3f, 0.32f), 0.5f, 0.6f);
            Cyl(root, "LightPoleL", new Vector3(-4.5f, 4f, 3.2f), new Vector3(0.15f, 4f, 0.15f), pole);
            Cyl(root, "LightPoleR", new Vector3(4.5f, 4f, 3.2f), new Vector3(0.15f, 4f, 0.15f), pole);
            AddLight(root, LightType.Spot, new Vector3(-4.5f, 7.6f, 2.6f), new Vector3(68f, 30f, 0f), new Color(0.92f, 0.96f, 1f), 18f, 18f, 70f, shadows: true);
            AddLight(root, LightType.Spot, new Vector3(4.5f, 7.6f, 2.6f), new Vector3(68f, -30f, 0f), new Color(0.92f, 0.96f, 1f), 14f, 18f, 70f);
            AddLight(root, LightType.Directional, new Vector3(0, 6, -2), new Vector3(30f, 20f, 0f), new Color(0.35f, 0.42f, 0.6f), 0.35f);
        }

        // ------------------------------------------------------------------ The Old Cistern

        private static void BuildCistern(Transform root, ArenaDefinition a)
        {
            Ambient(new Color(0.12f, 0.14f, 0.14f), new Color(0.12f, 0.1f, 0.08f), new Color(0.03f, 0.04f, 0.04f), new Color(0.05f, 0.06f, 0.06f), 0.03f);
            var brick = VisualAssets.Lit(new Color(0.42f, 0.22f, 0.16f), 0.25f);
            var brickDark = VisualAssets.Lit(new Color(0.28f, 0.15f, 0.11f), 0.25f);
            var stone = VisualAssets.Lit(new Color(0.3f, 0.3f, 0.28f), 0.4f);
            var water = VisualAssets.Lit(new Color(0.03f, 0.06f, 0.07f), 0.97f);

            Box(root, "Walkway", new Vector3(0, -0.05f, 1.2f), new Vector3(a.StageHalfWidth * 2f + 4f, 0.1f, 4f), stone, false);
            Box(root, "Water", new Vector3(0, -0.12f, 6f), new Vector3(40f, 0.02f, 8f), water, false);
            for (int i = -4; i <= 4; i++)
            {
                float x = i * 3.4f;
                Box(root, "Column", new Vector3(x, 2.4f, 4.6f), new Vector3(0.7f, 4.8f, 0.7f), (i & 1) == 0 ? brick : brickDark);
                Box(root, "Arch", new Vector3(x + 1.7f, 4.6f, 4.6f), new Vector3(2.8f, 0.5f, 0.7f), brickDark);
            }
            Box(root, "BackWall", new Vector3(0, 3f, 8.5f), new Vector3(40f, 6f, 0.5f), brickDark);
            Box(root, "EndWallL", new Vector3(-a.StageHalfWidth - 1.6f, 2.5f, 1.5f), new Vector3(0.6f, 5f, 6f), brick);
            Box(root, "EndWallR", new Vector3(a.StageHalfWidth + 1.6f, 2.5f, 1.5f), new Vector3(0.6f, 5f, 6f), brick);

            var tripod = VisualAssets.Lit(new Color(0.85f, 0.65f, 0.1f), 0.4f, 0.3f);
            Box(root, "WorkLightL", new Vector3(-3.8f, 1.6f, 3.2f), new Vector3(0.35f, 0.25f, 0.2f), tripod);
            Box(root, "WorkLightR", new Vector3(3.8f, 1.6f, 3.2f), new Vector3(0.35f, 0.25f, 0.2f), tripod);
            AddLight(root, LightType.Spot, new Vector3(-3.8f, 1.7f, 3.0f), new Vector3(10f, 150f, 0f), new Color(1f, 0.82f, 0.6f), 9f, 14f, 85f, shadows: true);
            AddLight(root, LightType.Spot, new Vector3(3.8f, 1.7f, 3.0f), new Vector3(10f, 210f, 0f), new Color(1f, 0.82f, 0.6f), 7f, 14f, 85f);
            AddLight(root, LightType.Directional, new Vector3(0, 6, -2), new Vector3(55f, 0f, 0f), new Color(0.4f, 0.5f, 0.55f), 0.25f);
        }

        // ------------------------------------------------------------------ Lantern Row Night Market

        private static void BuildMarket(Transform root, ArenaDefinition a)
        {
            Ambient(new Color(0.16f, 0.12f, 0.18f), new Color(0.2f, 0.12f, 0.1f), new Color(0.06f, 0.05f, 0.05f), new Color(0.09f, 0.06f, 0.07f), 0.018f);
            var tarmac = VisualAssets.Lit(new Color(0.16f, 0.16f, 0.17f), 0.55f);
            Box(root, "Street", new Vector3(0, -0.05f, 2f), new Vector3(40f, 0.1f, 14f), tarmac, false);
            Color[] canopy = { new Color(0.65f, 0.15f, 0.12f), new Color(0.12f, 0.35f, 0.4f), new Color(0.75f, 0.55f, 0.15f) };
            for (int i = -4; i <= 4; i++)
            {
                var stall = new Vector3(i * 2.6f, 0f, 4.4f);
                Box(root, "Counter", stall + new Vector3(0, 0.5f, 0), new Vector3(2.2f, 1f, 1f), VisualAssets.Lit(new Color(0.35f, 0.24f, 0.16f), 0.3f));
                Box(root, "Canopy", stall + new Vector3(0, 2.3f, 0.1f), new Vector3(2.4f, 0.08f, 1.6f), VisualAssets.Lit(canopy[(i + 4) % 3], 0.2f));
                Cyl(root, "Pole", stall + new Vector3(-1.1f, 1.15f, -0.6f), new Vector3(0.05f, 1.15f, 0.05f), VisualAssets.Lit(Color.gray, 0.4f, 0.5f));
            }
            var lantern = VisualAssets.Unlit(new Color(1f, 0.55f, 0.25f) * 1.4f);
            for (int i = -6; i <= 6; i++)
            {
                var ball = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                Object.Destroy(ball.GetComponent<Collider>());
                ball.transform.SetParent(root, false);
                ball.transform.localPosition = new Vector3(i * 1.1f, 3.2f + 0.15f * Mathf.Sin(i), 2.6f);
                ball.transform.localScale = new Vector3(0.28f, 0.36f, 0.28f);
                ball.GetComponent<MeshRenderer>().sharedMaterial = lantern;
                ball.GetComponent<MeshRenderer>().shadowCastingMode = ShadowCastingMode.Off;
            }
            AddLight(root, LightType.Point, new Vector3(-2.5f, 3f, 1.8f), Vector3.zero, new Color(1f, 0.6f, 0.3f), 2.2f, 9f);
            AddLight(root, LightType.Point, new Vector3(2.5f, 3f, 1.8f), Vector3.zero, new Color(1f, 0.6f, 0.3f), 2.2f, 9f);
            AddLight(root, LightType.Directional, new Vector3(0, 6, -2), new Vector3(40f, 15f, 0f), new Color(0.45f, 0.5f, 0.75f), 0.35f, shadows: true);
        }

        // ------------------------------------------------------------------ Sen Rooftop Garden

        private static void BuildRooftop(Transform root, ArenaDefinition a)
        {
            Ambient(new Color(0.35f, 0.4f, 0.5f), new Color(0.25f, 0.28f, 0.3f), new Color(0.1f, 0.12f, 0.1f), new Color(0.3f, 0.34f, 0.42f), 0.012f);
            var stone = VisualAssets.Lit(new Color(0.42f, 0.42f, 0.4f), 0.7f);
            var moss = VisualAssets.Lit(new Color(0.22f, 0.36f, 0.18f), 0.2f);
            var wood = VisualAssets.Lit(new Color(0.3f, 0.2f, 0.12f), 0.35f);
            for (int x = -7; x <= 7; x++)
            for (int z = 0; z <= 3; z++)
                Box(root, "Slab", new Vector3(x * 1.05f, -0.05f, z * 1.05f - 0.2f), new Vector3(1f, 0.1f, 1f), ((x + z) & 1) == 0 ? stone : moss, false);
            Box(root, "Parapet", new Vector3(0, 0.5f, 3.9f), new Vector3(16f, 1f, 0.3f), stone);
            for (int i = -3; i <= 3; i++) Box(root, "Post", new Vector3(i * 2.4f, 1.6f, 3.8f), new Vector3(0.18f, 3.2f, 0.18f), wood);
            Box(root, "Beam", new Vector3(0, 3.2f, 3.8f), new Vector3(16f, 0.2f, 0.25f), wood);
            for (int i = 0; i < 18; i++)
            {
                float h = 6f + (i * 37 % 11);
                Box(root, "Skyline", new Vector3(-20f + i * 2.4f, h * 0.5f - 2f, 22f), new Vector3(2f, h, 2f), VisualAssets.Lit(new Color(0.12f, 0.14f, 0.18f), 0.3f), false);
            }
            AddLight(root, LightType.Directional, new Vector3(0, 6, -2), new Vector3(35f, -25f, 0f), new Color(0.75f, 0.8f, 0.9f), 0.8f, shadows: true);
            AddLight(root, LightType.Point, new Vector3(0, 2.8f, 2.5f), Vector3.zero, new Color(1f, 0.75f, 0.45f), 1.5f, 8f);
        }

        // ------------------------------------------------------------------ Brandt Forge Yard

        private static void BuildForge(Transform root, ArenaDefinition a)
        {
            Ambient(new Color(0.14f, 0.1f, 0.08f), new Color(0.22f, 0.12f, 0.06f), new Color(0.05f, 0.04f, 0.03f), new Color(0.12f, 0.07f, 0.04f), 0.02f);
            var earth = VisualAssets.Lit(new Color(0.22f, 0.18f, 0.15f), 0.25f);
            var iron = VisualAssets.Lit(new Color(0.18f, 0.18f, 0.2f), 0.6f, 0.8f);
            var brick = VisualAssets.Lit(new Color(0.35f, 0.18f, 0.12f), 0.25f);
            Box(root, "Floor", new Vector3(0, -0.05f, 2f), new Vector3(40f, 0.1f, 14f), earth, false);
            Box(root, "Forge", new Vector3(-3.5f, 1.1f, 4.5f), new Vector3(2.6f, 2.2f, 1.6f), brick);
            var coals = VisualAssets.Unlit(new Color(1f, 0.35f, 0.08f) * 1.6f);
            Box(root, "Coals", new Vector3(-3.5f, 1.25f, 3.69f), new Vector3(1.6f, 0.5f, 0.02f), coals, false);
            Box(root, "Anvil", new Vector3(2.8f, 0.45f, 3.6f), new Vector3(0.9f, 0.3f, 0.35f), iron);
            Box(root, "AnvilBase", new Vector3(2.8f, 0.2f, 3.6f), new Vector3(0.45f, 0.4f, 0.35f), iron);
            for (int i = 0; i < 6; i++)
                Box(root, "RackBlade", new Vector3(5.2f + i * 0.18f, 1.2f, 4.8f), new Vector3(0.03f, 1.6f, 0.08f), VisualAssets.Lit(new Color(0.75f, 0.76f, 0.8f), 0.85f, 0.9f));
            Box(root, "BackWall", new Vector3(0, 3f, 6f), new Vector3(40f, 6f, 0.4f), brick);
            AddLight(root, LightType.Point, new Vector3(-3.5f, 1.4f, 3.2f), Vector3.zero, new Color(1f, 0.45f, 0.15f), 4f, 10f);
            AddLight(root, LightType.Directional, new Vector3(0, 6, -2), new Vector3(50f, 20f, 0f), new Color(0.55f, 0.5f, 0.6f), 0.35f, shadows: true);
        }

        // ------------------------------------------------------------------ The Undercurrent (finals)

        private static void BuildFinals(Transform root, ArenaDefinition a)
        {
            Ambient(new Color(0.05f, 0.05f, 0.07f), new Color(0.06f, 0.06f, 0.07f), new Color(0.02f, 0.02f, 0.02f), new Color(0.02f, 0.02f, 0.03f), 0.035f);
            var concrete = VisualAssets.Lit(new Color(0.2f, 0.2f, 0.21f), 0.5f);
            Box(root, "Basin", new Vector3(0, -0.05f, 2f), new Vector3(40f, 0.1f, 16f), concrete, false);
            var ring = VisualAssets.Lit(new Color(0.55f, 0.08f, 0.1f), 0.4f);
            Box(root, "RingLineFront", new Vector3(0, 0.002f, -0.9f), new Vector3(a.StageHalfWidth * 2f, 0.004f, 0.08f), ring, false);
            Box(root, "RingLineBack", new Vector3(0, 0.002f, 1.6f), new Vector3(a.StageHalfWidth * 2f, 0.004f, 0.08f), ring, false);
            var crowd = VisualAssets.Lit(new Color(0.08f, 0.08f, 0.09f), 0.2f);
            for (int row = 0; row < 3; row++)
            for (int i = -14; i <= 14; i++)
            {
                var person = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                Object.Destroy(person.GetComponent<Collider>());
                person.transform.SetParent(root, false);
                person.transform.localPosition = new Vector3(i * 0.9f + (row & 1) * 0.45f, 0.9f + row * 0.6f, 5.5f + row * 1.2f);
                person.transform.localScale = new Vector3(0.5f, 0.85f, 0.4f);
                person.GetComponent<MeshRenderer>().sharedMaterial = crowd;
                person.GetComponent<MeshRenderer>().shadowCastingMode = ShadowCastingMode.Off;
                person.isStatic = true;
            }
            AddLight(root, LightType.Spot, new Vector3(0, 9f, 0.5f), new Vector3(90f, 0f, 0f), new Color(1f, 0.95f, 0.85f), 30f, 16f, 60f, shadows: true);
            AddLight(root, LightType.Directional, new Vector3(0, 6, -2), new Vector3(30f, 0f, 0f), new Color(0.3f, 0.32f, 0.45f), 0.15f);
        }
    }
}
