using UnityEngine;

namespace Undercurrent.Runtime.Presentation
{
    public enum Joint
    {
        Hips, Spine, Chest, Neck, Head,
        ShoulderL, ElbowL, HandL,
        ShoulderR, ElbowR, HandR,
        HipL, KneeL, FootL,
        HipR, KneeR, FootR,
    }

    /// <summary>
    /// Placeholder articulated humanoid assembled from primitive meshes on a real joint hierarchy.
    /// It exists so the game is playable and readable before licensed character models are integrated; it is
    /// NOT final-quality art. Local space: facing +Z, up +Y, character's left is -X.
    /// </summary>
    public sealed class Mannequin
    {
        public const int JointCount = 17;

        public readonly Transform Root;
        public readonly Transform[] Joints = new Transform[JointCount];
        public readonly Renderer[] Renderers;
        public readonly float HipHeight;
        public readonly float Scale;

        private Mannequin(Transform root, float hipHeight, float scale)
        {
            Root = root;
            HipHeight = hipHeight;
            Scale = scale;
            Renderers = null;
        }

        private Mannequin(Mannequin m, Renderer[] renderers)
        {
            Root = m.Root;
            HipHeight = m.HipHeight;
            Scale = m.Scale;
            System.Array.Copy(m.Joints, Joints, JointCount);
            Renderers = renderers;
        }

        public Transform this[Joint j] => Joints[(int)j];

        public static Mannequin Build(Transform parent, CharacterLook look)
        {
            float s = look.Height / 1.78f;
            float bulk = Mathf.Lerp(0.85f, 1.3f, look.Build);
            var skin = VisualAssets.Lit(look.Skin, 0.42f);
            var hair = VisualAssets.Lit(look.Hair, 0.25f);
            var outfit = VisualAssets.Lit(look.Outfit, 0.2f);
            var trim = VisualAssets.Lit(look.OutfitTrim, 0.3f);
            var wraps = VisualAssets.Lit(look.Wraps, 0.15f);
            var dark = VisualAssets.Lit(new Color(0.06f, 0.05f, 0.05f), 0.6f);

            var root = new GameObject("Mannequin").transform;
            root.SetParent(parent, false);

            float thigh = 0.45f * s, shin = 0.45f * s, upperArm = 0.29f * s, forearm = 0.27f * s;
            float hipHeight = thigh + shin + 0.06f * s;
            float pelvisHalf = (look.FemaleProportions ? 0.105f : 0.095f) * s;
            float shoulderHalf = (look.FemaleProportions ? 0.175f : 0.2f) * s * Mathf.Lerp(0.95f, 1.12f, look.Build);

            var m = new Mannequin(root, hipHeight, s);
            Transform J(Joint j, Transform p, Vector3 pos)
            {
                var t = new GameObject(j.ToString()).transform;
                t.SetParent(p, false);
                t.localPosition = pos;
                m.Joints[(int)j] = t;
                return t;
            }

            var hips = J(Joint.Hips, root, new Vector3(0, hipHeight, 0));
            var spine = J(Joint.Spine, hips, new Vector3(0, 0.1f * s, 0));
            var chest = J(Joint.Chest, spine, new Vector3(0, 0.2f * s, 0));
            var neck = J(Joint.Neck, chest, new Vector3(0, 0.22f * s, 0));
            var head = J(Joint.Head, neck, new Vector3(0, 0.09f * s, 0));
            var shL = J(Joint.ShoulderL, chest, new Vector3(-shoulderHalf, 0.18f * s, 0));
            var elL = J(Joint.ElbowL, shL, new Vector3(0, -upperArm, 0));
            var hdL = J(Joint.HandL, elL, new Vector3(0, -forearm, 0));
            var shR = J(Joint.ShoulderR, chest, new Vector3(shoulderHalf, 0.18f * s, 0));
            var elR = J(Joint.ElbowR, shR, new Vector3(0, -upperArm, 0));
            var hdR = J(Joint.HandR, elR, new Vector3(0, -forearm, 0));
            var hpL = J(Joint.HipL, hips, new Vector3(-pelvisHalf, -0.04f * s, 0));
            var knL = J(Joint.KneeL, hpL, new Vector3(0, -thigh, 0));
            var ftL = J(Joint.FootL, knL, new Vector3(0, -shin, 0));
            var hpR = J(Joint.HipR, hips, new Vector3(pelvisHalf, -0.04f * s, 0));
            var knR = J(Joint.KneeR, hpR, new Vector3(0, -thigh, 0));
            var ftR = J(Joint.FootR, knR, new Vector3(0, -shin, 0));

            // ---- Meshes (capsules hang along -Y from their joint)
            float limbR = 0.055f * s * bulk;
            Limb(hpL, thigh, limbR * 1.35f, look.LongPants ? outfit : skin);
            Limb(hpR, thigh, limbR * 1.35f, look.LongPants ? outfit : skin);
            Limb(knL, shin, limbR * 1.05f, look.LongPants ? outfit : skin);
            Limb(knR, shin, limbR * 1.05f, look.LongPants ? outfit : skin);
            if (!look.LongPants)
            {
                // Fight shorts over the upper thighs.
                Limb(hpL, thigh * 0.55f, limbR * 1.55f, outfit);
                Limb(hpR, thigh * 0.55f, limbR * 1.55f, outfit);
            }
            Part(PrimitiveType.Cube, ftL, new Vector3(0, -0.02f * s, 0.06f * s), new Vector3(0.09f, 0.06f, 0.24f) * s, skin);
            Part(PrimitiveType.Cube, ftR, new Vector3(0, -0.02f * s, 0.06f * s), new Vector3(0.09f, 0.06f, 0.24f) * s, skin);

            Limb(shL, upperArm, limbR * 1.1f, skin);
            Limb(shR, upperArm, limbR * 1.1f, skin);
            Limb(elL, forearm, limbR * 0.95f, skin);
            Limb(elR, forearm, limbR * 0.95f, skin);
            Part(PrimitiveType.Sphere, hdL, new Vector3(0, -0.04f * s, 0), new Vector3(0.1f, 0.11f, 0.1f) * s * bulk, wraps);
            Part(PrimitiveType.Sphere, hdR, new Vector3(0, -0.04f * s, 0), new Vector3(0.1f, 0.11f, 0.1f) * s * bulk, wraps);
            Part(PrimitiveType.Capsule, elL, new Vector3(0, -forearm * 0.75f, 0), new Vector3(limbR * 2.15f, forearm * 0.2f, limbR * 2.15f), wraps);
            Part(PrimitiveType.Capsule, elR, new Vector3(0, -forearm * 0.75f, 0), new Vector3(limbR * 2.15f, forearm * 0.2f, limbR * 2.15f), wraps);

            // Pelvis & torso
            Part(PrimitiveType.Sphere, hips, new Vector3(0, -0.02f * s, 0), new Vector3(pelvisHalf * 2.9f, 0.2f * s, 0.2f * s * bulk), outfit);
            Part(PrimitiveType.Capsule, spine, new Vector3(0, 0.08f * s, 0), new Vector3(0.27f * s * bulk, 0.14f * s, 0.18f * s * bulk), look.Shirtless ? skin : outfit);
            Part(PrimitiveType.Cylinder, hips, new Vector3(0, 0.05f * s, 0), new Vector3(pelvisHalf * 2.7f, 0.025f * s, 0.2f * s * bulk), trim);
            var torso = look.Shirtless ? skin : outfit;
            Part(PrimitiveType.Capsule, chest, new Vector3(0, 0.1f * s, 0), new Vector3(shoulderHalf * 1.9f, 0.17f * s, 0.21f * s * bulk), torso);
            if (look.FemaleProportions && !look.Shirtless)
                Part(PrimitiveType.Sphere, chest, new Vector3(0, 0.1f * s, 0.07f * s), new Vector3(0.24f, 0.11f, 0.1f) * s, torso);
            Part(PrimitiveType.Sphere, shL, Vector3.zero, Vector3.one * limbR * 2.6f, look.Shirtless ? skin : torso);
            Part(PrimitiveType.Sphere, shR, Vector3.zero, Vector3.one * limbR * 2.6f, look.Shirtless ? skin : torso);
            Limb(neck, 0.1f * s, 0.05f * s * bulk, skin, up: true);

            // Head: skull, jaw, eyes (for facing readability), hair.
            Part(PrimitiveType.Sphere, head, new Vector3(0, 0.1f * s, 0.005f * s), new Vector3(0.19f, 0.23f, 0.21f) * s, skin);
            Part(PrimitiveType.Sphere, head, new Vector3(0, 0.035f * s, 0.035f * s), new Vector3(0.15f, 0.1f, 0.15f) * s, skin);
            Part(PrimitiveType.Sphere, head, new Vector3(-0.037f * s, 0.12f * s, 0.09f * s), Vector3.one * 0.022f * s, dark);
            Part(PrimitiveType.Sphere, head, new Vector3(0.037f * s, 0.12f * s, 0.09f * s), Vector3.one * 0.022f * s, dark);
            Part(PrimitiveType.Cube, head, new Vector3(0, 0.095f * s, 0.105f * s), new Vector3(0.022f, 0.045f, 0.03f) * s, skin);
            switch (look.HairStyle)
            {
                case HairStyle.Short:
                    Part(PrimitiveType.Sphere, head, new Vector3(0, 0.15f * s, -0.01f * s), new Vector3(0.2f, 0.17f, 0.21f) * s, hair);
                    break;
                case HairStyle.Ponytail:
                    Part(PrimitiveType.Sphere, head, new Vector3(0, 0.15f * s, -0.01f * s), new Vector3(0.205f, 0.18f, 0.215f) * s, hair);
                    Part(PrimitiveType.Capsule, head, new Vector3(0, 0.07f * s, -0.13f * s), new Vector3(0.06f, 0.1f, 0.06f) * s, hair);
                    break;
                case HairStyle.Bun:
                    Part(PrimitiveType.Sphere, head, new Vector3(0, 0.15f * s, -0.01f * s), new Vector3(0.2f, 0.17f, 0.21f) * s, hair);
                    Part(PrimitiveType.Sphere, head, new Vector3(0, 0.22f * s, -0.07f * s), Vector3.one * 0.08f * s, hair);
                    break;
                case HairStyle.Braids:
                    Part(PrimitiveType.Sphere, head, new Vector3(0, 0.15f * s, -0.01f * s), new Vector3(0.205f, 0.18f, 0.215f) * s, hair);
                    for (int i = -1; i <= 1; i++)
                        Part(PrimitiveType.Capsule, head, new Vector3(i * 0.05f * s, 0.02f * s, -0.11f * s), new Vector3(0.035f, 0.12f, 0.035f) * s, hair);
                    break;
                case HairStyle.Shaved:
                    Part(PrimitiveType.Sphere, head, new Vector3(0, 0.14f * s, -0.01f * s), new Vector3(0.195f, 0.16f, 0.205f) * s, VisualAssets.Lit(Color.Lerp(look.Skin, look.Hair, 0.35f), 0.3f));
                    break;
            }
            if (look.Beard)
                Part(PrimitiveType.Sphere, head, new Vector3(0, 0.03f * s, 0.045f * s), new Vector3(0.15f, 0.08f, 0.13f) * s, hair);

            return new Mannequin(m, root.GetComponentsInChildren<Renderer>());
        }

        private static void Limb(Transform joint, float length, float radius, Material mat, bool up = false)
        {
            float sign = up ? 1f : -1f;
            Part(PrimitiveType.Capsule, joint, new Vector3(0, sign * length * 0.5f, 0), new Vector3(radius * 2f, length * 0.5f + radius * 0.5f, radius * 2f), mat);
        }

        private static void Part(PrimitiveType type, Transform parent, Vector3 localPos, Vector3 scale, Material mat)
        {
            var go = GameObject.CreatePrimitive(type);
            var col = go.GetComponent<Collider>();
            if (col != null) Object.Destroy(col);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = scale;
            var r = go.GetComponent<MeshRenderer>();
            r.sharedMaterial = mat;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
            r.receiveShadows = true;
        }
    }
}
