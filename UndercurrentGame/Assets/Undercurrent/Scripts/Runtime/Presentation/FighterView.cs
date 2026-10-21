using Undercurrent.Core.Combat;
using UnityEngine;
using AttackStyle = Undercurrent.CharacterGen.AttackStyle;
using Joint = Undercurrent.CharacterGen.Joint;
using NVector3 = System.Numerics.Vector3;
using Pose = Undercurrent.CharacterGen.Pose;
using PoseLibrary = Undercurrent.CharacterGen.PoseLibrary;

namespace Undercurrent.Runtime.Presentation
{
    /// <summary>Drives a fighter's visuals from simulation state. Implementations never feed back into gameplay.</summary>
    public interface IFighterPresenter
    {
        void Tick(FighterRuntime fighter, FightSimulation sim, float deltaTime);
        void OnHit(bool heavy);
        Vector3 HeadPosition { get; }
    }

    /// <summary>
    /// World-space representation of one fighter. Picks a licensed character prefab when one exists at
    /// Resources/Undercurrent/Characters/&lt;fighterId&gt; (Humanoid rig + Animator), otherwise the placeholder mannequin.
    /// </summary>
    public sealed class FighterView : MonoBehaviour
    {
        private FighterRuntime _fighter;
        private IFighterPresenter _presenter;
        private Transform _visualRoot;
        private Renderer[] _renderers;
        private MaterialPropertyBlock _block;
        private float _flash;
        private Transform _shadowBlob;

        public FighterRuntime Fighter => _fighter;
        public Vector3 HeadPosition => _presenter != null ? _presenter.HeadPosition : transform.position + Vector3.up * 1.6f;
        public bool UsesLicensedModel { get; private set; }

        public void Setup(FighterRuntime fighter, CharacterLook look)
        {
            _fighter = fighter;
            _visualRoot = new GameObject("Visual").transform;
            _visualRoot.SetParent(transform, false);

            var prefab = Resources.Load<GameObject>("Undercurrent/Characters/" + fighter.Def.Id);
            Animator animator = null;
            if (prefab != null)
            {
                var instance = Instantiate(prefab, _visualRoot);
                animator = instance.GetComponentInChildren<Animator>();
                if (animator != null && animator.avatar != null && animator.avatar.isHuman)
                {
                    _presenter = new AnimatorFighterPresenter(animator, instance.transform);
                    UsesLicensedModel = true;
                }
                else
                {
                    Debug.LogWarning($"[Undercurrent] Character prefab for '{fighter.Def.Id}' has no humanoid Animator; using placeholder mannequin.");
                    Destroy(instance);
                }
            }
            if (_presenter == null)
            {
                var mannequin = Mannequin.Build(_visualRoot, look);
                _presenter = new MannequinPresenter(mannequin, look);
            }

            _renderers = _visualRoot.GetComponentsInChildren<Renderer>();
            _block = new MaterialPropertyBlock();

            // Soft contact shadow under the feet: readable even on tiers with real-time shadows disabled.
            var blob = GameObject.CreatePrimitive(PrimitiveType.Quad);
            Destroy(blob.GetComponent<Collider>());
            blob.name = "ContactShadow";
            _shadowBlob = blob.transform;
            _shadowBlob.SetParent(transform, false);
            _shadowBlob.localRotation = Quaternion.Euler(90f, 0f, 0f);
            _shadowBlob.localScale = new Vector3(0.9f, 0.55f, 1f);
            var mat = VisualAssets.Unlit(new Color(0f, 0f, 0f, 0.35f));
            ContactShadowTexture.Apply(mat);
            blob.GetComponent<MeshRenderer>().sharedMaterial = mat;
            blob.GetComponent<MeshRenderer>().shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        }

        public void Flash()
        {
            _flash = 1f;
            _presenter?.OnHit(true);
        }

        public void React(bool heavy)
        {
            _flash = heavy ? 1f : 0.6f;
            _presenter?.OnHit(heavy);
        }

        public void Tick(FightSimulation sim, float dt)
        {
            if (_fighter == null) return;
            transform.position = new Vector3(_fighter.X, _fighter.Y, 0f);
            // Mirror instead of turning around so both fighters keep their chest towards the camera (2.5D readability).
            _visualRoot.localRotation = Quaternion.Euler(0f, 90f * _fighter.Facing, 0f);
            _visualRoot.localScale = new Vector3(_fighter.Facing, 1f, 1f);
            _presenter.Tick(_fighter, sim, dt);

            if (_shadowBlob != null)
            {
                _shadowBlob.position = new Vector3(_fighter.X, 0.012f, 0f);
                float k = Mathf.Clamp01(1f - _fighter.Y * 0.5f);
                _shadowBlob.localScale = new Vector3(0.9f * k, 0.55f * k, 1f);
            }

            if (_flash > 0f)
            {
                _flash = Mathf.Max(0f, _flash - dt * 9f);
                var tint = Color.Lerp(Color.white, new Color(1f, 0.55f, 0.45f), _flash);
                foreach (var r in _renderers)
                {
                    if (r == null) continue;
                    r.GetPropertyBlock(_block);
                    _block.SetColor("_EmissionColor", tint * _flash * 0.6f);
                    r.SetPropertyBlock(_block);
                }
                if (_flash <= 0f)
                    foreach (var r in _renderers) if (r != null) r.SetPropertyBlock(null);
            }
        }
    }

    /// <summary>Generates a radial falloff texture for the contact shadow quad.</summary>
    internal static class ContactShadowTexture
    {
        private static Texture2D _tex;

        public static void Apply(Material m)
        {
            if (_tex == null)
            {
                const int size = 64;
                _tex = new Texture2D(size, size, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp, name = "UC_ContactShadow" };
                var px = new Color32[size * size];
                for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float dx = (x + 0.5f) / size * 2f - 1f;
                    float dy = (y + 0.5f) / size * 2f - 1f;
                    float a = Mathf.Clamp01(1f - Mathf.Sqrt(dx * dx + dy * dy));
                    px[y * size + x] = new Color32(0, 0, 0, (byte)(a * a * 255));
                }
                _tex.SetPixels32(px);
                _tex.Apply();
            }
            if (m.HasProperty("_BaseMap")) m.SetTexture("_BaseMap", _tex);
            if (m.HasProperty("_MainTex")) m.SetTexture("_MainTex", _tex);
            // URP Unlit: switch to transparent alpha blending.
            if (m.HasProperty("_Surface"))
            {
                m.SetFloat("_Surface", 1f);
                m.SetFloat("_Blend", 0f);
                m.SetOverrideTag("RenderType", "Transparent");
                m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                m.SetInt("_ZWrite", 0);
                m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            }
        }
    }

    /// <summary>Animates the placeholder mannequin procedurally from the fighter's state and move frame data.</summary>
    public sealed class MannequinPresenter : IFighterPresenter
    {
        private readonly Mannequin _m;
        private readonly CharacterLook _look;
        private readonly Pose _target = new Pose();
        private readonly Quaternion[] _baseRotations = new Quaternion[Mannequin.JointCount];
        private Vector3 _hipBase;
        private Vector3 _hipCurrent;
        private float _walkPhase;
        private float _lastX;
        private float _time;
        private float _hitIntensity;
        private float _hitTimer;

        public MannequinPresenter(Mannequin m, CharacterLook look)
        {
            _m = m;
            _look = look;
            for (int i = 0; i < Mannequin.JointCount; i++) _baseRotations[i] = m.Joints[i].localRotation;
            _hipBase = m[Joint.Hips].localPosition;
            _hipCurrent = _hipBase;
        }

        public Vector3 HeadPosition => _m[Joint.Head].position + Vector3.up * 0.1f;

        public void OnHit(bool heavy)
        {
            _hitIntensity = heavy ? 1f : 0.6f;
            _hitTimer = 0f;
        }

        public void Tick(FighterRuntime f, FightSimulation sim, float dt)
        {
            _time += dt;
            _hitTimer += dt;
            float rate = BuildTargetPose(f, sim, dt);
            ApplyGrounding(f);

            float k = 1f - Mathf.Exp(-rate * dt);
            for (int i = 0; i < Mannequin.JointCount; i++)
            {
                var e = _target.Euler[i];
                var target = _baseRotations[i] * Quaternion.Euler(e.X, e.Y, e.Z);
                var j = _m.Joints[i];
                j.localRotation = Quaternion.Slerp(j.localRotation, target, k);
            }
            var hipTarget = _hipBase + new Vector3(_target.HipOffset.X, _target.HipOffset.Y, _target.HipOffset.Z);
            _hipCurrent = Vector3.Lerp(_hipCurrent, hipTarget, k);
            _m[Joint.Hips].localPosition = _hipCurrent;
        }

        private static float Ease(float t) => t * t * (3f - 2f * t);

        /// <summary>Foot-planting IK for grounded states (feet neither float nor sink, lifted kick legs stay free).</summary>
        private void ApplyGrounding(FighterRuntime f)
        {
            switch (f.State)
            {
                case FighterStateId.Airborne:
                case FighterStateId.KnockedDown:
                case FighterStateId.KO:
                    return;
                case FighterStateId.GetUp:
                    if (f.StateFrame < f.Def.GetUpFrames / 2) return;
                    break;
                case FighterStateId.Attack:
                    if (f.CurrentMove.IsAirMove) return;
                    break;
                case FighterStateId.WalkForward:
                case FighterStateId.WalkBack:
                    _m.Grounding.Apply(_target, plantSecondFoot: false);
                    return;
            }
            _m.Grounding.Apply(_target);
        }

        /// <summary>Fills <see cref="_target"/> and returns the blend rate (higher = snappier).</summary>
        private float BuildTargetPose(FighterRuntime f, FightSimulation sim, float dt)
        {
            float moved = f.X - _lastX;
            _lastX = f.X;
            Pose.Lerp(PoseLibrary.Guard, PoseLibrary.Guard, 0f, _target);

            switch (f.State)
            {
                case FighterStateId.Intro:
                    // Fighters are not stepped during the intro, so time the bow from the round phase clock.
                    Pose.Lerp(PoseLibrary.Bow, PoseLibrary.Guard, Mathf.Clamp01((sim.PhaseTicks - 30) / 40f), _target);
                    return 8f;

                case FighterStateId.Idle:
                {
                    Pose.Lerp(PoseLibrary.Guard, PoseLibrary.Guard, 0f, _target);
                    float breath = Mathf.Sin(_time * 2.2f);
                    _target.Add(Joint.Chest, breath * 1.5f);
                    _target.HipOffset += new NVector3(0, Mathf.Abs(Mathf.Sin(_time * 4.2f)) * _look.Bounce - _look.Bounce * 0.5f, 0);
                    return 14f;
                }

                case FighterStateId.WalkForward:
                case FighterStateId.WalkBack:
                {
                    // Phase advances with real displacement so feet do not slide relative to the ground.
                    float stride = 0.62f * _m.Skeleton.Height / 1.78f;
                    _walkPhase += Mathf.Abs(moved) / stride * Mathf.PI * 2f * (f.State == FighterStateId.WalkBack ? -1f : 1f);
                    float sL = Mathf.Sin(_walkPhase), sR = Mathf.Sin(_walkPhase + Mathf.PI);
                    _target.Add(Joint.HipL, -sL * 16f).Add(Joint.KneeL, Mathf.Max(0f, sL) * 26f);
                    _target.Add(Joint.HipR, -sR * 16f).Add(Joint.KneeR, Mathf.Max(0f, sR) * 26f);
                    _target.HipOffset += new NVector3(0, Mathf.Abs(Mathf.Cos(_walkPhase)) * 0.025f - 0.02f, 0);
                    return 16f;
                }

                case FighterStateId.Crouch:
                    Pose.Lerp(PoseLibrary.Crouch, PoseLibrary.Crouch, 0f, _target);
                    return 18f;

                case FighterStateId.Blocking:
                    Pose.Lerp(PoseLibrary.Block, PoseLibrary.Block, 0f, _target);
                    return 30f;

                case FighterStateId.CrouchBlocking:
                    Pose.Lerp(PoseLibrary.CrouchBlock, PoseLibrary.CrouchBlock, 0f, _target);
                    return 30f;

                case FighterStateId.BlockStun:
                {
                    var b = f.IsCrouching ? PoseLibrary.CrouchBlock : PoseLibrary.Block;
                    Pose.Lerp(b, b, 0f, _target);
                    _target.HipOffset += new NVector3(0, 0, -0.05f);
                    _target.Add(Joint.Spine, -6f);
                    return 40f;
                }

                case FighterStateId.JumpSquat:
                case FighterStateId.Landing:
                    Pose.Lerp(PoseLibrary.Guard, PoseLibrary.Crouch, 0.45f, _target);
                    return 30f;

                case FighterStateId.Airborne:
                    Pose.Lerp(PoseLibrary.Airborne, PoseLibrary.Airborne, 0f, _target);
                    return 14f;

                case FighterStateId.Attack:
                    return BuildAttack(f);

                case FighterStateId.ThrowAttempt:
                {
                    float t = Mathf.Clamp01(f.StateFrame / (float)(FighterRuntime.ThrowStartup + 1));
                    Pose.Lerp(PoseLibrary.Guard, PoseLibrary.ThrowGrab, Ease(t), _target);
                    return 40f;
                }

                case FighterStateId.Throwing:
                {
                    float t = Mathf.Clamp01((f.StateFrame - 4) / 14f);
                    Pose.Lerp(PoseLibrary.ThrowGrab, PoseLibrary.ThrowHeave, Ease(t), _target);
                    return 20f;
                }

                case FighterStateId.Grabbed:
                    Pose.Lerp(PoseLibrary.Grabbed, PoseLibrary.Grabbed, 0f, _target);
                    return 25f;

                case FighterStateId.Dodge:
                    Pose.Lerp(PoseLibrary.Dodge, PoseLibrary.Dodge, 0f, _target);
                    return 35f;

                case FighterStateId.Parry:
                    Pose.Lerp(PoseLibrary.ParryStrike, PoseLibrary.Guard, Mathf.Clamp01((f.StateFrame - f.Def.ParryWindow) / (float)f.Def.ParryRecovery), _target);
                    return 45f;

                case FighterStateId.HitStun:
                {
                    // Snap back on impact, then recover towards guard as stun runs out.
                    float recover = Mathf.Clamp01(_hitTimer * 5f);
                    var hitPose = f.IsCrouching ? PoseLibrary.Crouch : PoseLibrary.HitBack;
                    Pose.Lerp(hitPose, PoseLibrary.Guard, recover * (1f - _hitIntensity * 0.4f), _target);
                    if (_hitTimer < 0.05f) _target.Add(Joint.Neck, -12f * _hitIntensity).Add(Joint.Chest, -6f * _hitIntensity);
                    return 55f;
                }

                case FighterStateId.Parried:
                    Pose.Lerp(PoseLibrary.Staggered, PoseLibrary.Staggered, 0f, _target);
                    _target.Add(Joint.Spine, Mathf.Sin(_time * 12f) * 3f);
                    return 30f;

                case FighterStateId.GuardBroken:
                    Pose.Lerp(PoseLibrary.GuardBroken, PoseLibrary.GuardBroken, 0f, _target);
                    _target.Add(Joint.Spine, 0, 0, Mathf.Sin(_time * 5f) * 6f);
                    return 20f;

                case FighterStateId.KnockedDown:
                case FighterStateId.KO:
                    Pose.Lerp(PoseLibrary.Lying, PoseLibrary.Lying, 0f, _target);
                    return f.StateFrame < 20 ? 12f : 6f;

                case FighterStateId.GetUp:
                {
                    float t = Mathf.Clamp01(f.StateFrame / (float)Mathf.Max(1, f.Def.GetUpFrames));
                    if (t < 0.5f) Pose.Lerp(PoseLibrary.Lying, PoseLibrary.Crouch, Ease(t * 2f), _target);
                    else Pose.Lerp(PoseLibrary.Crouch, PoseLibrary.Guard, Ease((t - 0.5f) * 2f), _target);
                    return 22f;
                }

                case FighterStateId.Victory:
                    Pose.Lerp(PoseLibrary.Victory, PoseLibrary.Victory, 0f, _target);
                    _target.Add(Joint.ShoulderR, Mathf.Sin(_time * 3f) * 4f);
                    return 6f;
            }
            return 14f;
        }

        private float BuildAttack(FighterRuntime f)
        {
            var m = f.CurrentMove;
            var style = PoseLibrary.StyleFor(m);
            var basePose = PoseLibrary.BaseFor(style);
            var (windup, strike) = PoseLibrary.For(style);
            int frame = f.MoveFrame;

            if (frame <= m.Startup)
            {
                float t = frame / (float)(m.Startup + 1);
                if (t < 0.5f) Pose.Lerp(basePose, windup, Ease(t * 2f), _target);
                else Pose.Lerp(windup, strike, Ease((t - 0.5f) * 2f), _target);
            }
            else if (frame <= m.LastActiveFrame)
            {
                Pose.Lerp(strike, strike, 0f, _target);
            }
            else
            {
                float t = Mathf.Clamp01((frame - m.LastActiveFrame) / (float)Mathf.Max(1, m.Recovery));
                Pose.Lerp(strike, basePose, Ease(t), _target);
            }

            if (style == AttackStyle.Spin && frame > m.LastActiveFrame)
            {
                // Finish the rotation instead of unwinding backwards.
                float t = Mathf.Clamp01((frame - m.LastActiveFrame) / (float)Mathf.Max(1, m.Recovery));
                var hips = _target.Euler[(int)Joint.Hips];
                _target.Euler[(int)Joint.Hips] = new NVector3(hips.X, Mathf.Lerp(185f, 382f, Ease(t)), hips.Z);
            }
            // Attacks track the authored timeline tightly; the strike pose must be on screen on the first active frame.
            return 60f;
        }
    }

    /// <summary>
    /// Presenter for licensed humanoid characters. Animator states are named after fighter states and move
    /// AnimationIds (see Editor/AnimatorControllerBuilder). Attack clips are scrubbed to the simulation frame so
    /// animation and hit timing cannot drift. Not verified with real assets in this environment.
    /// </summary>
    public sealed class AnimatorFighterPresenter : IFighterPresenter
    {
        private readonly Animator _animator;
        private readonly Transform _root;
        private readonly Transform _head;
        private string _currentState;
        private float _hitWeight;
        private readonly bool _hasHitWeight;

        public AnimatorFighterPresenter(Animator animator, Transform root)
        {
            _animator = animator;
            _root = root;
            _animator.applyRootMotion = false; // the simulation owns position
            _head = animator.GetBoneTransform(HumanBodyBones.Head);
            foreach (var p in animator.parameters) if (p.name == "HitWeight") _hasHitWeight = true;
        }

        public Vector3 HeadPosition => _head != null ? _head.position : _root.position + Vector3.up * 1.6f;

        public void OnHit(bool heavy)
        {
            _hitWeight = heavy ? 1f : 0.6f;
        }

        public static string StateName(FighterRuntime f)
        {
            switch (f.State)
            {
                case FighterStateId.Attack: return "Attack_" + f.CurrentMove.AnimationId;
                case FighterStateId.ThrowAttempt: return "ThrowAttempt";
                default: return f.State.ToString();
            }
        }

        public void Tick(FighterRuntime f, FightSimulation sim, float dt)
        {
            var state = StateName(f);
            int hash = Animator.StringToHash(state);
            if (!_animator.HasState(0, hash))
            {
                state = f.State == FighterStateId.Attack ? "Attack_Generic" : "Idle";
                hash = Animator.StringToHash(state);
            }

            if (f.State == FighterStateId.Attack)
            {
                // Scrub: normalized time follows the move's frame counter exactly.
                float normalized = Mathf.Clamp01((f.MoveFrame - 1) / (float)Mathf.Max(1, f.CurrentMove.TotalFrames - 1));
                _animator.speed = 0f;
                _animator.Play(hash, 0, normalized);
            }
            else
            {
                _animator.speed = 1f;
                if (state != _currentState) _animator.CrossFadeInFixedTime(hash, 0.08f, 0);
            }
            _currentState = state;
            _hitWeight = Mathf.Max(0f, _hitWeight - dt * 4f);
            if (_hasHitWeight) _animator.SetFloat("HitWeight", _hitWeight);
        }
    }
}
