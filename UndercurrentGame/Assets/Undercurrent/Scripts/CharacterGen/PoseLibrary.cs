using System.Collections.Generic;
using Undercurrent.Core.Combat;
using System;
using System.Numerics;

namespace Undercurrent.CharacterGen
{
    /// <summary>Joint rotations (Euler degrees, local) plus a hip offset in character space.</summary>
    public sealed class Pose
    {
        public readonly Vector3[] Euler = new Vector3[HumanSkeleton.JointCount];
        public Vector3 HipOffset;

        public Pose Copy()
        {
            var p = new Pose { HipOffset = HipOffset };
            System.Array.Copy(Euler, p.Euler, Euler.Length);
            return p;
        }

        public Pose Set(Joint j, float x, float y = 0f, float z = 0f)
        {
            Euler[(int)j] = new Vector3(x, y, z);
            return this;
        }

        public Pose Add(Joint j, float x, float y = 0f, float z = 0f)
        {
            Euler[(int)j] += new Vector3(x, y, z);
            return this;
        }

        public Pose Hip(float x, float y, float z)
        {
            HipOffset = new Vector3(x, y, z);
            return this;
        }

        /// <summary>Shortest-path angle interpolation in degrees.</summary>
        public static float LerpAngle(float a, float b, float t)
        {
            float delta = (b - a) % 360f;
            if (delta > 180f) delta -= 360f;
            if (delta < -180f) delta += 360f;
            return a + delta * Math.Max(0f, Math.Min(1f, t));
        }

        /// <summary>Angle-wise interpolation into <paramref name="into"/> (used to author timelines, not for final blending).</summary>
        public static Pose Lerp(Pose a, Pose b, float t, Pose into)
        {
            for (int i = 0; i < a.Euler.Length; i++)
            {
                var ea = a.Euler[i];
                var eb = b.Euler[i];
                into.Euler[i] = new Vector3(LerpAngle(ea.X, eb.X, t), LerpAngle(ea.Y, eb.Y, t), LerpAngle(ea.Z, eb.Z, t));
            }
            into.HipOffset = Vector3.Lerp(a.HipOffset, b.HipOffset, t);
            return into;
        }
    }

    public enum AttackStyle
    {
        Jab, Cross, LeadHook, BodyHook, Uppercut, Elbow, Overhand, Lunge, LeapPunch, Charge,
        FrontKick, Roundhouse, SideKick, AxeKick, Crescent, Spin, Knee, LowKick, Sweep, CrouchJab,
        AirKick, AirPunch, Throw, Special,
        // Weapon swings (weapon held in the right hand)
        WeaponSlash, TwoHandThrust, LowSlash,
    }

    /// <summary>
    /// Authored key poses for the placeholder mannequin. Each attack has a wind-up and a strike pose; the strike
    /// pose is reached exactly on the move's first active frame so visuals and hit detection stay synchronised.
    /// </summary>
    public static class PoseLibrary
    {
        public static readonly Pose Guard = new Pose()
            .Set(Joint.Hips, 0, 22).Set(Joint.Spine, 8, 4).Set(Joint.Chest, 4, 6).Set(Joint.Neck, -6, -16).Set(Joint.Head, -4, -10)
            .Set(Joint.ShoulderL, -52, 0, 12).Set(Joint.ElbowL, -112).Set(Joint.HandL, -10)
            .Set(Joint.ShoulderR, -38, 0, -14).Set(Joint.ElbowR, -128).Set(Joint.HandR, -10)
            .Set(Joint.HipL, -24, 0, -4).Set(Joint.KneeL, 30).Set(Joint.FootL, -6)
            .Set(Joint.HipR, 14, 0, 4).Set(Joint.KneeR, 24).Set(Joint.FootR, -28)
            .Hip(0, -0.06f, 0);

        public static readonly Pose Crouch = Guard.Copy()
            .Set(Joint.Spine, 24, 4).Set(Joint.Chest, 10, 6)
            .Set(Joint.HipL, -68, 0, -8).Set(Joint.KneeL, 98).Set(Joint.FootL, -30)
            .Set(Joint.HipR, -42, 0, 8).Set(Joint.KneeR, 92).Set(Joint.FootR, -46)
            .Hip(0, -0.3f, -0.05f);

        public static readonly Pose Block = Guard.Copy()
            .Set(Joint.Chest, 14, 4).Set(Joint.Neck, 14, -12)
            .Set(Joint.ShoulderL, -68, 0, 24).Set(Joint.ElbowL, -142)
            .Set(Joint.ShoulderR, -66, 0, -26).Set(Joint.ElbowR, -142);

        public static readonly Pose CrouchBlock = Crouch.Copy()
            .Set(Joint.Chest, 18, 4).Set(Joint.Neck, 12, -12)
            .Set(Joint.ShoulderL, -60, 0, 24).Set(Joint.ElbowL, -140)
            .Set(Joint.ShoulderR, -58, 0, -26).Set(Joint.ElbowR, -140);

        public static readonly Pose Airborne = Guard.Copy()
            .Set(Joint.HipL, -70, 0, -6).Set(Joint.KneeL, 110).Set(Joint.HipR, -45, 0, 6).Set(Joint.KneeR, 100)
            .Hip(0, 0.05f, 0);

        public static readonly Pose Lying = new Pose()
            .Set(Joint.Hips, -88, 10, 0).Set(Joint.Spine, -4).Set(Joint.Chest, -4).Set(Joint.Neck, 10, 20).Set(Joint.Head, 0, 10)
            .Set(Joint.ShoulderL, -20, 0, 70).Set(Joint.ElbowL, -30).Set(Joint.ShoulderR, -30, 0, -65).Set(Joint.ElbowR, -45)
            .Set(Joint.HipL, -10, 0, -10).Set(Joint.KneeL, 30).Set(Joint.HipR, -25, 0, 8).Set(Joint.KneeR, 50)
            .Hip(0, -0.82f, -0.35f);

        public static readonly Pose HitBack = Guard.Copy()
            .Set(Joint.Spine, -16, 4).Set(Joint.Chest, -14, -6).Set(Joint.Neck, -22, -18).Set(Joint.Head, -14, -10)
            .Set(Joint.ShoulderL, -30, 0, 30).Set(Joint.ElbowL, -80).Set(Joint.ShoulderR, -20, 0, -30).Set(Joint.ElbowR, -90)
            .Hip(0, -0.08f, -0.08f);

        public static readonly Pose Staggered = Guard.Copy()
            .Set(Joint.Spine, -22, -8).Set(Joint.Chest, -14, -10).Set(Joint.Neck, -20)
            .Set(Joint.ShoulderL, -15, 0, 65).Set(Joint.ElbowL, -40).Set(Joint.ShoulderR, -15, 0, -65).Set(Joint.ElbowR, -40)
            .Hip(0, -0.05f, -0.12f);

        public static readonly Pose GuardBroken = Guard.Copy()
            .Set(Joint.Spine, 30).Set(Joint.Chest, 14).Set(Joint.Neck, 20)
            .Set(Joint.ShoulderL, 5, 0, 8).Set(Joint.ElbowL, -20).Set(Joint.ShoulderR, 5, 0, -8).Set(Joint.ElbowR, -20)
            .Hip(0, -0.16f, -0.05f);

        public static readonly Pose Dodge = Guard.Copy()
            .Set(Joint.Spine, -18, 10, 8).Set(Joint.Chest, -10, 8).Set(Joint.HipL, -10).Set(Joint.KneeL, 40).Set(Joint.HipR, 30).Set(Joint.KneeR, 45)
            .Hip(0, -0.14f, -0.2f);

        public static readonly Pose ParryStrike = Guard.Copy()
            .Set(Joint.Chest, 2, 26).Set(Joint.ShoulderL, -78, -48, 0).Set(Joint.ElbowL, -62).Set(Joint.HandL, -25);

        public static readonly Pose Victory = new Pose()
            .Set(Joint.Hips, 0, 18).Set(Joint.Spine, -4).Set(Joint.Chest, -6, 6).Set(Joint.Neck, -12, -8)
            .Set(Joint.ShoulderR, -172, 0, -8).Set(Joint.ElbowR, -12).Set(Joint.ShoulderL, 4, 0, 10).Set(Joint.ElbowL, -20)
            .Set(Joint.HipL, -6, 0, -6).Set(Joint.KneeL, 6).Set(Joint.HipR, 6, 0, 6).Set(Joint.KneeR, 6)
            .Hip(0, 0f, 0);

        public static readonly Pose Bow = new Pose()
            .Set(Joint.Hips, 0, 0).Set(Joint.Spine, 26).Set(Joint.Chest, 12).Set(Joint.Neck, 12)
            .Set(Joint.ShoulderL, -18, 0, 20).Set(Joint.ElbowL, -100).Set(Joint.ShoulderR, -18, 0, -20).Set(Joint.ElbowR, -100)
            .Hip(0, -0.01f, -0.04f);

        public static readonly Pose Grabbed = Guard.Copy()
            .Set(Joint.Spine, 28).Set(Joint.Chest, 18).Set(Joint.Neck, 12)
            .Set(Joint.ShoulderL, -40, 0, 30).Set(Joint.ElbowL, -50).Set(Joint.ShoulderR, -40, 0, -30).Set(Joint.ElbowR, -50)
            .Hip(0, -0.1f, 0.06f);

        public static readonly Pose ThrowGrab = Guard.Copy()
            .Set(Joint.Spine, 14).Set(Joint.Chest, 6)
            .Set(Joint.ShoulderL, -82, 0, 18).Set(Joint.ElbowL, -38).Set(Joint.ShoulderR, -80, 0, -18).Set(Joint.ElbowR, -40)
            .Hip(0, -0.1f, 0.05f);

        public static readonly Pose ThrowHeave = ThrowGrab.Copy()
            .Set(Joint.Hips, 0, -70).Set(Joint.Spine, -10, -20, 10).Set(Joint.ShoulderL, -120, 0, 30).Set(Joint.ShoulderR, -60, 0, -50)
            .Hip(0, -0.18f, 0f);

        private static readonly Dictionary<AttackStyle, (Pose windup, Pose strike)> Attacks = BuildAttacks();

        private static Dictionary<AttackStyle, (Pose, Pose)> BuildAttacks()
        {
            var d = new Dictionary<AttackStyle, (Pose, Pose)>();
            d[AttackStyle.Jab] = (Guard.Copy().Set(Joint.ShoulderL, -60, 0, 8).Set(Joint.ElbowL, -125),
                Guard.Copy().Set(Joint.Hips, 0, 12).Set(Joint.Chest, 4, -8).Set(Joint.ShoulderL, -90, 0, 4).Set(Joint.ElbowL, -6).Set(Joint.Neck, -6, -10).Hip(0, -0.07f, 0.04f));
            d[AttackStyle.CrouchJab] = (Crouch.Copy().Set(Joint.ShoulderL, -60, 0, 8).Set(Joint.ElbowL, -120),
                Crouch.Copy().Set(Joint.Chest, 6, -8).Set(Joint.ShoulderL, -98, 0, 4).Set(Joint.ElbowL, -6));
            d[AttackStyle.Cross] = (Guard.Copy().Set(Joint.Chest, 4, 16).Set(Joint.Hips, 0, 28),
                Guard.Copy().Set(Joint.Hips, 0, 2).Set(Joint.Spine, 10, -4).Set(Joint.Chest, 6, -14).Set(Joint.ShoulderR, -92, 0, -2).Set(Joint.ElbowR, -6)
                    .Set(Joint.ShoulderL, -40, 0, 30).Set(Joint.ElbowL, -130).Set(Joint.HipR, 4, 0, 4).Set(Joint.FootR, -45).Hip(0, -0.08f, 0.08f));
            d[AttackStyle.Lunge] = (d[AttackStyle.Cross].Item1, d[AttackStyle.Cross].Item2.Copy().Set(Joint.Spine, 18, -10).Set(Joint.HipL, -40).Set(Joint.KneeL, 50).Hip(0, -0.16f, 0.2f));
            d[AttackStyle.LeadHook] = (Guard.Copy().Set(Joint.Chest, 4, 22),
                Guard.Copy().Set(Joint.Hips, 0, 4).Set(Joint.Chest, 6, -12).Set(Joint.ShoulderL, -82, 62, 0).Set(Joint.ElbowL, -95).Hip(0, -0.09f, 0.03f));
            d[AttackStyle.BodyHook] = (Crouch.Copy().Set(Joint.Hips, 0, 30),
                Guard.Copy().Set(Joint.Spine, 22, -6, 10).Set(Joint.Chest, 6, -14).Set(Joint.ShoulderR, -60, -55, 0).Set(Joint.ElbowR, -95).Hip(0, -0.2f, 0.05f));
            d[AttackStyle.Uppercut] = (Crouch.Copy().Set(Joint.ShoulderR, -10, 0, -10).Set(Joint.ElbowR, -100),
                Guard.Copy().Set(Joint.Hips, 0, 2).Set(Joint.Spine, -6, -6).Set(Joint.Chest, -8, -12).Set(Joint.ShoulderR, -150, 0, -6).Set(Joint.ElbowR, -40)
                    .Set(Joint.KneeL, 10).Set(Joint.KneeR, 10).Hip(0, 0.02f, 0.05f));
            d[AttackStyle.Elbow] = (Guard.Copy().Set(Joint.Chest, 4, 24),
                Guard.Copy().Set(Joint.Chest, 6, -10).Set(Joint.ShoulderL, -92, 72, 0).Set(Joint.ElbowL, -150).Hip(0, -0.08f, 0.06f));
            d[AttackStyle.Overhand] = (Guard.Copy().Set(Joint.ShoulderR, -150, 0, -30).Set(Joint.ElbowR, -60).Set(Joint.Chest, -6, 20),
                Guard.Copy().Set(Joint.Spine, 20, -6).Set(Joint.Chest, 14, -16).Set(Joint.ShoulderR, -100, 0, 10).Set(Joint.ElbowR, -10).Hip(0, -0.14f, 0.1f));
            d[AttackStyle.LeapPunch] = (Crouch, d[AttackStyle.Cross].Item2.Copy().Set(Joint.HipL, -60).Set(Joint.KneeL, 80).Set(Joint.HipR, 30).Hip(0, 0.18f, 0.2f));
            d[AttackStyle.Charge] = (Crouch.Copy().Set(Joint.Spine, 30, 40),
                Guard.Copy().Set(Joint.Hips, 0, 60).Set(Joint.Spine, 32, 10).Set(Joint.Chest, 10).Set(Joint.ShoulderL, -40, 0, 50).Set(Joint.ElbowL, -100)
                    .Set(Joint.HipL, -50).Set(Joint.KneeL, 50).Hip(0, -0.2f, 0.15f));

            d[AttackStyle.FrontKick] = (Guard.Copy().Set(Joint.HipL, -90).Set(Joint.KneeL, 110).Set(Joint.Spine, -4),
                Guard.Copy().Set(Joint.Spine, -12).Set(Joint.HipL, -88, 0, 0).Set(Joint.KneeL, 4).Set(Joint.FootL, -40).Set(Joint.KneeR, 14).Hip(0, -0.04f, 0.04f));
            d[AttackStyle.Roundhouse] = (Guard.Copy().Set(Joint.Hips, 0, -20).Set(Joint.HipR, -70, 0, 30).Set(Joint.KneeR, 110),
                Guard.Copy().Set(Joint.Hips, 0, -75).Set(Joint.Spine, 0, 0, 22).Set(Joint.Chest, -4, 10, 10).Set(Joint.HipR, -70, 0, 72).Set(Joint.KneeR, 8)
                    .Set(Joint.ShoulderR, 10, 0, -20).Set(Joint.ElbowR, -40).Set(Joint.KneeL, 10).Hip(0, -0.03f, 0f));
            d[AttackStyle.SideKick] = (Guard.Copy().Set(Joint.Hips, 0, 70).Set(Joint.HipL, -40, 0, -40).Set(Joint.KneeL, 120),
                Guard.Copy().Set(Joint.Hips, 0, 82).Set(Joint.Spine, 0, 0, 18).Set(Joint.HipL, 0, 0, -86).Set(Joint.KneeL, 2).Set(Joint.KneeR, 12).Hip(0, -0.05f, -0.05f));
            d[AttackStyle.AxeKick] = (Guard.Copy().Set(Joint.HipR, -150, 0, 20).Set(Joint.KneeR, 5).Set(Joint.Spine, -14),
                Guard.Copy().Set(Joint.Spine, 8).Set(Joint.HipR, -95, 0, 0).Set(Joint.KneeR, 0).Set(Joint.FootR, -10).Hip(0, -0.06f, 0.06f));
            d[AttackStyle.Crescent] = (Guard.Copy().Set(Joint.HipL, -70, 0, 40).Set(Joint.KneeL, 30),
                Guard.Copy().Set(Joint.Hips, 0, 40).Set(Joint.HipL, -110, 0, -30).Set(Joint.KneeL, 5).Hip(0, -0.04f, 0f));
            d[AttackStyle.Spin] = (Guard.Copy().Set(Joint.Hips, 0, 160).Set(Joint.Spine, 10),
                Guard.Copy().Set(Joint.Hips, 0, 185).Set(Joint.Spine, 14, 0, 0).Set(Joint.HipR, 82, 0, 0).Set(Joint.KneeR, 4).Set(Joint.FootR, 40).Set(Joint.KneeL, 14).Hip(0, -0.06f, 0f));
            d[AttackStyle.Knee] = (Guard.Copy().Set(Joint.Spine, -6),
                Guard.Copy().Set(Joint.Spine, -14).Set(Joint.Hips, 0, -10).Set(Joint.HipR, -110).Set(Joint.KneeR, 135)
                    .Set(Joint.ShoulderL, -95, 0, 30).Set(Joint.ElbowL, -70).Set(Joint.ShoulderR, -95, 0, -30).Set(Joint.ElbowR, -70).Hip(0, 0.02f, 0.12f));
            d[AttackStyle.LowKick] = (Crouch.Copy().Set(Joint.HipL, -50, 0, 0).Set(Joint.KneeL, 90),
                Crouch.Copy().Set(Joint.Spine, 10).Set(Joint.HipL, -70, 0, 0).Set(Joint.KneeL, 4).Set(Joint.FootL, -30).Hip(0, -0.36f, 0.05f));
            d[AttackStyle.Sweep] = (Crouch.Copy().Set(Joint.Hips, 0, -40),
                Crouch.Copy().Set(Joint.Hips, 0, -110).Set(Joint.Spine, 30, 0, -10).Set(Joint.HipR, -80, 0, 30).Set(Joint.KneeR, 4).Hip(0, -0.5f, 0f));
            d[AttackStyle.AirKick] = (Airborne.Copy().Set(Joint.HipR, -90).Set(Joint.KneeR, 120),
                Airborne.Copy().Set(Joint.Spine, -10).Set(Joint.HipR, -75, 0, 0).Set(Joint.KneeR, 2).Set(Joint.FootR, -30).Set(Joint.HipL, -40).Set(Joint.KneeL, 120));
            d[AttackStyle.AirPunch] = (Airborne.Copy().Set(Joint.ShoulderR, -170).Set(Joint.ElbowR, -40).Set(Joint.ShoulderL, -170).Set(Joint.ElbowL, -40),
                Airborne.Copy().Set(Joint.Spine, 20).Set(Joint.ShoulderR, -80).Set(Joint.ElbowR, -20).Set(Joint.ShoulderL, -80).Set(Joint.ElbowL, -20));
            d[AttackStyle.Throw] = (Guard, ThrowGrab);
            d[AttackStyle.WeaponSlash] = (Guard.Copy().Set(Joint.Hips, 0, 32).Set(Joint.Chest, 4, 22).Set(Joint.ShoulderR, -70, -70, -20).Set(Joint.ElbowR, -70),
                Guard.Copy().Set(Joint.Hips, 0, 0).Set(Joint.Spine, 8, -6).Set(Joint.Chest, 6, -14).Set(Joint.ShoulderR, -88, 55, 0).Set(Joint.ElbowR, -12)
                    .Set(Joint.HipL, -34).Set(Joint.KneeL, 40).Hip(0, -0.1f, 0.06f));
            d[AttackStyle.TwoHandThrust] = (Guard.Copy().Set(Joint.ShoulderR, -40, 0, -10).Set(Joint.ElbowR, -95).Set(Joint.ShoulderL, -60, 0, 20).Set(Joint.ElbowL, -80),
                Guard.Copy().Set(Joint.Spine, 12, -4).Set(Joint.Chest, 6, -8).Set(Joint.ShoulderR, -84, 0, -6).Set(Joint.ElbowR, -14)
                    .Set(Joint.ShoulderL, -88, 0, 14).Set(Joint.ElbowL, -10).Set(Joint.HipL, -40).Set(Joint.KneeL, 46).Hip(0, -0.1f, 0.12f));
            d[AttackStyle.LowSlash] = (Crouch.Copy().Set(Joint.ShoulderR, -40, -60, -10).Set(Joint.ElbowR, -60),
                Crouch.Copy().Set(Joint.Spine, 28, -8).Set(Joint.Chest, 10, -10).Set(Joint.ShoulderR, -60, 60, 0).Set(Joint.ElbowR, -10));
            d[AttackStyle.Special] = (Crouch.Copy().Set(Joint.Spine, 26, 30).Set(Joint.ShoulderR, -20, 0, -30).Set(Joint.ElbowR, -130),
                Guard.Copy().Set(Joint.Hips, 0, -4).Set(Joint.Spine, 16, -6).Set(Joint.Chest, 8, -16).Set(Joint.ShoulderR, -94).Set(Joint.ElbowR, -2)
                    .Set(Joint.ShoulderL, -94, 0, 20).Set(Joint.ElbowL, -20).Set(Joint.HipL, -55).Set(Joint.KneeL, 60).Set(Joint.HipR, 25).Set(Joint.KneeR, 10).Hip(0, -0.2f, 0.3f));
            return d;
        }

        private static readonly Dictionary<string, AttackStyle> ById = new Dictionary<string, AttackStyle>
        {
            ["jab"] = AttackStyle.Jab, ["reverse_punch"] = AttackStyle.Jab, ["short_jab"] = AttackStyle.Jab, ["crouch_jab"] = AttackStyle.CrouchJab,
            ["cross"] = AttackStyle.Cross, ["lunge_punch"] = AttackStyle.Lunge, ["hammer_fist"] = AttackStyle.Overhand, ["elbow"] = AttackStyle.Elbow,
            ["lead_hook"] = AttackStyle.LeadHook, ["body_hook"] = AttackStyle.BodyHook, ["overhand"] = AttackStyle.Overhand, ["uppercut"] = AttackStyle.Uppercut,
            ["superman_punch"] = AttackStyle.LeapPunch, ["shoulder_charge"] = AttackStyle.Charge,
            ["front_kick"] = AttackStyle.FrontKick, ["push_kick"] = AttackStyle.FrontKick, ["stomp_kick"] = AttackStyle.FrontKick, ["big_boot"] = AttackStyle.FrontKick,
            ["roundhouse"] = AttackStyle.Roundhouse, ["round_kick"] = AttackStyle.Roundhouse,
            ["side_kick"] = AttackStyle.SideKick, ["axe_kick"] = AttackStyle.AxeKick, ["crescent_kick"] = AttackStyle.Crescent,
            ["spin_back_kick"] = AttackStyle.Spin, ["spin_hook_kick"] = AttackStyle.Spin, ["spin_elbow"] = AttackStyle.Spin,
            ["knee"] = AttackStyle.Knee, ["low_kick"] = AttackStyle.LowKick, ["sweep"] = AttackStyle.Sweep,
            ["jump_kick"] = AttackStyle.AirKick, ["jump_heavy_kick"] = AttackStyle.AirKick, ["jump_punch"] = AttackStyle.AirPunch,
            ["special"] = AttackStyle.Special,
            ["knife_jab"] = AttackStyle.Cross, ["knife_slash"] = AttackStyle.WeaponSlash, ["knife_backhand"] = AttackStyle.WeaponSlash,
            ["knife_lunge"] = AttackStyle.Lunge, ["knife_low"] = AttackStyle.LowSlash, ["knife_rising"] = AttackStyle.Uppercut,
            ["sword_thrust"] = AttackStyle.Cross, ["sword_slash"] = AttackStyle.WeaponSlash, ["sword_riposte"] = AttackStyle.Cross,
            ["sword_overhead"] = AttackStyle.Overhand, ["sword_low"] = AttackStyle.LowSlash, ["sword_rising"] = AttackStyle.Uppercut,
            ["staff_poke"] = AttackStyle.TwoHandThrust, ["staff_swing"] = AttackStyle.WeaponSlash, ["staff_rising"] = AttackStyle.Uppercut,
            ["staff_vault"] = AttackStyle.Overhand, ["staff_low"] = AttackStyle.LowSlash, ["staff_sweep"] = AttackStyle.LowSlash,
            ["hook"] = AttackStyle.LeadHook, ["body_blow"] = AttackStyle.BodyHook, ["palm"] = AttackStyle.Jab, ["knee_flick"] = AttackStyle.Knee,
            ["stop_kick"] = AttackStyle.FrontKick,
        };

        public static AttackStyle StyleFor(MoveData move)
        {
            if (move == null) return AttackStyle.Jab;
            if (move.IsThrow) return AttackStyle.Throw;
            if (move.AnimationId != null && ById.TryGetValue(move.AnimationId, out var s)) return s;
            switch (move.Button)
            {
                case AttackButton.HeavyPunch: return AttackStyle.Cross;
                case AttackButton.LightKick: return move.IsAirMove ? AttackStyle.AirKick : AttackStyle.FrontKick;
                case AttackButton.HeavyKick: return move.IsAirMove ? AttackStyle.AirKick : AttackStyle.Roundhouse;
                case AttackButton.Special: return AttackStyle.Special;
                default: return AttackStyle.Jab;
            }
        }

        public static (Pose windup, Pose strike) For(AttackStyle style) => Attacks[style];

        /// <summary>Rest pose an attack starts from and returns to.</summary>
        public static Pose BaseFor(AttackStyle style)
        {
            switch (style)
            {
                case AttackStyle.CrouchJab:
                case AttackStyle.LowKick:
                case AttackStyle.Sweep:
                case AttackStyle.LowSlash:
                    return Crouch;
                case AttackStyle.AirKick:
                case AttackStyle.AirPunch:
                    return Airborne;
                default:
                    return Guard;
            }
        }
    }
}
