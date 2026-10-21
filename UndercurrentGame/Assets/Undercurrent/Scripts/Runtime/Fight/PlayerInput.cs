using Undercurrent.Core.Combat;
using UnityEngine;

namespace Undercurrent.Runtime.Fight
{
    /// <summary>
    /// Samples the local player's controls once per simulation tick: on-screen touch controls plus keyboard and
    /// gamepad (for development and controller players). Works with either Unity input backend.
    ///
    /// Keyboard: A/D or arrows move, W/Up jump, S/Down crouch, J=LP K=HP U=LK I=HK, L=Block, Space=Dodge,
    /// O=Parry, H=Throw, N=Special, Esc=Pause.
    /// </summary>
    public static class PlayerInput
    {
        public static FrameInput Sample(TouchControls touch)
        {
            var held = touch != null ? touch.Held : InputButtons.None;
            held |= Keyboard();
            held |= Gamepad();
            // Opposite directions cancel out rather than both being reported.
            if ((held & (InputButtons.Left | InputButtons.Right)) == (InputButtons.Left | InputButtons.Right))
                held &= ~(InputButtons.Left | InputButtons.Right);
            return new FrameInput(held);
        }

        public static bool PausePressed()
        {
#if UC_INPUT_SYSTEM && ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            if (kb != null && kb.escapeKey.wasPressedThisFrame) return true;
            var pad = UnityEngine.InputSystem.Gamepad.current;
            if (pad != null && pad.startButton.wasPressedThisFrame) return true;
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKeyDown(KeyCode.Escape)) return true; // also the Android back button
#endif
            return false;
        }

        private static InputButtons Keyboard()
        {
            var b = InputButtons.None;
#if UC_INPUT_SYSTEM && ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            if (kb != null)
            {
                if (kb.aKey.isPressed || kb.leftArrowKey.isPressed) b |= InputButtons.Left;
                if (kb.dKey.isPressed || kb.rightArrowKey.isPressed) b |= InputButtons.Right;
                if (kb.wKey.isPressed || kb.upArrowKey.isPressed) b |= InputButtons.Up;
                if (kb.sKey.isPressed || kb.downArrowKey.isPressed) b |= InputButtons.Down;
                if (kb.jKey.isPressed) b |= InputButtons.LightPunch;
                if (kb.kKey.isPressed) b |= InputButtons.HeavyPunch;
                if (kb.uKey.isPressed) b |= InputButtons.LightKick;
                if (kb.iKey.isPressed) b |= InputButtons.HeavyKick;
                if (kb.lKey.isPressed) b |= InputButtons.Block;
                if (kb.spaceKey.isPressed) b |= InputButtons.Dodge;
                if (kb.oKey.isPressed) b |= InputButtons.Parry;
                if (kb.hKey.isPressed) b |= InputButtons.Throw;
                if (kb.nKey.isPressed) b |= InputButtons.Special;
            }
#elif ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow)) b |= InputButtons.Left;
            if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)) b |= InputButtons.Right;
            if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow)) b |= InputButtons.Up;
            if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow)) b |= InputButtons.Down;
            if (Input.GetKey(KeyCode.J)) b |= InputButtons.LightPunch;
            if (Input.GetKey(KeyCode.K)) b |= InputButtons.HeavyPunch;
            if (Input.GetKey(KeyCode.U)) b |= InputButtons.LightKick;
            if (Input.GetKey(KeyCode.I)) b |= InputButtons.HeavyKick;
            if (Input.GetKey(KeyCode.L)) b |= InputButtons.Block;
            if (Input.GetKey(KeyCode.Space)) b |= InputButtons.Dodge;
            if (Input.GetKey(KeyCode.O)) b |= InputButtons.Parry;
            if (Input.GetKey(KeyCode.H)) b |= InputButtons.Throw;
            if (Input.GetKey(KeyCode.N)) b |= InputButtons.Special;
#endif
            return b;
        }

        private static InputButtons Gamepad()
        {
            var b = InputButtons.None;
#if UC_INPUT_SYSTEM && ENABLE_INPUT_SYSTEM
            var pad = UnityEngine.InputSystem.Gamepad.current;
            if (pad == null) return b;
            var stick = pad.leftStick.ReadValue() + pad.dpad.ReadValue();
            if (stick.x < -0.4f) b |= InputButtons.Left;
            if (stick.x > 0.4f) b |= InputButtons.Right;
            if (stick.y > 0.5f) b |= InputButtons.Up;
            if (stick.y < -0.5f) b |= InputButtons.Down;
            if (pad.buttonWest.isPressed) b |= InputButtons.LightPunch;
            if (pad.buttonNorth.isPressed) b |= InputButtons.HeavyPunch;
            if (pad.buttonSouth.isPressed) b |= InputButtons.LightKick;
            if (pad.buttonEast.isPressed) b |= InputButtons.HeavyKick;
            if (pad.rightShoulder.isPressed) b |= InputButtons.Block;
            if (pad.rightTrigger.isPressed) b |= InputButtons.Dodge;
            if (pad.leftShoulder.isPressed) b |= InputButtons.Parry;
            if (pad.leftTrigger.isPressed) b |= InputButtons.Throw;
            if (pad.leftStickButton.isPressed || pad.rightStickButton.isPressed) b |= InputButtons.Special;
#endif
            return b;
        }
    }
}
