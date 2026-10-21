#!/usr/bin/env bash
# Runs every check that works without a Unity Editor:
#  1. Core gameplay tests (same NUnit sources Unity's Test Runner uses) via .NET.
#  2. Type-check of Unity-facing scripts against NON-AUTHORITATIVE API stubs, for several define configurations.
# Requires the .NET 8 SDK. Exit code is non-zero on any failure.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

echo "== Core tests"
dotnet test "$here/CoreTests/Undercurrent.Core.Tests.csproj" --nologo -v q

for defines in \
  "UC_INPUT_SYSTEM;ENABLE_INPUT_SYSTEM;UC_URP" \
  "ENABLE_LEGACY_INPUT_MANAGER" \
  "UNITY_ANDROID;ENABLE_LEGACY_INPUT_MANAGER;UC_URP" \
  "UNITY_IOS;UC_INPUT_SYSTEM;ENABLE_INPUT_SYSTEM;ENABLE_LEGACY_INPUT_MANAGER"; do
  echo "== Unity-layer type-check (stubs) with defines: $defines"
  dotnet build "$here/UnityCompileCheck/UnityCompileCheck.csproj" --nologo -v q -warnaserror -p:UcDefines="\"$defines\""
done
echo "All checks passed."
