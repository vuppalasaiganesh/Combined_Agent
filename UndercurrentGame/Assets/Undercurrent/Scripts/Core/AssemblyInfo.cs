using System.Runtime.CompilerServices;

// Unity compiles the EditMode tests into their own assembly; they inspect internal simulation state.
[assembly: InternalsVisibleTo("Undercurrent.Tests.EditMode")]
