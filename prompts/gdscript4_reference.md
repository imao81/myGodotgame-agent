# GDScript 4.x runtime semantics (engine version: Godot {godot_version})
# These are FACTS. Do NOT contradict them in your analysis.

## Threading model
- Godot uses SINGLE-THREADED cooperative multitasking
- `await` YIELDS control to the engine loop — it does NOT block a thread
- There is no "input thread" — all code runs on the main thread
- Never describe `await` as "blocking" or "thread blocking"

## await semantics
- Any GDScript function that contains `await` is implicitly a coroutine — do NOT flag it as "non-async"
- `await X` suspends the current function until signal X fires, then resumes
- If the awaited signal has ALREADY fired before the `await` is reached, execution continues IMMEDIATELY with no suspension (Godot 4.x behavior)
- `await self.ready` in a property setter is SAFE — it is the standard Godot 4 pattern to guard access to @onready nodes when a setter may run before _ready()
- The real risk of `await` is RE-ENTRANCY (the same coroutine being called again while suspended), not blocking

## @onready semantics
- `@onready` defers the assignment expression to `_ready()`, NOT to class-level initialisation
- `@onready` variables are PER-INSTANCE — they are never shared across instances
- `@onready var x := await Y.new(self)` — the `await` runs inside `_ready()`, so `self` is already in the scene tree at that point

## Node lifecycle
- `_ready()` is called once when the node and all its children have entered the scene tree
- When a State object is instantiated AFTER its owner's `_ready()` has completed, `await owner.ready` inside the state's `_ready()` continues IMMEDIATELY in Godot 4.x — no deadlock
- EXCEPTION: if the state object is instantiated and added to the tree BEFORE the owner has emitted `ready`, then `await owner.ready` will suspend until the owner is ready — this is expected and correct behaviour

## Signal patterns
- Signals are per-instance, not class-level
- `is_connected()` is the correct guard to prevent duplicate connections — do NOT flag it as incorrect
- `connect()` / `disconnect()` operate on instance-level signals

## Common safe patterns — do NOT flag these as bugs:
- `await self.ready` inside a setter — standard null-safety for @onready vars
- `@onready var x := $NodePath` — type-inferred, per-instance, runs in _ready()
- `if not signal.is_connected(cb): signal.connect(cb)` — correct duplicate-connection guard
- `func foo(): await something` — this IS async; do not assert method_is_not_async for it
