# Harvester Game Parameterized Unit Tests
# ─────────────────────────────────────────────────────────────────────────────
# Companion to test_controller.gd. Never committed to the game repo.
# Run via: godot --headless --path project/ --script <this file> -- --test
#
# Covers pure logic units that do NOT need the live game scene:
#   - MathUtils.get_triangle_circumcircle_radius (utils/math_utils.gd)
#   - StatsShip.set_health / modifiers             (ships/player/stats_ship.gd)
#
# Cases are data tables split into three groups: normal / boundary / exception.
# Outputs TEST_RESULT: <json> lines that the Python agent parses.
# ─────────────────────────────────────────────────────────────────────────────
extends SceneTree

# Scripts are loaded by path, not via class_name globals: in --script mode the
# autoload singletons are unreliable at compile time, and loading by path keeps
# this file runnable from any location (see test_controller.gd header note).
const MATH_UTILS_PATH := "res://utils/math_utils.gd"
const STATS_SHIP_PATH := "res://ships/player/stats_ship.gd"

# ── Case tables ──────────────────────────────────────────────────────────────
# Each entry: {"group", "name", ...inputs..., "expected" / assertion fields}.

# get_triangle_circumcircle_radius(verts) -> radius (within FLOAT_TOLERANCE)
const CIRCLE_CASES := [
	# ── normal ────────────────────────────────────────────────────────────
	{"group": "normal", "name": "right_triangle_3_4_5",
		"verts": [Vector2(0, 0), Vector2(4, 0), Vector2(0, 3)],
		"expected": 2.5},  # circumradius of a right triangle = hypotenuse / 2
	{"group": "normal", "name": "equilateral_side_6",
		"verts": [Vector2(0, 0), Vector2(6, 0), Vector2(3, 3.0 * sqrt(3.0))],
		"expected": 2.0 * sqrt(3.0)},  # R = side / sqrt(3)
	{"group": "normal", "name": "negative_coordinates",
		"verts": [Vector2(-4, 0), Vector2(0, 0), Vector2(0, -3)],
		"expected": 2.5},
	# ── boundary ──────────────────────────────────────────────────────────
	{"group": "boundary", "name": "collinear_horizontal_fallback",
		"verts": [Vector2(0, 0), Vector2(1, 0), Vector2(2, 0)],
		"expected": 1.0},  # G ~= 0 -> fallback: max extent * 0.5
	{"group": "boundary", "name": "duplicate_vertex_fallback",
		"verts": [Vector2(1, 1), Vector2(1, 1), Vector2(3, 5)],
		"expected": sqrt(1.5 * 1.5 + 2.5 * 2.5)},  # G == 0 -> fallback branch
	{"group": "boundary", "name": "tiny_triangle",
		"verts": [Vector2(0, 0), Vector2(0.004, 0), Vector2(0, 0.003)],
		"expected": 0.0025, "tolerance": 1e-5},
	# ── exception ─────────────────────────────────────────────────────────
	{"group": "exception", "name": "collinear_vertical_fallback",
		"verts": [Vector2(0, 0), Vector2(0, 5), Vector2(0, 10)],
		"expected": 5.0},  # degenerate input, must not crash or return inf
	{"group": "exception", "name": "huge_coordinates",
		"verts": [Vector2(0, 0), Vector2(4e8, 0), Vector2(0, 3e8)],
		"expected": 2.5e8, "tolerance": 2.5e4},  # relative 1e-4
	{"group": "exception", "name": "result_is_finite_for_degenerate",
		"verts": [Vector2(2, 2), Vector2(2, 2), Vector2(2, 2)],
		"expected": sqrt(2.0)},  # all identical -> fallback is origin-relative:
		# dx=max_x*0.5, dy=max_y*0.5 -> R = |(1, 1)| = sqrt(2)
]

# stats.set_health(input) -> {clamped health, health_depleted fired}
const HEALTH_CASES := [
	# ── normal ────────────────────────────────────────────────────────────
	{"group": "normal", "name": "mid_damage",
		"input": 50.0, "expected_health": 50.0, "expect_depleted": false},
	{"group": "normal", "name": "full_heal_to_max",
		"input": 100.0, "expected_health": 100.0, "expect_depleted": false},
	# ── boundary ──────────────────────────────────────────────────────────
	{"group": "boundary", "name": "exactly_zero",
		"input": 0.0, "expected_health": 0.0, "expect_depleted": true},
	{"group": "boundary", "name": "just_above_zero",
		"input": 0.001, "expected_health": 0.001, "expect_depleted": false},
	{"group": "boundary", "name": "just_below_max",
		"input": 99.999, "expected_health": 99.999, "expect_depleted": false},
	# ── exception ─────────────────────────────────────────────────────────
	{"group": "exception", "name": "negative_clamps_to_zero",
		"input": -25.0, "expected_health": 0.0, "expect_depleted": true},
	{"group": "exception", "name": "over_max_clamps_to_max",
		"input": 150.0, "expected_health": 100.0, "expect_depleted": false},
]

# add_modifier("linear_speed_max", m) then read back; base is 350.0
const MODIFIER_CASES := [
	# ── normal ────────────────────────────────────────────────────────────
	{"group": "normal", "name": "single_positive",
		"modifiers": [50.0], "expected": 400.0},
	{"group": "normal", "name": "mixed_accumulate",
		"modifiers": [50.0, -30.0], "expected": 370.0},
	# ── boundary ──────────────────────────────────────────────────────────
	{"group": "boundary", "name": "zero_modifier_no_change",
		"modifiers": [0.0], "expected": 350.0},
	{"group": "boundary", "name": "modifier_lands_exactly_on_zero",
		"modifiers": [-350.0], "expected": 0.0},
	# ── exception ─────────────────────────────────────────────────────────
	{"group": "exception", "name": "overshoot_goes_negative",
		"modifiers": [-1000.0], "expected": -650.0},  # get_stat does NOT clamp
	{"group": "exception", "name": "reset_restores_base",
		"modifiers": [50.0, -30.0], "reset": true, "expected": 350.0},
]

const FLOAT_TOLERANCE := 1e-4

var _math_utils: GDScript = null
var _stats_ship_script: GDScript = null
var _depleted_count := 0
var _live_stats: Array = []  # keep Resources alive while signals are connected


func _initialize() -> void:
	if "--test" not in OS.get_cmdline_user_args():
		quit()
		return

	_math_utils = load(MATH_UTILS_PATH)
	if _math_utils == null:
		_emit_result("setup.load_math_utils", false, "Could not load " + MATH_UTILS_PATH)
		_finish()
		return
	_stats_ship_script = load(STATS_SHIP_PATH)
	if _stats_ship_script == null:
		_emit_result("setup.load_stats_ship", false, "Could not load " + STATS_SHIP_PATH)
		_finish()
		return

	_run_circle_cases()
	_run_health_cases()
	_run_modifier_cases()
	_finish()


func _process(_delta: float) -> bool:
	return false


# ── Case runners ─────────────────────────────────────────────────────────────

func _run_circle_cases() -> void:
	for case in CIRCLE_CASES:
		var verts := PackedVector2Array(case["verts"])
		var actual: float = _math_utils.get_triangle_circumcircle_radius(verts)
		var tol: float = case.get("tolerance", FLOAT_TOLERANCE)
		var ok: bool = abs(actual - case["expected"]) <= tol and not is_nan(actual) and not is_inf(actual)
		_emit_result(
			"circumcircle.%s.%s" % [case["group"], case["name"]],
			ok,
			"verts=%s expected=%s actual=%s" % [str(verts), str(case["expected"]), str(actual)],
			"utils/math_utils.gd",
			{"group": case["group"]}
		)


func _run_health_cases() -> void:
	for case in HEALTH_CASES:
		var stats: Resource = _stats_ship_script.new()
		_live_stats.append(stats)
		stats.initialize()
		_depleted_count = 0
		stats.connect("health_depleted", _on_health_depleted)
		stats.set_health(case["input"])
		var health_ok: bool = abs(stats.health - case["expected_health"]) <= FLOAT_TOLERANCE
		var signal_ok: bool = (_depleted_count > 0) == case["expect_depleted"]
		_emit_result(
			"set_health.%s.%s" % [case["group"], case["name"]],
			health_ok and signal_ok,
			"input=%s expected_health=%s actual=%s depleted_fired=%d" % [
				str(case["input"]), str(case["expected_health"]),
				str(stats.health), _depleted_count],
			"ships/player/stats_ship.gd",
			{"group": case["group"]}
		)


func _run_modifier_cases() -> void:
	for case in MODIFIER_CASES:
		var stats: Resource = _stats_ship_script.new()
		_live_stats.append(stats)
		stats.initialize()
		for m in case["modifiers"]:
			stats.add_modifier("linear_speed_max", m)
		if case.get("reset", false):
			stats.reset()
		var actual: float = stats.get_linear_speed_max()
		var ok: bool = abs(actual - case["expected"]) <= FLOAT_TOLERANCE
		_emit_result(
			"modifier.%s.%s" % [case["group"], case["name"]],
			ok,
			"modifiers=%s reset=%s expected=%s actual=%s" % [
				str(case["modifiers"]), str(case.get("reset", false)),
				str(case["expected"]), str(actual)],
			"ships/player/stats_ship.gd",
			{"group": case["group"]}
		)


func _on_health_depleted() -> void:
	_depleted_count += 1


# ── Output helpers ───────────────────────────────────────────────────────────

func _emit_result(
	scenario:    String,
	passed:      bool,
	message:     String  = "",
	file:        String  = "",
	extra:       Dictionary = {}
) -> void:
	var result := {
		"scenario":  scenario,
		"passed":    passed,
		"message":   message,
		"file":      file,
	}
	result.merge(extra)
	print("TEST_RESULT:" + JSON.stringify(result))


func _finish() -> void:
	quit()
