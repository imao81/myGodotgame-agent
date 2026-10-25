# Harvester Game Test Controller
# ─────────────────────────────────────────────────────────────────────────────
# Injected by the Playtest Agent. Never committed to the game repo.
# Run via: godot --headless --path project/ --script autoload/test_controller.gd -- --test
#
# Extends SceneTree so it can load the game scene and observe Events signals
# without requiring a real display.
#
# Outputs TEST_RESULT: <json> lines that the Python agent parses.
# ─────────────────────────────────────────────────────────────────────────────
extends SceneTree

const GAME_SCENE_PATH := "res://main/game.tscn"
const WAIT_FOR_SPAWN := 3.0   # seconds to wait for initial spawning
const TIMEOUT       := 12.0   # hard timeout before force-quit

# ── Observed state ─────────────────────────────────────────────────────────
var _player: Node    = null
var _station: Node   = null
var _pirates: Array  = []
var _asteroids: Array = []

var _mine_started := false
var _mine_finished := false
var _player_died := false
var _upgrade_unlocked := false
var _damage_events := 0

# Full observation table for the runtime verifier — every watched signal,
# initialised to 0 so "never fired" is distinguishable from "not watched".
var _signal_counts := {
    "station_spawned": 0,
    "asteroid_spawned": 0,
    "pirate_spawned": 0,
    "player_died": 0,
    "mine_started": 0,
    "mine_finished": 0,
    "upgrade_unlocked": 0,
    "damaged": 0,
}

var _elapsed := 0.0
var _phase   := "init"


func _initialize() -> void:
	if "--test" not in OS.get_cmdline_user_args():
		quit()
		return

	# Hook Events signals before loading the scene. NOTE: scripts run via
	# --script cannot resolve autoload globals at compile time ("Identifier
	# not found: Events"), although the singleton IS in the tree at runtime —
	# so connect through the node, not the global identifier.
	var events: Node = get_root().get_node_or_null("Events")
	if events == null:
		_emit_result("events_autoload_available", false,
			"Events autoload not found in the scene tree")
		_finish()
		return
	events.connect("station_spawned", _on_station_spawned)
	events.connect("asteroid_spawned", _on_asteroid_spawned)
	events.connect("pirate_spawned", _on_pirate_spawned)
	events.connect("player_died", _on_player_died)
	events.connect("mine_started", _on_mine_started)
	events.connect("mine_finished", _on_mine_finished)
	events.connect("upgrade_unlocked", _on_upgrade_unlocked)
	events.connect("damaged", _on_damaged)

	# Load the game scene directly (skip main menu)
	var packed = load(GAME_SCENE_PATH)
	if not packed:
		_emit_result("game_scene_loads", false, "Could not load " + GAME_SCENE_PATH)
		_finish()
		return

	var scene = packed.instantiate()
	get_root().add_child(scene)
	_phase = "spawning"


func _process(delta: float) -> bool:
	_elapsed += delta
	if _elapsed >= TIMEOUT:
		_run_assertions()
		_finish()
		return false   # stop processing
	if _phase == "spawning" and _elapsed >= WAIT_FOR_SPAWN:
		_phase = "testing"
		_run_assertions()
		_finish()
	return false


# ── Signal handlers ─────────────────────────────────────────────────────────

func _count(signal_name: String) -> void:
	_signal_counts[signal_name] = _signal_counts.get(signal_name, 0) + 1


func _on_station_spawned(station: Node, player: Node) -> void:
	_count("station_spawned")
	_station = station
	_player  = player

func _on_asteroid_spawned(_object: Node) -> void:
	_count("asteroid_spawned")
	_asteroids.append(_object)

func _on_pirate_spawned(pirate: Node) -> void:
	_count("pirate_spawned")
	_pirates.append(pirate)

func _on_player_died() -> void:
	_count("player_died")
	_player_died = true

func _on_mine_started(_pos: Variant) -> void:
	_count("mine_started")
	_mine_started = true

func _on_mine_finished() -> void:
	_count("mine_finished")
	_mine_finished = true

func _on_upgrade_unlocked() -> void:
	_count("upgrade_unlocked")
	_upgrade_unlocked = true

func _on_damaged(_target: Variant, _damage: Variant, _shooter: Variant) -> void:
	_count("damaged")
	_damage_events += 1


# ── Test assertions ──────────────────────────────────────────────────────────

func _run_assertions() -> void:
	# ── T1: Game scene loaded and station spawned ────────────────────────
	_emit_result(
		"station_spawns_on_load",
		_station != null,
		"Expected Events.station_spawned to fire within %.1fs" % WAIT_FOR_SPAWN,
		"world/spawners/station_spawner.gd"
	)

	# ── T2: Player ship created ──────────────────────────────────────────
	_emit_result(
		"player_ship_created",
		_player != null,
		"Expected player ship to be passed via Events.station_spawned",
		"ships/player/player_ship.gd"
	)

	# ── T3: At least one asteroid cluster spawned ────────────────────────
	_emit_result(
		"asteroids_spawn_on_load",
		_asteroids.size() > 0,
		"Expected Events.asteroid_spawned to fire at least once (got %d)" % _asteroids.size(),
		"world/spawners/asteroid_spawner.gd"
	)

	# ── T4: At least one pirate group spawned ────────────────────────────
	_emit_result(
		"pirates_spawn_on_load",
		_pirates.size() > 0,
		"Expected Events.pirate_spawned to fire at least once (got %d)" % _pirates.size(),
		"ships/enemies/pirate/pirate_spawner.gd"
	)

	# ── T5: Player did NOT die immediately ───────────────────────────────
	_emit_result(
		"player_survives_spawn",
		not _player_died,
		"Player died before %.1fs — possible spawn collision or instant damage" % WAIT_FOR_SPAWN
	)

	# ── T6: Player has valid stats ───────────────────────────────────────
	if _player != null and _player.has_method("get") and _player.get("stats") != null:
		var stats = _player.stats
		var health_ok = stats.get_max_health() > 0.0
		var speed_ok  = stats.get_linear_speed_max() > 0.0
		_emit_result(
			"player_stats_valid",
			health_ok and speed_ok,
			"Player stats appear invalid: max_health=%s linear_speed=%s" % [
				str(stats.get_max_health()) if health_ok else "0",
				str(stats.get_linear_speed_max()) if speed_ok else "0",
			],
			"ships/player/stats_ship.gd",
			{"balance_issue": "player stats are zero — check player_stats.tres"} if not (health_ok and speed_ok) else {}
		)
	else:
		_emit_result(
			"player_stats_valid",
			false,
			"Could not access player.stats — ship may not have initialized",
			"ships/player/player_ship.gd"
		)

	# ── T7: Pirate ships have state machines ────────────────────────────
	if _pirates.size() > 0:
		var pirate = _pirates[0]
		var has_state = pirate.has_node("StateMachine") or pirate.get("_state_machine") != null
		_emit_result(
			"pirate_has_state_machine",
			has_state,
			"First pirate has no StateMachine node — AI behavior broken",
			"ships/enemies/pirate/pirate_ship.gd"
		)

	# ── T8: Spawn count balance check ───────────────────────────────────
	# If zero pirates spawned when asteroids are present, something is wrong
	if _asteroids.size() > 0 and _pirates.size() == 0:
		_emit_result(
			"pirate_count_balance",
			false,
			"Asteroids spawned (%d) but zero pirates — pirate_spawner may be broken" % _asteroids.size(),
			"world/spawners/pirate_spawner.gd",
			{"balance_issue": "pirates not spawning despite asteroids being present"}
		)


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


# ── Coverage collection ──────────────────────────────────────────────────────

func _collect_coverage() -> Array:
	# Walk the live scene tree and collect every script resource actually in
	# use. The injected controller itself is excluded.
	var seen := {}
	_walk_scripts(get_root(), seen)
	return seen.keys()


func _walk_scripts(node: Node, seen: Dictionary) -> void:
	var s = node.get_script()
	if s != null and s.resource_path != "" \
			and not s.resource_path.ends_with("test_controller.gd"):
		seen[s.resource_path] = true
	for child in node.get_children():
		_walk_scripts(child, seen)


func _finish() -> void:
	# Short wait so remaining signals flush
	await create_timer(0.2).timeout
	print("SIGNALS:" + JSON.stringify(_signal_counts))
	print("COVERAGE:" + JSON.stringify(_collect_coverage()))
	quit()
