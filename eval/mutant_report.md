# Mutant Recall Report

Model: `kimi/kimi-k3`

**Recall: 73.1% (19/26 injected bugs detected)**

| Operator | Detected | Total |
|----------|---------:|-----:|
| break_preload_path | 4 | 5 |
| dead_guard_emission | 3 | 4 |
| rename_class_reference | 5 | 5 |
| rename_emitted_signal | 7 | 12 |

## Missed mutants

- `m001` ships/guns/projectiles/projectile.gd [rename_emitted_signal] — Emits Events.damaged_broken which is not declared in the Events autoload
- `m004` world/spawners/station_spawner.gd [rename_emitted_signal] — Emits Events.station_spawned_broken which is not declared in the Events autoload
- `m012` ui/minimap/map_view.gd [break_preload_path] — Preloads res://ui/minimap/map_sprite_missing.tscn which does not exist on disk
- `m015` ui/menus/upgrades/upgrade_menu.gd [rename_emitted_signal] — Emits Events.upgrade_chosen_broken which is not declared in the Events autoload
- `m017` ships/player/cargo.gd [dead_guard_emission] — Events.force_undock.emit() is unreachable behind `if false:`
- `m019` ships/enemies/pirate/states/attack.gd [rename_emitted_signal] — Emits Events.call_off_pursuit_broken which is not declared in the Events autoload
- `m022` world/spawners/asteroid_cluster.gd [rename_emitted_signal] — Emits Events.asteroid_spawned_broken which is not declared in the Events autoload
