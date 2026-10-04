extends "res://addons/godot_rl_agents/controller/ai_controller_2d.gd"

var last_action := 0.0


func get_obs() -> Dictionary:
	return {"obs": [last_action, clampf(float(n_steps) / 4.0, 0.0, 1.0)]}


func get_reward() -> float:
	return 1.0 - abs(last_action)


func get_action_space() -> Dictionary:
	return {"action": {"size": 1, "action_type": "continuous"}}


func set_action(action: Dictionary) -> void:
	last_action = clampf(float(action["action"][0]), -1.0, 1.0)


func _physics_process(delta) -> void:
	super._physics_process(delta)
	if n_steps >= 4:
		if last_action < 0:
			terminated = true
		else:
			truncated = true


func reset() -> void:
	super.reset()
	last_action = 0.0
