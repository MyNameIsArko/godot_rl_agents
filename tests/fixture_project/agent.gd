extends "res://addons/godot_rl_agents/controller/ai_controller_2d.gd"

var last_action := 0.0


func get_obs() -> Dictionary:
	return {"obs": [last_action]}


func get_reward() -> float:
	return 1.0 - abs(last_action)


func get_action_space() -> Dictionary:
	return {"action": {"size": 1, "action_type": "continuous"}}


func set_action(action: Dictionary) -> void:
	var values = action.get("action", [])
	if values is Array and not values.is_empty():
		last_action = clampf(float(values[0]), -1.0, 1.0)


func reset() -> void:
	super.reset()
	last_action = 0.0
