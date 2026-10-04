extends "res://addons/godot_rl_agents/controller/ai_controller_2d.gd"

static var shared_action_0 := 0
static var shared_action_1 := 0

var last_action := 0


func get_obs() -> Dictionary:
	return {"obs": [float(last_action), clampf(float(n_steps) / 8.0, -1.0, 1.0)]}


func get_reward() -> float:
	if not terminated:
		return 0.0
	var outcome = _outcome()
	if outcome == "draw":
		return 0.0
	return 1.0 if outcome == "win" else -1.0


func get_action_space() -> Dictionary:
	return {"action": {"size": 2, "action_type": "discrete"}}


func set_action(action: Dictionary) -> void:
	last_action = int(action.get("action", 0))
	if agent_id == "player_0":
		shared_action_0 = last_action
	else:
		shared_action_1 = last_action


func get_info() -> Dictionary:
	if terminated:
		return {"outcome": _outcome()}
	return {}


func _physics_process(_delta) -> void:
	super._physics_process(_delta)
	if n_steps >= 8:
		terminated = true


func reset() -> void:
	super.reset()
	last_action = 0
	if agent_id == "player_0":
		shared_action_0 = 0
		shared_action_1 = 0


func _outcome() -> String:
	if shared_action_0 == shared_action_1:
		return "draw"
	if agent_id == "player_0":
		return "win" if shared_action_0 < shared_action_1 else "loss"
	return "win" if shared_action_1 < shared_action_0 else "loss"
