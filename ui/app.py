"""Simple Gradio UI for manually stepping through one circuit task."""

from __future__ import annotations

import gradio as gr

from models import CircuitAction
from server.environment import CircuitEnvironment
from server.simulator import valid_actions
from server.task_loader import load_task

_TASK = load_task("tasks/lp_1khz_budget.json")
_ENV = CircuitEnvironment({_TASK.task_id: _TASK})
_ENV.reset(_TASK.task_id)


def do_step(action: str):
    """Apply one action and return compact telemetry for display."""

    obs, reward, done = _ENV.step(CircuitAction(action=action))
    return (
        (
            f"fc={obs.current_hz:.2f}Hz, error={obs.normalized_error:.3f}, "
            f"cost={obs.current_cost:.3f}, reward={reward:.3f}, done={done}"
        ),
        _ENV.score(),
        done,
    )


with gr.Blocks(title="circuitrl") as demo:
    gr.Markdown("# circuitrl demo\nQuick manual stepping UI.")
    action = gr.Dropdown(list(valid_actions()), value="r_up", label="Action")
    out = gr.Textbox(label="Observation")
    score = gr.Number(label="Score")
    done = gr.Checkbox(label="Done")
    run = gr.Button("Step")
    run.click(do_step, inputs=[action], outputs=[out, score, done])

if __name__ == "__main__":
    demo.launch()
