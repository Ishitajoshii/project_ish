"""Simple Gradio UI for manually stepping through one circuit task."""

from __future__ import annotations

import gradio as gr

from server.environment import CircuitEnvironment
from server.task_loader import load_task

_TASK = load_task("tasks/lp_1khz_budget.json")
_ENV = CircuitEnvironment(_TASK)
_ENV.reset()


def do_step(component: str, delta: float):
    """Apply one action and return compact telemetry for display."""

    obs = _ENV.step({"component": component, "delta": delta})
    return (
        (
            f"fc={obs.current_output_hz:.2f}Hz, error={obs.normalized_error:.3f}, "
            f"cost={obs.current_cost:.3f}, solved={obs.solved}"
        ),
        _ENV.score(),
        _ENV.is_done,
    )


with gr.Blocks(title="circuitrl") as demo:
    gr.Markdown("# circuitrl demo\nQuick manual stepping UI.")
    comp = gr.Dropdown(["R", "C"], value="R", label="Component")
    delta = gr.Slider(-0.5, 0.5, value=0.2, step=0.05, label="Delta")
    out = gr.Textbox(label="Observation")
    score = gr.Number(label="Score")
    done = gr.Checkbox(label="Done")
    run = gr.Button("Step")
    run.click(do_step, inputs=[comp, delta], outputs=[out, score, done])

if __name__ == "__main__":
    demo.launch()
