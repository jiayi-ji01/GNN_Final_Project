from src.config import setup_environment
from src.plotting import plot_training_loss
from src.workflow import train_workflow


def main():
    setup_environment(seed=42)
    workflow, history = train_workflow(seed=42)
    plot_training_loss(history)
    print("E0 workflow training finished.")
    return workflow, history


if __name__ == "__main__":
    main()
