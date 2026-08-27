"""Constants for continual learning experiments."""

# Split-MNIST task definitions (5 binary tasks)
CL_NUM_TASKS = 5
CL_CLASSES_PER_TASK = 2
CL_TOTAL_CLASSES = CL_NUM_TASKS * CL_CLASSES_PER_TASK  # 10

SPLIT_MNIST_TASKS = [
    (0, 1),  # Task 0
    (2, 3),  # Task 1
    (4, 5),  # Task 2
    (6, 7),  # Task 3
    (8, 9),  # Task 4
]

__all__ = [
    "CL_CLASSES_PER_TASK",
    "CL_NUM_TASKS",
    "CL_TOTAL_CLASSES",
    "SPLIT_MNIST_TASKS",
]
