"""A tiny stand-in for Airflow's DAG model."""
from datetime import datetime


class DAG:
    """A directed acyclic graph of tasks."""

    def __init__(self, dag_id: str, start_date: datetime | None = None):
        self.dag_id = dag_id
        self.start_date = start_date
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)


def get_dag(dag_id: str) -> DAG:
    return DAG(dag_id)
