from locust import HttpUser, task, between
import random

class ModelServerUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def infer_normal(self):
        self.client.post("/infer", json={
            "model_name": "echo_model",
            "inputs": {"float_input": [[
                random.uniform(0, 1),
                random.uniform(0, 1),
            ]]},
            "priority": "normal",
            "tenant_id": "team_a",
        })

    @task(2)
    def infer_urgent(self):
        self.client.post("/infer", json={
            "model_name": "echo_model",
            "inputs": {"float_input": [[1.0, 2.0]]},
            "priority": "urgent",
            "tenant_id": "team_b",
        })

    @task(1)
    def check_health(self):
        self.client.get("/health")

    @task(1)
    def check_queue(self):
        self.client.get("/queue/stats")