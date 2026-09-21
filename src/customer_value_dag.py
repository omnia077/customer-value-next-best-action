"""Airflow DAG `customer_value_monthly`.

Tasks call the same step functions as `score_monthly.py`. The churn-score step waits for Project 1's DAG
(`telecom_churn_platform`) with an ExternalTaskSensor; until Project 1 publishes scores, `churn_proxy` supplies them.
Airflow is not installed in this dev environment: the file imports Airflow lazily and has not been executed by a
scheduler. Validate with `python -c "import ast;ast.parse(open('src/customer_value_dag.py').read())"` and by running
the same steps via `python src/score_monthly.py`.
"""
from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.sensors.external_task import ExternalTaskSensor
except ImportError:                                   # dev environment without Airflow
    DAG = None

if DAG is not None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import score_monthly as sm

    default_args = {"owner": "data-science", "retries": 1, "retry_delay": timedelta(minutes=10),
                    "email_on_failure": False}
    with DAG(dag_id="customer_value_monthly", start_date=datetime(2024, 10, 1), schedule="0 5 2 * *",
             catchup=False, default_args=default_args, tags=["customer-value", "prepaid"]) as dag:
        tasks = {}
        for name, fn in sm.STEPS:
            if name == "churn_scores":
                wait = ExternalTaskSensor(task_id="wait_for_churn_dag", external_dag_id="telecom_churn_platform",
                                          external_task_id=None, mode="reschedule", timeout=6 * 3600, poke_interval=600)
                tasks["wait_for_churn_dag"] = wait
            tasks[name] = PythonOperator(task_id=name, python_callable=fn)
        order = [n for n, _ in sm.STEPS]
        chain = []
        for n in order:
            if n == "churn_scores":
                chain.append("wait_for_churn_dag")
            chain.append(n)
        for a, b in zip(chain, chain[1:]):
            tasks[a] >> tasks[b]
