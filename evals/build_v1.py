"""Build eval set v1 from merged Airflow bug-fix commits.

Each question was written (by Claude, then meant for human spot-checking) from a fix
commit's title, phrased as a developer would ask about the behavior and avoiding the
file name. Gold evidence = the non-test files under airflow/models/ that the commit
changed, read from `git show`, kept only if they exist in the indexed checkout.

Usage: python evals/build_v1.py data/airflow > evals/airflow_v1.jsonl
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

MODELS = "airflow-core/src/airflow/models/"

QUESTIONS = {
    "1c870b347": "Why would XCom values pushed from a task-end trigger event end up JSON-encoded twice?",
    "79a1dd12e": "Why could dag.test() hang when a task reads a Variable inside a subprocess?",
    "cd3ae408a": "What happens to mapped tasks when I clear a Dag run with 'run on latest version'?",
    "ce01a733e": "Why might the scheduler use a lot of memory when a Dag run has a very large conf?",
    "9e662c23f": "How does the scheduler clean up queued asset event rows and decide whether an asset-triggered Dag is ready?",
    "a211be1c1": "How are human-in-the-loop responses filtered by who responded, and why did that break on MySQL?",
    "bf022cc64": "How does Airflow avoid creating duplicate pending Dag runs for the same asset partition key?",
    "05981d3e2": "Which bundle version is used for task clears and callbacks after only the Dag code changes?",
    "e3110a309": "How does Trigger.clean_unused handle a session passed in by the caller?",
    "3ce8d152d": "How are queued connection tests prevented from overwriting a connection that was updated later?",
    "d79c0b18f": "Why might ORM models fail to configure in a process that never set up the database?",
    "43129ecb6": "When clearing a task with upstream and downstream selected, how are the related tasks chosen?",
    "07a0c4da0": "How is a queued-at deadline interval stored as JSON handled when a Dag run is cleared?",
    "96582adf6": "How does clearing a running Dag run respect the 'run on latest version' option?",
    "6e5f3136b": "Why could running dag.test without a logical date delete other Dag runs?",
    "94a677e8f": "How are callback triggers assigned to triggerers, and how are they kept on healthy ones?",
    "79de1d480": "When a task is retried, is refresh_from_task applied so operator changes take effect?",
    "ae6139d58": "When is try_number initialized relative to computing a task's priority weight?",
    "64943ba6d": "How is run_duration calculated for a Dag run created by airflow dags test?",
    "05519cd77": "How are task attempt UUIDs kept consistent with try numbers?",
    "01fc0c5ff": "What task instance is passed to Dag-level callbacks when the Dag version is missing?",
    "a7b60ec14": "Why could backfill Dag runs never get scheduled?",
    "39a5b937b": "Do single-row XCom lookups limit the query to one row?",
    "132ec4f59": "Does Variable.set change the team of a variable that already exists?",
    "f5ec36107": "What happens when tasks are cleared in a Dag run that has no Dag version?",
    "f03b3bb5e": "How are deadline alerts serialized, and how are old deadlines pruned?",
    "537feafb3": "How are Variable values that are JSON but not strings stored and returned?",
    "dd1f84ab2": "When a deferred task's trigger fires a TaskFailedEvent, are the task's retries respected?",
    "ecd4d951d": "How does the scheduler count finished upstream tasks for trigger rules without repeating that query?",
    "00ce83c6c": "Why might a Dag deadline stop firing after the Dag is edited?",
    "57cff7c40": "How is the data for deadline callbacks persisted?",
    "ff52f44e9": "How are asset events submitted from triggers, and how are N+1 queries avoided there?",
    "7dd84f3ee": "How is a task instance's duration calculated, and what is different on SQLite?",
    "a5ffa6c79": "How are upstream map indexes resolved after a mapped task expands from a placeholder?",
    "04f19d19b": "What does the scheduler do when the last task instance of a run predates Dag versioning?",
    "db04a4ef8": "How do Connections and Variables read values from secrets backends, including older custom backends?",
    "8395c29ff": "How does the scheduler reset orphaned task instances and log them?",
    "597891128": "How does a Callback handle events and emit metrics with tag values?",
    "153623856": "What happens when a backfill is requested with an invalid date range?",
    "76c5131df": "How does XCom look up values from prior dates when the same run_id exists in different Dags?",
    "69b6c54ef": "Are trigger kwargs deserialized when a serialized Dag is loaded?",
    "345ddb8df": "Do deferred task instances count toward max_active_tis_per_dag?",
    "8c43743d3": "Which Dag version do task instances get when cleared with run on latest version?",
    "8b401f28f": "Do backfill params override the conf of a Dag run that already exists?",
    "f1cd3f950": "How does Airflow decide to auto-pause a Dag after consecutive failed runs, and how are the runs ordered?",
    "4ecbd59c0": "How does get_task_instance look up a task instance?",
    "d292e0e9e": "How are trigger kwargs encoded when a trigger is saved to the database?",
    "891c7fbb6": "What happens when a deferred task resumes on a worker running an older version than the server?",
    "860277d7b": "How is a task's priority weight computed for operators that are not serialized?",
    "4b510de4e": "When a Dag bundle refreshes, how are import errors for zipped Dags handled?",
    "9ad468cb9": "How are Dags whose files were deleted marked as inactive?",
    "4eabf7964": "How does a Dag run emit its OpenTelemetry span, and what if there is no context carrier?",
    "a31db1fb0": "How does Airflow stop the same task from running twice when several schedulers are running?",
    "c7eaa64d9": "Which columns of the deadline table are indexed?",
    "447771d5f": "How are callbacks picked for execution so that one Dag bundle can't starve the others?",
    "dd1a69c6f": "How does deserializing deadline alerts stay compatible with older serialized Dags?",
    "3a75c2213": "What run_id does task_instance_mutation_hook see when a TaskInstance is created?",
    "686914171": "How are deadline alerts hashed?",
    "f4f885388": "How is the dependency context handled when checking whether a task can run, including reschedule mode?",
    "9880716dc": "How does the reschedule dependency check find a task's latest reschedule request?",
    "b0462de91": "Can task_instance_mutation_hook access the task's Dag run?",
    "235595b6c": "When writing a serialized Dag, how does Airflow check whether a Dag version is in use without loading every task instance?",
    "dd0eab9dd": "Does the scheduler eagerly load all task instances of a Dag run?",
    "8da26cce4": "How does dags_needing_dagruns load assets without an N+1 query?",
    "97ea9fb1c": "How does the secrets cache handle teams for Connections and Variables?",
    "dfcd04907": "How does the Dag processor deal with MySQL deadlocks when writing assets?",
    "91af86ecb": "How is the audit Log model joined to task instances?",
    "a377033b0": "How is a connection URI built when the extras contain non-string values?",
    "2b38b1946": "Is a Dag run's queued_at updated when its tasks are cleared?",
    "5aa90c0cd": "How are mapped task instances handled when clearing with downstream or upstream?",
    "1d8fa22b1": "What does the scheduler do with an asset Dag run queue entry whose Dag no longer exists?",
    "4dd5ea02a": "In what order does DagRun.fetch_task_instances return task instances?",
    "5398923f4": "How does writing a serialized Dag avoid leaving orphaned Dag versions?",
    "e6448226e": "How are callback metric tags built from enum values?",
    "1ab5bb63f": "What is retry_delay on a mapped operator when it isn't set explicitly?",
    "3e1638250": "How is a task instance's log URL generated from base_url?",
    "31f0eac1e": "How is the operator name of a task instance recorded for custom operators?",
    "33658f0ff": "How does next_retry_datetime handle exponential backoff that would overflow?",
    "05d2229d0": "How is a Dag run's created_dag_version loaded?",
    "79e6fd398": "What does DagRun.dag_versions return when created_dag_version is missing?",
    "ef80507e8": "What context is passed to Dag success and failure callbacks?",
    "2d6d594ee": "How is a Connection parsed from a URI whose host includes a protocol?",
    "75f8aa91c": "Why might dag.test fail when deep-copying Jinja2 templates?",
}

# Identifier-like tokens: snake_case, dotted names, CamelCase with 2+ capitals.
_SYMBOL = re.compile(r"\b(?:[a-z]+_[a-z_]+|\w+\.\w+\(?\)?|[A-Z][a-z]+[A-Z]\w*)\b")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def main(repo: Path) -> None:
    for i, (sha, question) in enumerate(QUESTIONS.items(), 1):
        meta = git(repo, "show", "-s", "--format=%H|%ad|%s", "--date=short", sha).strip()
        full, date, title = meta.split("|", 2)
        files = git(repo, "show", "--name-only", "--format=", sha).split()
        gold = sorted(
            f for f in files
            if f.startswith(MODELS) and f.endswith(".py") and (repo / f).exists()
        )
        if not gold:
            print(f"skip {sha}: no gold file exists at the indexed commit", file=sys.stderr)
            continue
        pr = re.findall(r"\(#(\d+)\)", title)
        print(json.dumps({
            "id": f"v1-{i:03d}",
            "question": question,
            "gold_files": gold,
            "commit": full,
            "pr": int(pr[-1]) if pr else None,
            "date": date,
            "title": title,
            "mentions_symbol": bool(_SYMBOL.search(question)),
        }))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
