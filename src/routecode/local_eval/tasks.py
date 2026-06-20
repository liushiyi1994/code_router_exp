from __future__ import annotations

from routecode.local_eval.generation_runner import LocalEvalTask


SMOKE_TASKS = [
    LocalEvalTask(
        query_id="gsm8k_smoke_000",
        query_text="A box has 18 red marbles and 24 blue marbles. How many marbles are in the box?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="42",
    ),
    LocalEvalTask(
        query_id="gsm8k_smoke_001",
        query_text="Mia read 12 pages on Monday and 15 pages on Tuesday. How many pages did she read total?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="27",
    ),
    LocalEvalTask(
        query_id="gsm8k_smoke_002",
        query_text="A train travels 30 miles in the first hour and 45 miles in the second hour. What is the total distance?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="75",
    ),
    LocalEvalTask(
        query_id="gsm8k_smoke_003",
        query_text="There are 9 bags with 6 apples each. How many apples are there?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="54",
    ),
    LocalEvalTask(
        query_id="gsm8k_smoke_004",
        query_text="Sam had 100 dollars and spent 37 dollars. How many dollars remain?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="63",
    ),
    LocalEvalTask(
        query_id="gsm8k_smoke_005",
        query_text="A baker made 8 trays with 7 cookies on each tray. How many cookies did the baker make?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="56",
    ),
    LocalEvalTask(
        query_id="gsm8k_smoke_006",
        query_text="Nora has 45 stickers and gives 18 to a friend. How many stickers does Nora have left?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="27",
    ),
    LocalEvalTask(
        query_id="gsm8k_smoke_007",
        query_text="A class has 6 groups with 5 students in each group. How many students are in the groups?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="30",
    ),
    LocalEvalTask(
        query_id="gsm8k_smoke_008",
        query_text="Leo buys 3 notebooks that cost 4 dollars each. What is the total cost?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="12",
    ),
    LocalEvalTask(
        query_id="gsm8k_smoke_009",
        query_text="A rope is 90 meters long. It is cut into 3 equal pieces. How long is each piece?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="30",
    ),
    LocalEvalTask(
        query_id="mmlu_smoke_000",
        query_text="Which gas do plants primarily absorb for photosynthesis?",
        dataset="mmlu_smoke",
        domain="broad_knowledge",
        task_type="multiple_choice",
        gold_answer="C",
        choices=["A. Oxygen", "B. Nitrogen", "C. Carbon dioxide", "D. Helium"],
    ),
    LocalEvalTask(
        query_id="mmlu_smoke_001",
        query_text="What is the capital city of France?",
        dataset="mmlu_smoke",
        domain="broad_knowledge",
        task_type="multiple_choice",
        gold_answer="B",
        choices=["A. Madrid", "B. Paris", "C. Rome", "D. Berlin"],
    ),
    LocalEvalTask(
        query_id="mmlu_smoke_002",
        query_text="Which number is prime?",
        dataset="mmlu_smoke",
        domain="broad_knowledge",
        task_type="multiple_choice",
        gold_answer="D",
        choices=["A. 21", "B. 27", "C. 33", "D. 37"],
    ),
    LocalEvalTask(
        query_id="mmlu_smoke_003",
        query_text="Which organ pumps blood through the human body?",
        dataset="mmlu_smoke",
        domain="broad_knowledge",
        task_type="multiple_choice",
        gold_answer="A",
        choices=["A. Heart", "B. Lung", "C. Kidney", "D. Stomach"],
    ),
    LocalEvalTask(
        query_id="mmlu_smoke_004",
        query_text="Which planet is known as the Red Planet?",
        dataset="mmlu_smoke",
        domain="broad_knowledge",
        task_type="multiple_choice",
        gold_answer="A",
        choices=["A. Mars", "B. Venus", "C. Jupiter", "D. Mercury"],
    ),
    LocalEvalTask(
        query_id="mmlu_smoke_005",
        query_text="Which process turns liquid water into water vapor?",
        dataset="mmlu_smoke",
        domain="broad_knowledge",
        task_type="multiple_choice",
        gold_answer="C",
        choices=["A. Freezing", "B. Melting", "C. Evaporation", "D. Condensation"],
    ),
    LocalEvalTask(
        query_id="mmlu_smoke_006",
        query_text="Which shape has three sides?",
        dataset="mmlu_smoke",
        domain="broad_knowledge",
        task_type="multiple_choice",
        gold_answer="B",
        choices=["A. Square", "B. Triangle", "C. Pentagon", "D. Circle"],
    ),
    LocalEvalTask(
        query_id="mmlu_smoke_007",
        query_text="Which unit is commonly used to measure electric current?",
        dataset="mmlu_smoke",
        domain="broad_knowledge",
        task_type="multiple_choice",
        gold_answer="D",
        choices=["A. Meter", "B. Gram", "C. Liter", "D. Ampere"],
    ),
    LocalEvalTask(
        query_id="mmlu_smoke_008",
        query_text="Which language is primarily used for styling web pages?",
        dataset="mmlu_smoke",
        domain="broad_knowledge",
        task_type="multiple_choice",
        gold_answer="A",
        choices=["A. CSS", "B. SQL", "C. Bash", "D. JSON"],
    ),
    LocalEvalTask(
        query_id="mmlu_smoke_009",
        query_text="Which layer of Earth is liquid and lies below the mantle?",
        dataset="mmlu_smoke",
        domain="broad_knowledge",
        task_type="multiple_choice",
        gold_answer="C",
        choices=["A. Crust", "B. Lithosphere", "C. Outer core", "D. Inner core"],
    ),
]


def load_smoke_tasks(datasets: list[str] | None = None, max_queries: int | None = None) -> list[LocalEvalTask]:
    selected = SMOKE_TASKS
    if datasets:
        allowed = {str(dataset) for dataset in datasets}
        selected = _round_robin_by_dataset([task for task in selected if task.dataset in allowed], list(datasets))
    if max_queries is not None:
        selected = selected[: max(0, int(max_queries))]
    return list(selected)


def _round_robin_by_dataset(tasks: list[LocalEvalTask], dataset_order: list[str]) -> list[LocalEvalTask]:
    grouped = {dataset: [task for task in tasks if task.dataset == dataset] for dataset in dataset_order}
    ordered: list[LocalEvalTask] = []
    offset = 0
    while True:
        added = False
        for dataset in dataset_order:
            bucket = grouped.get(dataset, [])
            if offset < len(bucket):
                ordered.append(bucket[offset])
                added = True
        if not added:
            return ordered
        offset += 1
