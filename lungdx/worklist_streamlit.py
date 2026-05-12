"""Рабочая очередь (mutates st.session_state)."""

from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

from lungdx.clinical import priority_label, priority_rank, sla_by_priority, sla_minutes


def enqueue_worklist(rows: list[dict]) -> None:
    existing = {x["StudyID"] for x in st.session_state["worklist"]}
    for row in rows:
        if row["StudyID"] not in existing:
            p = row.get("Приоритет", "Планово")
            st.session_state["worklist"].append(
                {
                    "StudyID": row["StudyID"],
                    "Файл": row["Файл"],
                    "Приоритет": p,
                    "Целевой SLA": sla_by_priority(p),
                    "Диагноз": row["Диагноз"],
                    "Уверенность, %": row["Уверенность, %"],
                    "Статус": "В очереди",
                    "Время поступления": datetime.now().isoformat(timespec="seconds"),
                    "Дедлайн": (datetime.now() + timedelta(minutes=sla_minutes(p))).isoformat(
                        timespec="seconds"
                    ),
                }
            )


def worklist_view_rows() -> list[dict]:
    now = datetime.now()
    view = []
    for row in st.session_state["worklist"]:
        deadline = datetime.fromisoformat(row["Дедлайн"])
        overdue = now > deadline and row["Статус"] != "Завершено"
        view.append(
            {
                **row,
                "Приоритет (UI)": priority_label(row["Приоритет"]),
                "SLA статус": "⚠️ Просрочено" if overdue else "✅ В норме",
            }
        )
    return sorted(view, key=lambda x: (priority_rank(x["Приоритет"]), x["Статус"] != "В очереди"))
