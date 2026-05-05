from __future__ import annotations

import __main__
import argparse
import pickle
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd

from src import PYDATA
from src.AUDVIS import Behavior, load_in_data
from src.VisualAreas import Areas


AREA_NAMES = ["V1", "AM/PM", "A/RL/AL", "LM"]
XML_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class LoggedSession:
    group: str
    animal: str
    log_date: str
    log_date_iso: str | None
    session_type: str
    mouse_vid: str
    calcium_note: str
    log_roi_count: int | None
    behavior_note: str
    facemap_note: str
    note: str
    note2: str
    note3: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        default=str(Path(PYDATA) / "decoding_population_sessionwise" / "pre_glm_clean"),
    )
    return parser.parse_args()


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [
        "".join(text_node.text or "" for text_node in si.findall(".//x:t", XML_NS))
        for si in root.findall("x:si", XML_NS)
    ]


def _xlsx_rows(path: str | Path) -> list[dict[str, str]]:
    def col_letter(cell_ref: str) -> str:
        out = []
        for ch in cell_ref:
            if ch.isalpha():
                out.append(ch)
            else:
                break
        return "".join(out)

    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(path) as zf:
        shared_strings = _xlsx_shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        for row in root.findall(".//x:sheetData/x:row", XML_NS):
            values: dict[str, str] = {}
            for cell in row.findall("x:c", XML_NS):
                ref = cell.attrib["r"]
                kind = cell.attrib.get("t")
                value_node = cell.find("x:v", XML_NS)
                if value_node is None:
                    inline = cell.find("x:is", XML_NS)
                    value = (
                        "".join(t.text or "" for t in inline.findall(".//x:t", XML_NS))
                        if inline is not None
                        else ""
                    )
                elif kind == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    value = value_node.text or ""
                values[col_letter(ref)] = value
            rows.append(values)
    return rows


def _parse_date(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d.%m.%Y", "%d.%m.%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    # accept single-digit day/month variants like 2.1.2022
    try:
        day, month, year = raw.split(".")
        if day and month and year:
            return datetime(int(year), int(month), int(day)).strftime("%Y%m%d")
    except Exception:
        return None
    return None


def _extract_leading_int(text: str) -> int | None:
    match = re.search(r"(\d+)", text or "")
    return int(match.group(1)) if match else None


def parse_preprocess_logs() -> pd.DataFrame:
    files = {
        "g1": Path("lab_books & records/g1-preprocess-logs.xlsx"),
        "g2": Path("lab_books & records/g2-preprocess-logs.xlsx"),
    }
    records: list[LoggedSession] = []

    for group, path in files.items():
        current_animal = ""
        current_date = ""
        for row in _xlsx_rows(path):
            if row.get("A") == "animal":
                continue
            animal = row.get("A", "").strip()
            date_raw = row.get("B", "").strip()
            session_type = row.get("C", "").strip()
            if animal:
                current_animal = animal
            if date_raw:
                current_date = date_raw
            if not session_type.startswith("Bar_Tone_LR"):
                continue
            calcium = row.get("E", "").strip()
            records.append(
                LoggedSession(
                    group=group,
                    animal=current_animal,
                    log_date=current_date,
                    log_date_iso=_parse_date(current_date),
                    session_type=session_type,
                    mouse_vid=row.get("D", "").strip(),
                    calcium_note=calcium,
                    log_roi_count=_extract_leading_int(calcium),
                    behavior_note=row.get("F", "").strip(),
                    facemap_note=row.get("G", "").strip(),
                    note=row.get("H", "").strip(),
                    note2=row.get("I", "").strip(),
                    note3=row.get("J", "").strip(),
                )
            )

    return pd.DataFrame([vars(record) for record in records])


def load_sessions_overview() -> pd.DataFrame:
    with Path(PYDATA, "SessionsOverview.pkl").open("rb") as handle:
        overview = pickle.load(handle)

    pattern = re.compile(
        r".*/(?P<group>g[12])/(?P<animal>[^/]+)/(?P<date>\d{8})/(?P<session_type>Bar_Tone_LR2?|Bar_Tone_LR)/(?:[^/]+)$"
    )
    rows = []
    for group, sessions in overview.items():
        for key, path in sessions.items():
            match = pattern.match(path)
            if not match:
                continue
            key_tail = key.split("_")
            key_neurons = None
            if key_tail and key_tail[-1].isdigit():
                key_neurons = int(key_tail[-1])
            rows.append(
                {
                    "group": group,
                    "animal": match.group("animal"),
                    "date": match.group("date"),
                    "session_type": match.group("session_type"),
                    "available_path": path,
                    "overview_key": key,
                    "overview_count_hint": key_neurons,
                }
            )
    return pd.DataFrame(rows)


def assign_pre_post(available: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, animal), animal_df in available.groupby(["group", "animal"], sort=False):
        dates = sorted(animal_df["date"].unique())
        pre_date = dates[0]
        post_date = dates[-1]
        for _, row in animal_df.iterrows():
            selection = "pre" if row["date"] == pre_date else ""
            if len(dates) > 1 and row["date"] == post_date:
                selection = "post" if row["date"] != pre_date else "pre"
            if len(dates) > 2 and pre_date < row["date"] < post_date:
                selection = "intermediate"
            rows.append(
                {
                    **row.to_dict(),
                    "pre_date": pre_date,
                    "post_date": post_date if len(dates) > 1 else "",
                    "grouping_selection": selection,
                }
            )
    return pd.DataFrame(rows)


def load_current_pre_sessions(area_names: Iterable[str]) -> pd.DataFrame:
    __main__.Behavior = Behavior
    area_names = list(area_names)
    records = []
    for av in load_in_data(pre_post="pre"):
        areas = Areas(av, get_indices=True)
        for session_index in sorted(av.sessions):
            session = av.sessions[session_index]
            start, stop = av.session_neurons[session_index]
            session_name = session["session"]
            animal = session_name.split("_")[0]
            area_counts = {}
            for area_name in area_names:
                if area_name not in areas.area_indices:
                    area_counts[area_name] = 0
                    continue
                count = int(
                    len(
                        set(range(start, stop)).intersection(
                            set(areas.area_indices[area_name].tolist())
                        )
                    )
                )
                area_counts[area_name] = count

            records.append(
                {
                    "group": av.NAME[:2],
                    "loaded_group_name": av.NAME,
                    "animal": animal,
                    "session_name": session_name,
                    "total_n_neurons": int(session["n_neurons"]),
                    **{f"{area}_n": area_counts[area] for area in area_names},
                    "areas_used_in_decoding": ",".join(
                        area for area in area_names if area_counts[area] >= 2
                    ),
                }
            )
    return pd.DataFrame(records)


def resolve_log_rows(logs: pd.DataFrame, available: pd.DataFrame, current_pre: pd.DataFrame) -> pd.DataFrame:
    available = available.copy()
    current_pre = current_pre.copy()
    out_rows = []

    for _, row in logs.iterrows():
        candidates = available.loc[
            (available["group"] == row["group"])
            & (available["animal"] == row["animal"])
            & (available["session_type"] == row["session_type"])
        ].copy()

        exact = candidates.loc[candidates["date"] == row["log_date_iso"]]
        if not exact.empty:
            chosen = exact
        else:
            roi = row["log_roi_count"]
            by_roi = candidates.loc[candidates["overview_count_hint"] == roi] if roi is not None else candidates.iloc[0:0]
            if not by_roi.empty:
                chosen = by_roi
            else:
                daymonth = row["log_date_iso"][4:] if row["log_date_iso"] else ""
                by_daymonth = candidates.loc[candidates["date"].str[4:] == daymonth] if daymonth else candidates.iloc[0:0]
                chosen = by_daymonth if not by_daymonth.empty else candidates.iloc[0:0]

        available_row = chosen.iloc[0].to_dict() if not chosen.empty else {}
        resolved_date = available_row.get("date", row["log_date_iso"])
        selection = available_row.get("grouping_selection", "")

        pre_candidates = current_pre.loc[
            (current_pre["group"] == row["group"])
            & (current_pre["animal"] == row["animal"])
        ].copy()
        if resolved_date:
            pre_candidates = pre_candidates.loc[
                pre_candidates["session_name"].str.contains(resolved_date, na=False)
                | (pre_candidates["total_n_neurons"] == row["log_roi_count"])
            ]
        elif row["log_roi_count"] is not None:
            pre_candidates = pre_candidates.loc[pre_candidates["total_n_neurons"] == row["log_roi_count"]]

        if len(pre_candidates) > 1 and row["log_roi_count"] is not None:
            pre_candidates = pre_candidates.loc[pre_candidates["total_n_neurons"] == row["log_roi_count"]]

        loaded = pre_candidates.iloc[0].to_dict() if not pre_candidates.empty else {}

        if available_row:
            if selection == "pre":
                if available_row["pre_date"] == available_row["post_date"] or not available_row["post_date"]:
                    reason = "included: only available date for this animal among sessions with _SPSIG_Res"
                elif available_row["session_type"] == "Bar_Tone_LR2":
                    reason = "included: additional recording on the animal's earliest available date"
                else:
                    reason = "included: earliest available date for this animal becomes pre"
            elif selection == "post":
                reason = "excluded from current pre: latest available date for this animal is assigned to post"
            elif selection == "intermediate":
                reason = "excluded from current pre: intermediate available date; grouping keeps only earliest as pre and latest as post"
            else:
                reason = "available but not selected by current grouping logic"
        else:
            reason = "excluded before grouping: session is not present in SessionsOverview, so no usable _SPSIG_Res export was found"

        out_rows.append(
            {
                **row.to_dict(),
                "resolved_date": resolved_date,
                "available_in_sessions_overview": bool(available_row),
                "available_path": available_row.get("available_path", ""),
                "grouping_selection": selection,
                "pre_date_for_animal": available_row.get("pre_date", ""),
                "post_date_for_animal": available_row.get("post_date", ""),
                "loaded_pre_session_name": loaded.get("session_name", ""),
                "loaded_pre_total_n_neurons": loaded.get("total_n_neurons", pd.NA),
                **{
                    f"loaded_pre_{area}_n": loaded.get(f"{area}_n", pd.NA)
                    for area in AREA_NAMES
                },
                "loaded_pre_areas_used_in_decoding": loaded.get("areas_used_in_decoding", ""),
                "current_pre_status": reason,
            }
        )

    return pd.DataFrame(out_rows)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logs = parse_preprocess_logs()
    available = assign_pre_post(load_sessions_overview())
    current_pre = load_current_pre_sessions(AREA_NAMES)
    audit = resolve_log_rows(logs, available, current_pre)
    audit = audit.sort_values(["group", "animal", "resolved_date", "session_type"], na_position="last")

    audit.to_csv(save_dir / "session_inclusion_audit.csv", index=False)


if __name__ == "__main__":
    main()
