"""Aggregate measured statistics over ingested manifests (audit/report tooling).

Reads datasets/manifests/*.json plus their datasets/reports/*_validation_*.json
counterparts and emits one JSON + one Markdown summary with the numbers the
dataset audit needs: counts per class/split, duration totals and mean/median,
FPS and resolution distributions, corruption/duplicate counts, duplicate
content groups, and temporal-window capacity at the recognizer's window sizes.

Nothing here decodes video: every figure comes from metadata already probed at
ingest time, so re-running this is cheap.

Usage:
    python scripts/summarize_datasets.py
    python scripts/summarize_datasets.py --windows 16 32 --out datasets/reports/_audit_summary.json
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from suraksha.data.sequences import count_sequences  # noqa: E402

DEFAULT_STRIDE = 8


def _dist(values, fmt=str) -> dict[str, int]:
    c = Counter(fmt(v) for v in values if v is not None)
    return {k: c[k] for k in sorted(c, key=lambda x: (-c[x], x))}


def _num_summary(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "total": round(sum(vals), 2),
        "mean": round(statistics.fmean(vals), 2),
        "median": round(statistics.median(vals), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "missing": len(values) - len(vals),
    }


def load_validation(reports_dir: Path, manifest_name: str) -> dict[str, dict]:
    """clip_id -> probe record, from the newest validation report for a manifest."""
    cands = sorted(reports_dir.glob(f"{manifest_name}_validation_*.json"))
    if not cands:
        return {}
    rep = json.loads(cands[-1].read_text(encoding="utf-8"))
    return {e["clip_id"]: e for e in rep.get("entries", [])}


def summarize_manifest(mpath: Path, reports_dir: Path, windows: list[int]) -> dict:
    m = json.loads(mpath.read_text(encoding="utf-8"))
    entries = m.get("entries", [])
    probes = load_validation(reports_dir, m.get("name", mpath.stem))

    labels, splits, statuses = Counter(), Counter(), Counter()
    label_by_split: dict[str, Counter] = defaultdict(Counter)
    durations, fpss, frame_counts = [], [], []
    durations_by_label: dict[str, list[float]] = defaultdict(list)
    resolutions, fourccs, exts = [], [], []
    corrupt_frames_total = 0
    problem_files: list[dict] = []
    sha_groups: dict[str, list[str]] = defaultdict(list)
    win_totals = {str(w): 0 for w in windows}
    win_by_label = {str(w): Counter() for w in windows}
    short_for_window = {str(w): 0 for w in windows}
    unknown_meta = 0

    for e in entries:
        label = e.get("label") or "UNKNOWN"
        split = e.get("split") or "unassigned"
        labels[label] += 1
        splits[split] += 1
        label_by_split[split][label] += 1

        p = probes.get(e["clip_id"], {})
        meta = p.get("meta") or {}
        status = p.get("status") or e.get("status") or "UNPROBED"
        statuses[status] += 1
        corrupt_frames_total += int(meta.get("corrupt_frames") or 0)
        if status in ("CORRUPTED", "MISSING", "UNSUPPORTED"):
            problem_files.append({
                "clip_id": e["clip_id"], "status": status,
                "details": p.get("details", ""), "path": e.get("path", ""),
                "label": label, "split": split,
                "source_class": (e.get("notes") or {}).get("source_class"),
            })

        dur = e.get("duration_sec", meta.get("duration_sec"))
        fps = e.get("fps", meta.get("fps"))
        fc = e.get("frame_count", meta.get("frame_count"))
        w, h = e.get("width", meta.get("width")), e.get("height", meta.get("height"))
        durations.append(dur)
        if dur is not None:
            durations_by_label[label].append(dur)
        fpss.append(fps)
        frame_counts.append(fc)
        if w and h:
            resolutions.append(f"{w}x{h}")
        if meta.get("fourcc"):
            fourccs.append(meta["fourcc"])
        exts.append(Path(e.get("path", "")).suffix.lower() or "?")
        if dur is None or fps is None or fc is None or not (w and h):
            unknown_meta += 1
        if e.get("sha256"):
            sha_groups[e["sha256"]].append(e["clip_id"])

        if fc is not None:
            for win in windows:
                n = count_sequences(fc, win, DEFAULT_STRIDE)
                win_totals[str(win)] += n
                win_by_label[str(win)][label] += n
                if fc < win:
                    short_for_window[str(win)] += 1

    dup_groups = {k[:12]: v for k, v in sorted(sha_groups.items()) if len(v) > 1}
    total_dur = sum(v for v in durations if v is not None)
    ratio = {}
    if labels.get("FIGHT"):
        ratio["nonfight_per_fight"] = round(labels["NON_FIGHT"] / labels["FIGHT"], 3)
        ratio["fight_share"] = round(labels["FIGHT"] / max(sum(labels.values()), 1), 3)

    return {
        "manifest": m.get("name", mpath.stem),
        "group": m.get("group"),
        "task": m.get("task"),
        "license": m.get("license"),
        "created_at": m.get("created_at"),
        "datasets": sorted({e.get("dataset_name") for e in entries if e.get("dataset_name")}),
        "videos": len(entries),
        "labels": dict(labels),
        "label_ratio": ratio,
        "splits": dict(splits),
        "labels_by_split": {k: dict(v) for k, v in sorted(label_by_split.items())},
        "validation_statuses": dict(statuses),
        "corrupt_frames_total": corrupt_frames_total,
        "problem_files": problem_files,
        "duration_sec": _num_summary(durations),
        "duration_by_label_sec": {k: _num_summary(v) for k, v in sorted(durations_by_label.items())},
        "total_duration_hours": round(total_dur / 3600.0, 3),
        "fps": _num_summary(fpss),
        "fps_distribution": _dist(fpss, lambda v: f"{round(v, 2):g}"),
        "frame_count": _num_summary([float(v) if v is not None else None for v in frame_counts]),
        "resolution_distribution": _dist(resolutions),
        "distinct_resolutions": len(set(resolutions)),
        "fourcc_distribution": _dist(fourccs),
        "container_distribution": _dist(exts),
        "entries_with_unknown_metadata": unknown_meta,
        "duplicate_content_groups": len(dup_groups),
        "duplicate_content_files": sum(len(v) for v in dup_groups.values()),
        "duplicates": dup_groups,
        "temporal_windows": {
            "stride": DEFAULT_STRIDE,
            "totals": win_totals,
            "by_label": {k: dict(v) for k, v in win_by_label.items()},
            "mean_per_video": {k: round(v / len(entries), 2) if entries else 0 for k, v in win_totals.items()},
            "videos_shorter_than_window": short_for_window,
        },
        "source_classes": _dist([
            (e.get("notes") or {}).get("source_class") for e in entries
        ], lambda v: v or "?"),
        "proposed_labels": _dist([
            (e.get("notes") or {}).get("proposed_label") for e in entries
        ], lambda v: v or "none"),
        "provenance": next(
            ((e.get("notes") or {}).get("provenance") for e in entries
             if (e.get("notes") or {}).get("provenance")), None),
        "sources": _dist([e.get("source") for e in entries], lambda v: v or "?"),
    }


def render_markdown(summaries: list[dict]) -> str:
    out = [f"# Dataset summary — measured at ingest ({date.today().isoformat()})", ""]
    out += ["## Per-manifest overview", "",
            "| Manifest | Group | Videos | FIGHT | NON_FIGHT | UNKNOWN | Train | Val | Test | VALID | DUP | CORR | Hours |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        st = s["validation_statuses"]
        sp = s["splits"]
        out.append(
            f"| {s['manifest']} | {s['group'] or '-'} | {s['videos']} "
            f"| {s['labels'].get('FIGHT', 0)} | {s['labels'].get('NON_FIGHT', 0)} "
            f"| {s['labels'].get('UNKNOWN', 0)} "
            f"| {sp.get('train', 0)} | {sp.get('val', 0)} | {sp.get('test', 0)} "
            f"| {st.get('VALID', 0)} | {st.get('DUPLICATE', 0)} | {st.get('CORRUPTED', 0)} "
            f"| {s['total_duration_hours']} |")
    out += ["", "## Duration / FPS / frames", "",
            "| Manifest | dur total s | dur mean | dur median | dur min | dur max | fps min | fps max | fps mean | frames median |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        d, f, fc = s["duration_sec"], s["fps"], s["frame_count"]
        out.append(
            f"| {s['manifest']} | {d.get('total', 0)} | {d.get('mean', '-')} | {d.get('median', '-')} "
            f"| {d.get('min', '-')} | {d.get('max', '-')} | {f.get('min', '-')} | {f.get('max', '-')} "
            f"| {f.get('mean', '-')} | {fc.get('median', '-')} |")
    out += ["", "## FPS distribution (distinct values)", ""]
    for s in summaries:
        fps = s["fps_distribution"]
        items = ", ".join(f"{k} fps×{v}" for k, v in list(fps.items())[:12])
        out.append(f"- **{s['manifest']}** — {len(fps)} distinct: {items}")
    out += ["", "## Resolution distribution", ""]
    for s in summaries:
        res = s["resolution_distribution"]
        items = ", ".join(f"{k}×{v}" for k, v in list(res.items())[:10])
        out.append(f"- **{s['manifest']}** — {s['distinct_resolutions']} distinct: {items}")
    out += ["", "## Temporal window capacity (stride 8, computed — nothing extracted)", "",
            "| Manifest | windows @16 | windows @32 | mean/video @32 | videos <32 frames | videos <16 frames |",
            "|---|---|---|---|---|---|"]
    for s in summaries:
        tw = s["temporal_windows"]
        out.append(
            f"| {s['manifest']} | {tw['totals'].get('16', 0)} | {tw['totals'].get('32', 0)} "
            f"| {tw['mean_per_video'].get('32', 0)} "
            f"| {tw['videos_shorter_than_window'].get('32', 0)} "
            f"| {tw['videos_shorter_than_window'].get('16', 0)} |")
    out += ["", "## Duplicate content (reported, never deleted)", ""]
    for s in summaries:
        if s["duplicate_content_groups"]:
            out.append(f"- **{s['manifest']}** — {s['duplicate_content_groups']} group(s), "
                       f"{s['duplicate_content_files']} files:")
            shown = list(s["duplicates"].items())[:25]
            for sha, ids in shown:
                out.append(f"  - `{sha}…` → {', '.join(ids)}")
            if len(s["duplicates"]) > len(shown):
                out.append(f"  - … {len(s['duplicates']) - len(shown)} more group(s); "
                           f"full list in `datasets/reports/_audit_summary.json`")
        else:
            out.append(f"- **{s['manifest']}** — none")
    out += ["", "## Corrupted / missing / unsupported (reported, never deleted)", ""]
    flagged = [s for s in summaries if s["problem_files"]]
    if not flagged:
        out.append("- none in any manifest")
    for s in flagged:
        out.append(f"- **{s['manifest']}** — {len(s['problem_files'])} file(s):")
        for p in s["problem_files"]:
            out.append(f"  - `{p['clip_id']}` {p['status']} — {p['details'] or 'no detail'} "
                       f"· class `{p['source_class']}` · split `{p['split']}` · `{p['path']}`")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", default=str(PROJECT_ROOT / "datasets/manifests"))
    ap.add_argument("--reports-dir", default=str(PROJECT_ROOT / "datasets/reports"))
    ap.add_argument("--windows", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--exclude", nargs="*", default=["_demo*", "example_*"],
                    help="manifest name globs to skip (tooling demos, not real data)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these manifest names")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "datasets/reports/_audit_summary.json"))
    ap.add_argument("--md", default=str(PROJECT_ROOT / "datasets/reports/_audit_summary.md"))
    a = ap.parse_args()

    mdir, rdir = Path(a.manifests_dir), Path(a.reports_dir)
    paths = sorted(p for p in mdir.glob("*.json") if not p.name.startswith("_audit"))
    if a.only:
        paths = [p for p in paths if p.stem in a.only]
    else:
        paths = [p for p in paths
                 if not any(fnmatch.fnmatch(p.stem, g) for g in a.exclude)]

    summaries = [summarize_manifest(p, rdir, a.windows) for p in paths]
    by_group: dict[str, dict] = defaultdict(lambda: {"manifests": 0, "videos": 0, "hours": 0.0,
                                                      "labels": Counter(), "statuses": Counter()})
    for s in summaries:
        g = by_group[s["group"] or "?"]
        g["manifests"] += 1
        g["videos"] += s["videos"]
        g["hours"] += s["total_duration_hours"]
        g["labels"].update(s["labels"])
        g["statuses"].update(s["validation_statuses"])
    totals = {
        "manifests": len(summaries),
        "videos": sum(s["videos"] for s in summaries),
        "total_duration_hours": round(sum(s["total_duration_hours"] for s in summaries), 3),
        "labels": dict(sum((Counter(s["labels"]) for s in summaries), Counter())),
        "validation_statuses": dict(sum((Counter(s["validation_statuses"]) for s in summaries), Counter())),
        "duplicate_content_groups": sum(s["duplicate_content_groups"] for s in summaries),
        "by_group": {k: {**v, "hours": round(v["hours"], 3),
                         "labels": dict(v["labels"]), "statuses": dict(v["statuses"])}
                     for k, v in sorted(by_group.items())},
    }
    payload = {"generated_at": date.today().isoformat(), "totals": totals, "datasets": summaries}

    Path(a.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(a.md).write_text(render_markdown(summaries), encoding="utf-8")
    print(f"Wrote {a.out}")
    print(f"Wrote {a.md}")
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
