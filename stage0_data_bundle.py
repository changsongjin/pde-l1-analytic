"""Save complete legacy raw observations for an identical-input contract check.

These files are audit fixtures, not the later development/final-test split.
Every consumer receives ALL trajectories. Never reconstruct noise per method.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

import noise_inverse as N
from stage0_evaluation import array_hash

ROOT = Path(__file__).resolve().parent


def save_bundle(path, x, t, observations, metadata):
    path = Path(path)
    arrays = {"x": np.asarray(x), "t": np.asarray(t)}
    arrays.update({f"u_{i}": np.asarray(u) for i, u in enumerate(observations)})
    if not observations or any(u.shape != (len(t), len(x)) for u in observations):
        raise ValueError("all observations must have shape (nt, nx)")
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite raw observations")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    manifest = {**metadata, "trajectory_count": len(observations),
                "arrays": {key: {"shape": list(value.shape), "sha256": array_hash(value)}
                           for key, value in arrays.items()},
                "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def load_bundle(path):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["npz_sha256"]:
        raise ValueError("bundle file hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    if set(arrays) != set(manifest["arrays"]):
        raise ValueError("bundle keys mismatch")
    for name, array in arrays.items():
        if array_hash(array) != manifest["arrays"][name]["sha256"]:
            raise ValueError("array hash mismatch: " + name)
    observations = [arrays[f"u_{i}"] for i in range(manifest["trajectory_count"])]
    return arrays["x"], arrays["t"], observations, manifest


def assert_same_inputs(first, second):
    """Compare manifests actually used by methods, including every trajectory."""
    for key in ("trajectory_count", "arrays"):
        if first[key] != second[key]:
            raise ValueError("methods did not receive identical input: " + key)


def main():
    rows = []
    for case_name, case in N.CASES.items():
        x, t = np.linspace(*case["x"]), np.linspace(*case["t"])
        xx, tt = np.meshgrid(x, t)
        clean = [f(xx.ravel(), tt.ravel()).reshape(xx.shape) for f in case["traj"]]
        rng = np.random.default_rng(20261106)
        observed = [u + 0.01 * np.std(u) * rng.standard_normal(u.shape) for u in clean]
        path = ROOT / "stage0_data" / (case_name + "_audit.npz")
        manifest = save_bundle(path, x, t, observed,
                               {"case": case_name, "sigma": 0.01, "seed": 20261106,
                                "role": "legacy_input_contract_audit_only",
                                "target": case["target"], "source": "noise_inverse.CASES",
                                "formal_split": False})
        _, _, loaded, restored = load_bundle(path)
        assert_same_inputs(manifest, restored)
        assert len(loaded) == len(case["traj"])
        assert all(np.array_equal(a, b) for a, b in zip(observed, loaded))
        rejected_subset = None
        if len(loaded) > 1:
            wrong = {**manifest, "trajectory_count": 1}
            try:
                assert_same_inputs(manifest, wrong)
            except ValueError:
                rejected_subset = True
            assert rejected_subset
        rows.append({"case": case_name, "file": str(path.relative_to(ROOT)),
                     "trajectory_count": len(loaded), "hash": manifest["npz_sha256"],
                     "all_trajectories_roundtrip": True,
                     "first_trajectory_only_rejected": rejected_subset,
                     "size_bytes": path.stat().st_size})
    report = {"stage": 0, "rows": rows, "all_checks_passed": True,
              "scope": "actual persisted common raw arrays; no algorithm benchmark rerun",
              "legacy_native_comparisons": "ineligible until rerun with matching full-input manifests"}
    (ROOT / "stage0_data_bundle_result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"all_checks_passed": True, "files": len(rows),
                      "total_bytes": sum(row["size_bytes"] for row in rows)}))


if __name__ == "__main__":
    main()
