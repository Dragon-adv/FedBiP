import os
import shutil
import subprocess
import sys
from pathlib import Path


TEST_OUTPUT_DIR = Path("test_exps")


def run_cmd(args, env, stage_name: str):
    print(f"\n=== [{stage_name}] Running ===")
    print(" ".join(args))
    subprocess.run(args, env=env, check=True)


def assert_exists(path: Path, stage_name: str):
    if not path.exists():
        raise FileNotFoundError(f"[{stage_name}] Missing expected output: {path}")


def assert_has_any_images(root: Path, stage_name: str):
    if not root.exists():
        raise FileNotFoundError(f"[{stage_name}] Missing generated_images folder: {root}")
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            return p
    raise FileNotFoundError(
        f"[{stage_name}] No images found under: {root}\n"
        f"Please check Stage 2 parameters (domain/test_type/start_idx/end_idx) and artifacts from Stage 1."
    )


def main():
    # Preflight: make sure we're running under the correct Python environment.
    # (verify_pipeline uses sys.executable to spawn subprocesses, so this matters.)
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "当前 Python 环境缺少 torch，无法进行冒烟测试。\n"
            "请先激活包含依赖的环境（例如 `conda activate fedbip`），然后再运行：\n"
            "  python verify_pipeline.py\n"
            f"当前解释器：{sys.executable}\n"
        ) from e

    # 1) Simulate single-GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Enable dummy dataset fallback for smoke tests when real datasets are not prepared.
    os.environ["FEDBIP_SMOKE_DUMMY"] = "1"
    env = os.environ.copy()

    # Keep it small & deterministic
    dataset = env.get("FEDBIP_DATASET", "dermamnist")
    domain = env.get("FEDBIP_DOMAIN", "client_0")
    train_type = env.get("FEDBIP_TRAIN_TYPE", "prompt")
    test_type = env.get("FEDBIP_TEST_TYPE", "syn_wnoise_0.1_interpolated")

    out_dir = TEST_OUTPUT_DIR / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: Train (prompt vectors + mean/std) ----
    stage1_args = [
        sys.executable,
        "train.py",
        "--dataset",
        dataset,
        "--domain",
        domain,
        "--train_type",
        train_type,
        "--output_dir",
        str(out_dir),
        "--mixed_precision",
        "bf16",
        "--train_batch_size",
        "1",
        "--client_num",
        "1",
        "--num_shot",
        "1",
        "--num_train_epochs",
        "1",
        "--max_train_steps",
        "2",
        "--save_steps",
        "1",
        "--num_workers",
        "0",
        "--skip_evaluation",
    ]
    run_cmd(stage1_args, env, "Stage 1 (train.py)")

    # Expect at least client 0 artifacts
    assert_exists(out_dir / "mean_0.pt", "Stage 1")
    assert_exists(out_dir / "std_0.pt", "Stage 1")
    assert_exists(out_dir / "prompt_domain_0.pth", "Stage 1")
    assert_exists(out_dir / "prompt_class_0.pth", "Stage 1")

    # ---- Stage 2: Generate (at least one image) ----
    stage2_args = [
        sys.executable,
        "generate.py",
        "--dataset",
        dataset,
        "--domain",
        domain,
        "--output_dir",
        str(out_dir),
        "--mixed_precision",
        "bf16",
        "--test_type",
        test_type,
        "--num_inference_steps",
        "5",
        "--start_idx",
        "0",
        "--end_idx",
        "0",
        "--num_test_samples",
        "1",
    ]
    run_cmd(stage2_args, env, "Stage 2 (generate.py)")

    images_root = out_dir / "generated_images"
    first_img = assert_has_any_images(images_root, "Stage 2")
    print(f"[Stage 2] Found generated image: {first_img}")

    # ---- Stage 3: Classifier train (one epoch, synthetic images) ----
    # We set train_type to include "syn" so clf_train.py switches to generated_images loader.
    stage3_train_type = f"train_{test_type}_1"
    stage3_args = [
        sys.executable,
        "clf_train.py",
        "--dataset",
        dataset,
        "--output_dir",
        str(out_dir),
        "--train_type",
        stage3_train_type,
        "--test_type",
        test_type,
        "--train_batch_size",
        "1",
        "--num_workers",
        "0",
        "--num_epochs",
        "1",
        "--skip_evaluation",
    ]
    run_cmd(stage3_args, env, "Stage 3 (clf_train.py)")

    print("\n=== verify_pipeline.py PASS ===")
    print(f"All outputs are under: {out_dir}")

    # ---- Cleanup prompt ----
    try:
        ans = input(f"\nDelete test directory '{TEST_OUTPUT_DIR}'? [y/N]: ").strip().lower()
    except EOFError:
        ans = ""
    if ans in {"y", "yes"}:
        shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)
        print("Deleted.")
    else:
        print("Kept.")


if __name__ == "__main__":
    main()


