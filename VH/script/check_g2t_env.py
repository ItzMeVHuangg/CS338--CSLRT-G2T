import importlib
import importlib.util
import os


os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


REQUIRED = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("yaml", "PyYAML"),
    ("numpy", "numpy"),
    ("tqdm", "tqdm"),
    ("editdistance", "editdistance"),
    ("sacrebleu", "sacrebleu"),
    ("rouge_score", "rouge-score"),
    ("nltk", "nltk"),
    ("PIL", "pillow"),
    ("transformers", "transformers"),
    ("sentencepiece", "sentencepiece"),
]


def main():
    missing = []
    for module_name, package_name in REQUIRED:
        try:
            if module_name == "nltk":
                if importlib.util.find_spec(module_name) is None:
                    raise ImportError("module spec not found")
            else:
                importlib.import_module(module_name)
            print(f"[OK] {package_name}")
        except Exception as exc:
            missing.append(package_name)
            print(f"[MISS] {package_name}: {exc}")

    try:
        import torch
        print(f"[Torch] version={torch.__version__}")
        print(f"[Torch] cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Torch] gpu={torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    if missing:
        print("\nInstall missing non-torch packages with:")
        print("  pip install -r requirements_g2t_slt.txt")
        raise SystemExit(1)

    print("\nEnvironment looks ready for G2T SLT.")


if __name__ == "__main__":
    main()
