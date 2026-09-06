import requests
from pathlib import Path

patient = Path(
    r"C:\Users\chand\BrainTumorProject\dataset\brats2023\test\BraTS-GLI-00002-000"
)

files = {}

try:
    for modality in ["t1n", "t1c", "t2w", "t2f"]:
        file_path = patient / f"BraTS-GLI-00002-000-{modality}.nii.gz"

        files[modality] = (
            file_path.name,
            open(file_path, "rb"),
            "application/gzip"
        )

    print("=" * 70)
    print("SENDING 4 MRI MODALITIES TO FASTAPI")
    print("=" * 70)

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files=files
    )

    print("\nHTTP STATUS:", response.status_code)
    print("\nAPI RESPONSE:")
    print(response.text)

finally:
    for _, file_object, _ in files.values():
        file_object.close()