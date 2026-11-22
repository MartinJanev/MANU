"""GoodBaiBai88/M3D-LaMed-Llama-2-7B: see models/_m3dlamed.py.

Usage: sbatch -J m3dlamed_llama2 jobs/run_model.sh m3dlamed_llama2 --split dev|paper [--controls]
"""
from common import runner
from models._m3dlamed import M3DLaMedAdapter


class M3DLlama2Adapter(M3DLaMedAdapter):
    name = "m3d-lamed-llama2"
    model_id = "GoodBaiBai88/M3D-LaMed-Llama-2-7B"


if __name__ == "__main__":
    runner.main(M3DLlama2Adapter())
