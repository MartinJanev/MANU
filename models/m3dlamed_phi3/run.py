"""GoodBaiBai88/M3D-LaMed-Phi-3-4B: see models/_m3dlamed.py.

Usage: sbatch -J m3dlamed_phi3 jobs/run_model.sh m3dlamed_phi3 --split dev|paper [--controls]
"""
from common import runner
from models._m3dlamed import M3DLaMedAdapter


class M3DPhi3Adapter(M3DLaMedAdapter):
    name = "m3d-lamed-phi3"
    model_id = "GoodBaiBai88/M3D-LaMed-Phi-3-4B"


if __name__ == "__main__":
    runner.main(M3DPhi3Adapter())
