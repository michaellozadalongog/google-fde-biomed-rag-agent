import runpy, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
runpy.run_module("biomed_rag", run_name="__main__")
