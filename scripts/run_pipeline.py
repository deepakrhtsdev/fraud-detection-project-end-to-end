import subprocess

def run_script(script_name):
    print(f"Running {script_name}........")
    result = subprocess.run(["python3", f"scripts/{script_name}"])
    if result.returncode == 0:
        print(f"{script_name} completed successfully. \n")

    else:
        print(f"{script_name} failed!!!!")

if __name__ == "__main__":
    scripts = ["preprocessing.py","analysis.py","modelling.py"]
    for script in scripts:
        run_script(script)