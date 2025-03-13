import subprocess

print("Starting self-traing-pipeline")

subprocess.run(["python", "package-installation.py"])

subprocess.run(["python", "interface.py"])
