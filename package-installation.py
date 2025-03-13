import sys, subprocess, pkg_resources

# required librieries
required = {'opencv-python', 'pillow', 'ultralytics', 'pyyaml','tk'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
