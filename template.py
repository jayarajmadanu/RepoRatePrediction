import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/repoRatePred/__init__.py",
    f"src/repoRatePred/components/__init__.py",
    f"src/repoRatePred/utils/__init__.py",
    f"src/repoRatePred/utils/common.py",
    f"src/repoRatePred/config/__init__.py",
    f"src/repoRatePred/config/configuration.py",
    f"src/repoRatePred/pipeline/__init__.py",
    f"src/repoRatePred/entity/__init__.py",
    f"src/repoRatePred/entity/config_entity.py",
    f"src/repoRatePred/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/",
    "templates/index.html",
    "test.py"


]




for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")