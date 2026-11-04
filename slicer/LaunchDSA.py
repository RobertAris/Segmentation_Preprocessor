"""Slicer entry point used by Start DSA Workflow.command."""

import importlib.util
import os
from pathlib import Path

import qt
import slicer


def launch():
    workspace_setting = os.environ.get("DSA_WORKFLOW_ROOT")
    if workspace_setting:
        workspace = Path(workspace_setting).expanduser().resolve()
    elif "__file__" in globals():
        workspace = Path(__file__).resolve().parents[1]
    else:
        raise RuntimeError("Use Start DSA Workflow.command to open the workflow")
    spec = importlib.util.spec_from_file_location("DSAWorkflow", workspace / "slicer" / "DSAWorkflow.py")
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    prepared = os.environ.get("DSA_PREPARED_DIR")
    if not prepared:
        starting_directory = workspace / "outputs" / "reviewed-roi"
        if not starting_directory.exists():
            starting_directory = workspace / "outputs"
        prepared = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(), "Choose prepared DSA run containing manifest.json",
            str(starting_directory))
    if not prepared:
        return None
    directory = Path(prepared).expanduser().resolve()
    if directory.name == "manifest.json" and directory.is_file():
        directory = directory.parent
    try:
        session = workflow.load_prepared(directory)
        session.show_controls()
        return session
    except Exception as error:
        slicer.util.errorDisplay("Could not open the prepared DSA run:\n" + str(error))
        raise


# Slicer keeps this global available in its Python interactor for further use.
dsa_session = launch()
