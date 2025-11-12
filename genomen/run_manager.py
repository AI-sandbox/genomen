import logging
import os
import shutil
import uuid

from dotenv import load_dotenv


class RunManager:
    """Manages run directories, paths, and logging for experiments."""

    _instance = None  # class variable to store the singleton instance

    def __new__(cls):
        """Implement singleton pattern to ensure only one RunManager exists."""
        if cls._instance is None:
            cls._instance = super(RunManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the RunManager (only runs once due to singleton pattern)."""
        if self._initialized:
            return

        self.run_id = str(uuid.uuid4())[:8]
        self.base_path = None
        self.run_path = None
        self._logger = logging.getLogger(self.__class__.__name__)
        self._initialized = True

    def init_path(self, run_dir=None):
        """Initialize the path for this run."""
        load_dotenv()

        if "GENOMEN_RESULT_PATH" not in os.environ:
            raise EnvironmentError(
                "GENOMEN_RESULT_PATH environment variable is not set. "
                "Please add GENOMEN_RESULT_PATH to your .env file or environment variables."
            )

        base_path = os.path.abspath(os.environ["GENOMEN_RESULT_PATH"])

        if run_dir is None:
            run_dir = self.run_id
        else:
            run_dir = run_dir + "_" + self.run_id

        self.base_path = base_path
        self.run_path = os.path.join(base_path, run_dir)
        os.makedirs(self.run_path, exist_ok=True)

        self._logger.info(f"Initialized result path for run: {self.run_path}")
        return self.run_path

    def update_run_dir(self, new_run_dir):
        """Update the run directory with a new name, preserving all existing files.

        Args:
            new_run_dir: New name for the run directory (will still be appended with run_id)

        Returns:
            Path to the new run directory
        """
        if self.run_path is None:
            return self.init_path(new_run_dir)

        # Create new path with the new run directory name
        new_run_dir = new_run_dir + "_" + self.run_id
        new_run_path = os.path.join(self.base_path, new_run_dir)

        # Skip if the paths are identical
        if new_run_path == self.run_path:
            return self.run_path

        # Check if new directory exists and is empty
        if os.path.exists(new_run_path):
            if len(os.listdir(new_run_path)) > 0:
                self._logger.warning(f"Directory {new_run_path} already exists and is not empty.")
                return self.run_path
        else:
            os.makedirs(new_run_path, exist_ok=True)

        # Move all files from old directory to new directory
        for item in os.listdir(self.run_path):
            src = os.path.join(self.run_path, item)
            dst = os.path.join(new_run_path, item)

            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        # Remove the old directory
        shutil.rmtree(self.run_path)

        # Update run_path
        old_path = self.run_path
        self.run_path = new_run_path
        self._logger.info(f"Renamed run directory: {old_path} -> {self.run_path}")

        return self.run_path

    def get_path(self, sub_dir=None, filename=None):
        """Get a path within this run's directory."""
        if self.run_path is None:
            self.init_path()

        if sub_dir is None:
            return self.run_path

        sub_dir_path = os.path.join(self.run_path, sub_dir)
        os.makedirs(sub_dir_path, exist_ok=True)

        if filename is None:
            return sub_dir_path

        return os.path.join(sub_dir_path, filename)
