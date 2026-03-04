import os
import json
import zipfile
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger("utils/data_manager.py")

class DataManager:
    @staticmethod
    def scan_directory(directory_path: Path) -> list[tuple[str, str | None]]:
        """Index all JSON files within a directory, supporting both raw files and ZIP archives.
        Args:
            directory_path (Path): The filesystem path to the directory containing 
                the data files.

        Returns:
            list[tuple[str, str | None]]: A list of descriptors for each found sample.
                Each descriptor is a tuple of (absolute_file_path, internal_zip_name).
                If the file is a standard JSON on disk, internal_zip_name is None.
        """
        file_paths = []
        logger.info(f"Scanning {directory_path} for data...")
        
        with os.scandir(directory_path) as entries:
            for entry in tqdm(entries, desc="Indexing files", leave=False):
                if entry.is_file():
                    if entry.name.endswith(".json"):
                        file_paths.append((entry.path, None))
                    elif entry.name.endswith(".zip"):
                        with zipfile.ZipFile(entry.path, "r") as z:
                            for name in z.namelist():
                                if name.endswith(".json"):
                                    file_paths.append((entry.path, name))
        return file_paths

    @staticmethod
    def load_sample(path: str, internal_name: str | None = None) -> dict:
        """A unified interface for reading JSON data from either a direct file path or a ZIP archive.
        Args:
            path (str): The absolute filesystem path to the target file or ZIP archive.
            internal_name (str | None): The name of the specific file inside the ZIP 
                archive to be read. Defaults to None for non-compressed files.

        Returns:
            dict: The parsed JSON content as a dictionary.
        """
        if internal_name:
            with zipfile.ZipFile(path, "r") as z, z.open(internal_name) as f:
                return json.load(f)
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _process_json(path_tuple: tuple[str, str | None]) -> tuple[int, int] | None:
        """Process a JSON file and return the length and maximum value.
        Args:
            path_tuple (tuple[str, str | None]): A tuple containing the absolute path 
                to the file and an optional internal filename (if the file is inside a ZIP).
            
        Returns:
            tuple[int, int] | None: A tuple containing (sequence_length, max_symbol_id) 
                if processing is successful; None if the file is malformed or an error occurs.
        """
        try:
            data = DataManager.load_sample(*path_tuple)
            
            ciphertext = data.get("ciphertext", [])
            if isinstance(ciphertext, str):
                ciphertext = [int(x) for x in ciphertext.split()]
            
            actual_max_val = max(ciphertext) if ciphertext else 0
            actual_len = len(ciphertext)
            
            return actual_len, actual_max_val
        except Exception as e:
            logger.warning(f"Skipping {path_tuple}: {e}")
            return None

    @classmethod
    def get_max_stats(cls, file_paths: list[tuple[str, str | None]]) -> tuple[int, int]:
        """Calculate dataset-wide statistics across multiple files using parallel processing.
        Args:
            file_paths (list[tuple[str, str | None]]): A list of tuples where each 
                tuple contains (absolute_path, internal_zip_name). If internal_zip_name 
                is None, the file is treated as a standard disk file.

        Returns:
            tuple[int, int]: A tuple containing (max_sequence_length, max_symbol_id) 
                found across the entire provided file list.
        """
        if not file_paths:
            raise FileNotFoundError("No files provided for analysis.")

        max_length, max_symbols, skipped_count = 0, 0, 0
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(cls._process_json, file_paths), 
                total=len(file_paths), 
                desc="Analyzing Stats"
            ))
        
        for res in results:
            if res is None:
                skipped_count += 1
                continue

            length, symbols = res
            if length > max_length:
                max_length = length
            if symbols > max_symbols:
                max_symbols = symbols

        if skipped_count > 0:
            logger.warning(
                f"\nFinished with warnings: {skipped_count} files were malformed and skipped."
            )

        logger.info(
            f"Scan complete. Max Seq Len: {max_length}, Highest Symbol ID: {max_symbols}"
        )
        return max_length, max_symbols
    
    @staticmethod
    def get_latest_checkpoint(base_path: Path) -> Path | None:
        """Searches all exp_* folders for the newest latest.pth file."""
        checkpoints = list(base_path.glob("exp_*/latest.pth"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)