import os
import requests
import tarfile
import zipfile


def download_and_extract_archive(url, download_root, extract_root=None,
                                 filename=None, remove_finished=False):
    """Downloads an archive from a URL and extracts it.

    Args:
        url (str): URL of the archive to download.
        download_root (str): Directory to download the archive to.
        extract_root (str, optional): Directory to extract the archive to.
            If None, defaults to download_root.
        filename (str, optional): Name of the downloaded file. If None,
            the filename is inferred from the URL.
        remove_finished (bool, optional): If True, removes the downloaded
            archive after extraction.

    Returns:
        str: Path to the extracted directory.
    """

    if extract_root is None:
        extract_root = download_root

    if filename is None:
        filename = os.path.basename(url)
        if filename == "":  # Handle cases where basename is empty
            raise ValueError(f"Could not determine filename from URL: {url}")


    download_path = os.path.join(download_root, filename)

    if not os.path.exists(download_root):
        os.makedirs(download_root)

    # Download
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

    # Extract
    try:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(extract_root)
        elif filename.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(download_path, "r") as tar_ref:
                tar_ref.extractall(extract_root)
        else:
            print(f"Unsupported archive type: {filename}")
            return None

    except (zipfile.BadZipFile, tarfile.ReadError) as e:
        print(f"Error extracting {download_path}: {e}")
        return None

    extracted_path = os.path.join(extract_root,
                                  filename.rsplit('.', 1)[0]) # infer extracted dir name

    if remove_finished:
        os.remove(download_path)

    if os.path.isdir(extracted_path):
        return extracted_path
    else:
        # if the archive extracts to a single file, rather than a directory, return the parent directory
        return extract_root
