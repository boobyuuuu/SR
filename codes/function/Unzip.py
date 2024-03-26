import zipfile
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

zip_file_path = r'../../4purge.zip'
extract_to_directory = r'../../'
unzip_file(zip_file_path, extract_to_directory)