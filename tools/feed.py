import os
import pickle
import pandas as pd
import zipfile
import tarfile


def data_info(data: pd.DataFrame, sorted: bool = False) -> pd.DataFrame:
    """
    Function to describe the variables of a DataFrame
    Analogous to the .describe() or .info() pandas methods
    """
    df = pd.DataFrame(pd.Series(data.columns))
    df.columns = ["columna"]
    df["NaNs"] = data.isna().sum().values
    df["pct_nan"] = round(df["NaNs"] / data.shape[0] * 100, 2)
    df["dtype"] = data.dtypes.values
    df["count"] = data.count().values
    df["count_unique"] = [
        len(data[elemento].value_counts()) for elemento in data.columns
    ]
    df["pct_unique"] = (df["count_unique"].values / data.shape[0] * 100).round(2)
    if sorted:
        df = df.reset_index(drop=False)
        df = df.sort_values(by=["dtype", "count_unique"])
    df = df.reset_index(drop=True)
    return df



def create_directory(path_output: str, verbose=True) -> None:
    # Obtener el path absoluto del directorio de salida
    path_absoluto_output = os.path.abspath(path_output)
    # Crear el directorio de salida si no existe
    if not os.path.exists(path_absoluto_output):
        if verbose:
            print(f"Se crea el directorio {path_output} en ubicación:")
            print(path_absoluto_output)
        os.makedirs(path_absoluto_output)
    else:
        if verbose:
            print(f"Directorio {path_absoluto_output} ya existe")


def enumerated_subfolder_name(path: str) -> str:
    if not os.path.exists(path):
        path += "/1"
    else:
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        max_int = float("-inf")
        for char in dirs:
            if char.isdigit():
                max_int = max(max_int, int(char))
        if max_int == float("-inf"):
            path += "/1"
        path += f"/{max_int + 1}"
    return path


def create_enumerated_subfolder(path_output: str, verbose: bool = True) -> None:
    path_output_i_N = enumerated_subfolder_name(path_output)
    if path_output_i_N.endswith("-inf"):
        path_output_i_N = path_output_i_N[:-5]
    create_directory(path_output_i_N, verbose=verbose)
    return path_output_i_N

def compress_files(input_path, output_path, type="zip"):
    """Comprime el archivo de entrada en el formato especificado (zip o tar)

    Args:
        input_path (str): Ruta del archivo a comprimir.
        output_path (str): Ruta donde se guardará el archivo comprimido.
        type (str, optional): Formato de compresión ('zip' o 'tar'). Por defecto 'zip'.
    """
    create_directory(output_path)
    if type == "zip":
        compress_path = os.path.join("backup", output_path + ".zip")
        with zipfile.ZipFile(
            compress_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as zipf:
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(
                        file_path, arcname=os.path.relpath(file_path, input_path)
                    )
    elif type == "tar":
        compress_path = os.path.join(output_path, output_path + ".tar")
        with tarfile.open(compress_path, "w") as tar:
            tar.add(input_path, arcname=os.path.basename(input_path))
    else:
        print("Didn't recognized the compression type name")


def get_folder_size(folder_path):
    "Devuelve el tamaño de la carpeta en bytes'"
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(path, file)
            total_size += os.path.getsize(file_path)
    return total_size


def bytes_to_kilobytes(bytes):
    return bytes / 1024


def get_file_size(file_path):
    """Obtiene el tamaño de un archivo en bytes"""
    return os.path.getsize(file_path)


def get_size(path):
    full_path = os.path.abspath(path)
    if os.path.isfile(full_path):
        file = "fichero"
        size = get_file_size(path)
    elif os.path.isdir(full_path):
        file = "directorio"
        size = get_folder_size(path)
    else:
        print("No existe el archivo o carpeta")
        return
    msg = """\
    Tamaño del {} "{}":
    >>  {} [KB]
    """
    print(msg.format(file, path, round(bytes_to_kilobytes(size), 2)))
    return round(bytes_to_kilobytes(size), 2)
