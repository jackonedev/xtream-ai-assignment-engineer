import os
import pickle
import pandas as pd
import zipfile
import tarfile


def data_info(data: pd.DataFrame, sorted: bool = False) -> pd.DataFrame:
    """
    Function to describe the variables of a dataframe
    Analogous to the .describe() method of pandas
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
    df["pct_unique"] = (df["count_unique"].values / data.shape[0] * 100).round(
        2
    )  # TODO
    if sorted:
        df = df.reset_index(drop=False)
        df = df.sort_values(by=["dtype", "count_unique"])
    df = df.reset_index(drop=True)
    return df


def procesar_file_csv(file_name: str) -> tuple:
    if file_name.endswith(".csv"):
        nombre = file_name[:-4].replace(" ", "_")
        archivo = file_name
    else:
        nombre = file_name.replace(" ", "_")
        archivo = file_name + ".csv"

    return nombre.lower(), archivo.lower()


def procesar_file_png(file_name: str) -> tuple:
    if file_name.endswith(".png"):
        nombre = file_name[:-4].replace(" ", "_")
        archivo = file_name
    else:
        nombre = file_name.replace(" ", "_")
        archivo = file_name + ".png"

    return nombre.lower(), archivo.lower()


def leer_csv(path_input: str) -> pd.DataFrame:
    """
    Lee un archivo CSV en la ruta especificada y lo retorna como un objeto DataFrame de Pandas.

    Argumentos:
    - path_input: ruta relativa o absoluta donde se encuentra el archivo CSV.

    Retorna:
    - data: DataFrame de Pandas con los datos del archivo CSV.

    """
    # Obtener el path absoluto del archivo de entrada
    path_absoluto_input = os.path.abspath(path_input)
    print("Leyendo ubicación:")
    print(path_absoluto_input)
    # Abrir el archivo de entrada
    data = None
    with open(path_absoluto_input, "r") as archivo_input:
        # Leer el archivo csv
        data = pd.read_csv(archivo_input)
    # Retornar la data
    if not isinstance(data, type(None)):
        return data


def guardar_csv(path_output: str, data: pd.DataFrame) -> None:
    # Obtener el path absoluto del archivo de salida
    path_absoluto_output = os.path.abspath(path_output)
    print("Guardando en ubicación:")
    print(path_absoluto_output)
    # Abrir el archivo de salida
    with open(path_absoluto_output, "w") as archivo_output:
        # Guardar el archivo csv
        data.to_csv(archivo_output, index=False)


def crear_directorio(path_output: str, verbose=True) -> None:
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


def nombrar_subfolder_numerada(path: str) -> str:
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


def crear_directorio_salida_numerado(path_output: str, verbose: bool = True) -> None:
    path_output_i_N = nombrar_subfolder_numerada(path_output)
    if path_output_i_N.endswith("-inf"):
        path_output_i_N = path_output_i_N[:-5]
    crear_directorio(path_output_i_N, verbose=verbose)
    return path_output_i_N


def split_text(path_input: str, encoding: str = None) -> pd.DataFrame:
    """Lee un archivo de texto y lo divide en fragmentos de texto separados por líneas en mayúsculas."""
    # crear un path absoluto a partir de path_input
    path_abs = os.path.abspath(path_input)
    # abrir el archivo en modo lectura utilizando with
    with open(path_abs, "r", encoding=encoding) as f:
        text = f.read()
    fragments = []
    current_fragment = ""
    for line in text.split("\n"):
        if line.isupper():
            # Si la línea está completamente en mayúsculas, empezamos un nuevo fragmento
            if current_fragment:
                fragments.append(current_fragment.strip())
            current_fragment = line + "\n"
        # elif line == '':
        #     continue
        else:
            # Si la línea no está en mayúsculas, agregamos la línea al fragmento actual
            current_fragment += line + "\n"
    # Agregamos el último fragmento a la lista de fragmentos
    if current_fragment:
        fragments.append(current_fragment.strip())
    # Creamos un DataFrame con los fragmentos
    splitted_fragments = [frag.split("\n") for frag in fragments]
    text_df = pd.DataFrame(splitted_fragments).transpose()
    text_df.columns = [
        "fragment_{:02d}".format(i) for i in range(1, len(text_df.columns) + 1)
    ]
    return text_df


def split_fragments_into_files(
    fragments: pd.DataFrame, path_output: str, encoding: str = None, verbose=False
) -> None:
    """Guarda los fragmentos en archivos de texto en la ruta especificada."""
    crear_directorio(path_output, verbose=verbose)
    fragment_series = fragments.apply(lambda serie: "\n".join(serie.dropna().tolist()))
    for i in range(len(fragment_series)):
        path = os.path.abspath(path_output)
        path = os.path.join(path, fragment_series.index[i] + ".txt")
        with open(path, "w") as f:
            f.write(fragment_series.values[i])


def convert_fragments_into_corpus(
    fragments: pd.DataFrame, path_output: str, encoding: str = None
) -> pd.DataFrame:
    """
    fragments: pd.DataFrame -> El nombre de las columnas corresponde al nombre de los ficheros
    path_output: str -> Path donde se encuentran los fragmentos.txt y se descargan los corpus.pkl
    encoding: str -> Codificación del texto

    return: pd.DataFrame -> Corpus de cada fragmento
    """

    crear_directorio(f"{path_output}/corpus", verbose=False)

    results = []
    for i, fragment in enumerate(fragments.columns):
        path_abs = path_output + f"/{fragment}.txt"

        with open(path_abs, "r", encoding=encoding) as f:
            text = f.read()

        text.splitlines()

        current_fragment = ""
        corpus = []
        for line in text.splitlines():
            current_fragment += line + " "
            if line == "":
                corpus.append(current_fragment.strip())
                current_fragment = ""
        if current_fragment:
            corpus.append(current_fragment.strip())

        corpus = list(filter(lambda x: x != "", corpus))
        results.append(corpus)

        with open(path_output + f"/corpus/corpus_{i+1}.pkl", "wb") as f:
            pickle.dump(pd.Series(corpus).to_frame(), f)

    df = pd.DataFrame(results).T
    df = df.set_axis(fragments.columns, axis=1)

    with open(path_output + f"/corpus/corpus_df.pkl", "wb") as f:
        pickle.dump(pd.Series(corpus).to_frame(), f)

    return df


def compress_files(input_path, output_path, type="zip"):
    """Comprime el archivo de entrada en el formato especificado (zip o tar)

    Args:
        input_path (str): Ruta del archivo a comprimir.
        output_path (str): Ruta donde se guardará el archivo comprimido.
        type (str, optional): Formato de compresión ('zip' o 'tar'). Por defecto 'zip'.
    """
    crear_directorio(output_path)
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
