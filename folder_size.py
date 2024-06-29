import os

def get_directory_size_and_file_count(directory):
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
                file_count += 1
            except FileNotFoundError:
                # Datei nicht gefunden, überspringen
                continue
            except PermissionError:
                # Keine Berechtigung, Datei zu lesen, überspringen
                continue
    return total_size, file_count

def print_directory_sizes_and_file_counts(base_directory, output_file):
    try:
        dirnames = next(os.walk(base_directory))[1]
    except StopIteration:
        print(f"Das Verzeichnis '{base_directory}' ist leer oder existiert nicht.")
        return
    
    directory_data = []

    for dirname in dirnames:
        dir_full_path = os.path.join(base_directory, dirname)
        size_bytes, file_count = get_directory_size_and_file_count(dir_full_path)
        size_gb = size_bytes / (1024 * 1024 * 1024)  # Konvertierung in Gigabyte
        directory_data.append((dir_full_path, size_gb, file_count))
    
    # Sortieren der Verzeichnisse nach Größe in GB absteigend
    directory_data.sort(key=lambda x: x[1], reverse=True)

    with open(output_file, 'w') as f:
        for dir_full_path, size_gb, file_count in directory_data:
            result = f"{size_gb:.2f} GB in Ordner '{dir_full_path}' und es enthält {file_count} Dateien"
            print(result)
            f.write(result + "\n")

if __name__ == "__main__":
    base_directory = "/workspace"  # Basisverzeichnis festlegen
    output_file = "directory_sizes_and_file_counts.txt"  # Ausgabe-Datei festlegen
    print_directory_sizes_and_file_counts(base_directory, output_file)
