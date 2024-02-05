import random

def generate_random_matrix_file(file_path, n, min, max):
    with open(file_path, 'w') as file:
        file.write(f"{n}\n")
        for i in range(n):
            random_values = [round(random.uniform(min, max), decimal_places) for _ in range(n)]
            line = " ".join(map(str, random_values))
            file.write(f"{line}\n")

if __name__ == "__main__":

    n = 6
    file_path = f"input_test_{n}__.txt"
    min = 0.0
    max = 1000.0
    decimal_places = 1

    generate_random_matrix_file(file_path, n, min, max)

    print(f"File '{file_path}' has been successfully generated.\n")
