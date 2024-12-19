from model import get_parts, get_values
from train import prepare_data_with_decomposition


def main():
    parts = get_parts()
    for part in parts:
        values = get_values(part[0])
        df_decomposed = prepare_data_with_decomposition(values)

        print(df_decomposed)


if __name__ == "__main__":
    main()
