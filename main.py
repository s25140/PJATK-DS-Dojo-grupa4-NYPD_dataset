import argparse

# Импорт функций
from download_data import download_data
from preprocess_data import preprocess_data
from train_model import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Scripts for this project"
    )
    parser.add_argument(
        "command",
        choices=[
            "download_data",
            "preprocess_data",
            "train_model",
        ],
        help="The command to execute.",
    )


    args = parser.parse_args()

    if args.command == "download_data":
        print('Downloading started')
        download_data()
        print("Data downloaded successfully!")
    elif args.command == "preprocess_data":
        print('Preprocess started')
        preprocess_data()
        print(f"Data preprocessed successfully! Saved to {args.output}")

    elif args.command == "train_model":
        print('Train started')
        train_model()
        print("Model trained successfully!")


if __name__ == "__main__":
    main()


