import zipfile
import os

csv_file_path = "time_mes.csv"
zip_file_path = "time_mes.zip"

with open(csv_file_path, "r") as csv_file:
    csv_content = csv_file.read()

# round float values to 2 decimals


def maybe_round(val):
    try:
        return str(round(float(val), 2))
    except ValueError:
        return val


csv_content = "\n".join([",".join([maybe_round(val) for val in line.split(",")]) for line in csv_content.split("\n")])

with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
    zip_file.writestr(csv_file_path, csv_content)

print(f"CSV file {csv_file_path} has been zipped and saved as {zip_file_path}")
