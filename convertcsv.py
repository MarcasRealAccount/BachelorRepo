import argparse
import os
import sys

def convertcsvfiles(dir):
	for d in [ d for d in os.listdir(dir) if os.path.isdir(f"{dir}/{d}") ]:
		sys.stdout.write(f"\r{dir}/{d}/                                                  ")
		for f in [ f for f in os.listdir(f"{dir}/{d}/") if os.path.isfile(f"{dir}/{d}/{f}") and f.endswith(".csv") ]:
			sys.stdout.write(f"\r{dir}/{d}/{f}                                             ")
			
			with open(f"{dir}/{d}/{f}", "r+") as file:
				while True:
					c   = file.tell()
					buf = file.read(4096)
					if len(buf) == 0:
						break
					buf = buf.replace(';', ',')
					file.seek(c, os.SEEK_SET)
					file.write(buf)
	sys.stdout.write(f"\r{dir} DONE                                              \n")

def main():
	argParser = argparse.ArgumentParser(
		prog="convertcsv",
		description="Converts from ; to , for CSV files in the tests")
	argParser.add_argument("testname")
	args = argParser.parse_args()

	testname:str = args.testname
	dir          = f"tests/{testname}/"
	if not os.path.isdir(dir):
		print(f"Test '{testname}' does not exist")
		return
	
	convertcsvfiles(f"tests/{testname}/dynamic")
	convertcsvfiles(f"tests/{testname}/dynamic_raw")
	convertcsvfiles(f"tests/{testname}/static")
	convertcsvfiles(f"tests/{testname}/static_cancelled")
	convertcsvfiles(f"tests/{testname}/static_raw")
	convertcsvfiles(f"tests/{testname}/static_raw_cancelled")
	
if __name__ == "__main__":
	main()