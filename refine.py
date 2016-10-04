import sys
import re

DELIMITER = "\t"
INNER_DELIMITER = ","

if __name__ == "__main__":
	inputFilename = sys.argv[1]
	outputFilename = sys.argv[2]

	f = open(inputFilename, "r")
	f2 = open(outputFilename, "w")

	f.readline()

	for line in f:
		if line.strip() != "":
			tokens = line.strip().split(DELIMITER)
			codes = re.sub("\"", "", tokens[-1][1:-1]).split(INNER_DELIMITER)

			if len(codes) > 10:
				f2.write(" ".join(codes) + "\n")

	f2.close()
	f.close()
