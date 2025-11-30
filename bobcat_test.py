from lambeq import CCGBankParser
parser = CCGBankParser(root=None)
diagram = parser.sentence2diagram("This is a test sentence")
