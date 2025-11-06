import glob

FILES = glob.glob("pyaf/TS/*.py") +  glob.glob("pyaf/*.py")

print(FILES)

lCopyright = None
with open("copyright.txt", "r") as lCopyFile:
    lCopyright = lCopyFile.read();

def add_copyright(fname):
    lines = None
    with open(fname, "r") as infile:
        lines = infile.readlines();
    if(len(lines) == 0):
        return
    text = "".join(lines)
    if(lines[0].startswith('# Copyright')):
        text = "".join(lines[5:])        
    with open(fname, "w") as outfile:
        outfile.write(lCopyright);
        outfile.write(text);
    
    
    

for f in FILES:
    add_copyright(f)
