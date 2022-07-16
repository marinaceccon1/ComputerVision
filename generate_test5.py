import os

image_files = []
os.chdir("/content/data/test/")
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("/content/data/test/" + filename)
os.chdir("/content/data/")
with open("/content/data/test.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("/content/data")
